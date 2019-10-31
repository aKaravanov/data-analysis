function [xCoordinates, yCoodinates] = findCoordinates(data, leftBorder, ...
    rightBorder, timeSize, xSize, ySize)

xCoordinates = zeros(3,timeSize);
yCoodinates = zeros(3,timeSize);
counts = zeros(3,timeSize);

for i = 1:timeSize 
    for j = 1:xSize
        for k = 1:ySize
            for l = 1:3
                if data(j,k,i,l) >= 240 && k >= leftBorder(l) && k <= rightBorder(l)
                xCoordinates(l, i) = xCoordinates(l, i) + j;
                yCoodinates(l, i) = yCoodinates(l, i) + k;
                counts(l, i) = counts(l, i) + 1;
                end
            end
        end
    end
    
    for j = 1:3
        xCoordinates(j, i) = xCoordinates(j, i)/counts(j, i);
        yCoodinates(j, i) = yCoodinates(j, i)/counts(j, i);
    end
end
end