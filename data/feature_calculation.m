%% data static_1 fist_1
load('spread_10.mat');
raw_data = spread_10;
[m,n]=size(raw_data);
nt = m;
epsilon = 2.5;
i = 0;
% data procressing
for j=1:n
i = i + 1;
        sum = 0;
        for a= 1:1:nt
                sum = sum + raw_data(a,j)^2;
        end
       %% RMS  signal energy 
        F_RMS_1 = sqrt(sum/a);
        F_RMS_2(i) = F_RMS_1;
       %% WL  signal complexity
        sum = 0;
        for a=1:1:nt-1
            sum = sum + abs(raw_data(a,j)-raw_data(a+1,j));
        end
        F_WL_1 = sum;
        F_WL_2(i) = F_WL_1;
       %% WA  potential motion ability of muscle
        sum = 0;
        for a=1:1:nt-1
            sum = sum + comp(abs(raw_data(a,j)-raw_data(a+1,j)),epsilon);
        end
        F_WA_1 = sum;
        F_WA_2(i) = F_WA_1;
end
feature_data = [F_RMS_2;F_WL_2;F_WA_2];
f_spread_10 = feature_data;
clc;