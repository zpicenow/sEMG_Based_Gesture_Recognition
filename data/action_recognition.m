clear;
clc;
%% import data
load('raw_data2.mat');
for i = 1:1:30
    a1 = 'f_static_'; a2 = 'f_fist_'; a3 = 'f_spread_'; b='.mat';
    if i<=10
       s=sprintf('%s%d%s',a1,i,b);
       load(s);
    else if (i>10 && i<=20)
            ii = i-10;
            s=sprintf('%s%d%s',a2,ii,b);
            load(s);
        else
            iii = i-20;
            s=sprintf('%s%d%s',a3,iii,b);
            load(s);
        end
    end
end
raw_data = raw_data2;
[m,n]=size(raw_data);
nc = 10;
nt = 50;
epsilon = 2.5;
i = 0;
%% data procressing
for j=1:n
    for k=1:nc:(m-nt)
        i = i + 1;
        sum = 0;
        for a=0:1:nt
              sum = sum + raw_data(k+a,j)^2;
              F_RMS_1 = sqrt(sum/a);
		end	
       %% RMS  signal energy 
        F_RMS_2(i) = F_RMS_1;
       %% WL  signal complexity
        sum = 0;
        for a=0:1:nt-1
            sum = sum + abs(raw_data(k+a,j)-raw_data(k+a+1,j));
        end
        F_WL_1 = sum;
        F_WL_2(i) = F_WL_1;
       %% WA  potential motion ability of muscle
        sum = 0;
        for a=0:1:nt-1
            sum = sum + comp(abs(raw_data(k+a,j)-raw_data(k+a+1,j)),epsilon);
        end
        F_WA_1 = sum;
        F_WA_2(i) = F_WA_1;
    end
A = F_RMS_2';
B = F_WL_2';
C = F_WA_2';    
end
new_row = (m-nt)/nc;
F_RMS = reshape(A,new_row,n); F_WL = reshape(B,new_row,n); F_WA = reshape(C,new_row,n);
feature_data = [];
for a=1:1:new_row		    
    feature_data = [feature_data;F_RMS(a,:);F_WL(a,:);F_WA(a,:)];
end
raw_feature = floor(feature_data/400);
[m,n]=size(raw_feature);
k = 3;
num = 0;

for i = 1:k:(m-k+1)
    num = num + 1; 
    test = raw_feature(i:(i+k-1),:);
    
 %   var1 = abs(test - floor(f_static_1)); var2 = abs(test - floor(f_static_2)); var3 = abs(test - floor(f_static_3)); var4 = abs(test - floor(f_static_4)); var5 = abs(test - floor(f_static_5));
 %   var6 = abs(test - floor(f_static_6)); var7 = abs(test - floor(f_static_7)); var8 = abs(test - floor(f_static_8)); var9 = abs(test - floor(f_static_9)); var10 = abs(test - floor(f_static_10));
 %   var11 = abs(test - floor(f_fist_1)); var12 = abs(test - floor(f_fist_2)); var13 = abs(test - floor(f_fist_3)); var14 = abs(test - floor(f_fist_4)); var15 = abs(test - floor(f_fist_5));
 %   var16 = abs(test - floor(f_fist_6)); var17 = abs(test - floor(f_fist_7)); var18 = abs(test - floor(f_fist_8)); var19 = abs(test - floor(f_fist_9)); var20 = abs(test - floor(f_fist_10)); 
 %   var21 = abs(test - floor(f_spread_1)); var22 = abs(test - floor(f_spread_2)); var23 = abs(test - floor(f_spread_3)); var24 = abs(test - floor(f_spread_4)); var25 = abs(test - floor(f_spread_5));
 %   var26 = abs(test - floor(f_spread_6)); var27 = abs(test - floor(f_spread_7)); var28 = abs(test - floor(f_spread_8)); var29 = abs(test - floor(f_spread_9)); var30 = abs(test - floor(f_spread_10)); 
    
    var1 = abs(test - floor(f_static_1/400)); var2 = abs(test - floor(f_static_2/400)); var3 = abs(test - floor(f_static_3/400)); var4 = abs(test - floor(f_static_4/400)); var5 = abs(test - floor(f_static_5/400));
    var6 = abs(test - floor(f_static_6/400)); var7 = abs(test - floor(f_static_7/400)); var8 = abs(test - floor(f_static_8/400)); var9 = abs(test - floor(f_static_9/400)); var10 = abs(test - floor(f_static_10/400));
    var11 = abs(test - floor(f_fist_1/400)); var12 = abs(test - floor(f_fist_2/400)); var13 = abs(test - floor(f_fist_3/400)); var14 = abs(test - floor(f_fist_4/400)); var15 = abs(test - floor(f_fist_5/400));
    var16 = abs(test - floor(f_fist_6/400)); var17 = abs(test - floor(f_fist_7/400)); var18 = abs(test - floor(f_fist_8/400)); var19 = abs(test - floor(f_fist_9/400)); var20 = abs(test - floor(f_fist_10/400)); 
    var21 = abs(test - floor(f_spread_1/400)); var22 = abs(test - floor(f_spread_2/400)); var23 = abs(test - floor(f_spread_3/400)); var24 = abs(test - floor(f_spread_4/400)); var25 = abs(test - floor(f_spread_5/400));
    var26 = abs(test - floor(f_spread_6/400)); var27 = abs(test - floor(f_spread_7/400)); var28 = abs(test - floor(f_spread_8/400)); var29 = abs(test - floor(f_spread_9/400)); var30 = abs(test - floor(f_spread_10/400)); 
    
    
    sum1 = 0;sum2 = 0;sum3 = 0;sum4 = 0;sum5 = 0;sum6 = 0;sum7 = 0;sum8 = 0;sum9 = 0;sum10 = 0;
    sum11 = 0;sum12 = 0;sum13 = 0;sum14 = 0;sum15 = 0;sum16 = 0;sum17 = 0;sum18 = 0;sum19 = 0;sum20 = 0;
    sum21 = 0;sum22 = 0;sum23 = 0;sum24 = 0;sum25 = 0;sum26 = 0;sum27 = 0;sum28 = 0;sum29 = 0;sum30 = 0;
    
    for p=1:1:3
        for q=1:1:8
            sum1 = sum1 + var1(p,q); sum2 = sum2 + var2(p,q); sum3 = sum3 + var3(p,q); sum4 = sum4 + var4(p,q); sum5 = sum5 + var5(p,q);
            sum6 = sum6 + var6(p,q); sum7 = sum7 + var7(p,q); sum8 = sum8 + var8(p,q); sum9 = sum9 + var9(p,q); sum10 = sum10 + var10(p,q);
            sum11 = sum11 + var11(p,q); sum12 = sum12 + var12(p,q); sum13 = sum13 + var13(p,q); sum14 = sum14 + var14(p,q); sum15 = sum15 + var15(p,q);
            sum16 = sum16 + var16(p,q); sum17 = sum17 + var17(p,q); sum18 = sum18 + var18(p,q); sum19 = sum19 + var19(p,q); sum20 = sum20 + var20(p,q);
            sum21 = sum21 + var21(p,q); sum22 = sum22 + var22(p,q); sum23 = sum23 + var3(p,q); sum24 = sum24 + var4(p,q); sum25 = sum25 + var25(p,q);
            sum26 = sum26 + var26(p,q); sum27 = sum27 + var27(p,q); sum28 = sum28 + var8(p,q); sum29 = sum29 + var9(p,q); sum30 = sum30 + var30(p,q);
        end
    end
    
     bijiao(1) = sum1;bijiao(2) = sum2;bijiao(3) = sum3;bijiao(4) = sum4; bijiao(5) = sum5;bijiao(6) = sum6;bijiao(7) = sum7;bijiao(8) = sum8; bijiao(9) = sum9;bijiao(10) = sum10;
     bijiao(11) = sum11;bijiao(12) = sum12;bijiao(13) = sum13;bijiao(14) = sum14; bijiao(15) = sum15;bijiao(16) = sum16;bijiao(17) = sum17;bijiao(18) = sum18;bijiao(19) = sum19;bijiao(20) = sum20;
     bijiao(21) = sum21;bijiao(22) = sum22;bijiao(23) = sum23;bijiao(24) = sum24; bijiao(25) = sum25;bijiao(26) = sum26;bijiao(27) = sum27;bijiao(28) = sum28; bijiao(29) = sum29;bijiao(30) = sum30;
     
     final_result = reshape(bijiao,10,3);
     t=sort(final_result(:));
     [m,n]=find(final_result<=t(3),3);
     table = tabulate(n');
     zhunque = max(table(:,3));
     [maxCount,idx] = max(table(:,2));
     table(idx);
         
      switch idx      
      case 1
         state = 'Static';
      case 2
         state = 'Fist';
      case 3
         state = 'Spread';
      end
      disp(['Time ' num2str(num) '  ' state '  Percentage  ' num2str(zhunque)]);      
     
end 
    








