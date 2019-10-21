file_path='./training_data/'; 
str1='doubletap_'; str2='fist_'; str3='spread_'; 
str4='static_'; str5='wavein_'; str6='waveout_';
mat_format='.mat'; txt_format='.txt';
file_save_path='./training_data_txt/';

i_var=0;
for i = 1:1:42
    i_var=i_var+1;
    file1=sprintf('%s%s%d%s',file_path,str1,i,mat_format);
    file2=sprintf('%s%s%d%s',file_path,str2,i,mat_format);
    file3=sprintf('%s%s%d%s',file_path,str3,i,mat_format);
    file4=sprintf('%s%s%d%s',file_path,str4,i,mat_format);
    file5=sprintf('%s%s%d%s',file_path,str5,i,mat_format);
    file6=sprintf('%s%s%d%s',file_path,str6,i,mat_format);
    load(file1); load(file2); load(file3);%may change the vaule of variable i
    load(file4); load(file5); load(file6);%may change the vaule of variable i
    file1_var=sprintf('%s%d',str1,i_var);
    file2_var=sprintf('%s%d',str2,i_var);
    file3_var=sprintf('%s%d',str3,i_var);
    file4_var=sprintf('%s%d',str4,i_var);
    file5_var=sprintf('%s%d',str5,i_var);
    file6_var=sprintf('%s%d',str6,i_var);
    file1_var=eval(file1_var);file2_var=eval(file2_var);file3_var=eval(file3_var);
    dlmwrite(sprintf('%s%s%d%s',file_save_path,str1,i_var,txt_format),file1_var,'delimiter',' ','precision','%.2f');
    dlmwrite(sprintf('%s%s%d%s',file_save_path,str2,i_var,txt_format),file2_var,'delimiter',' ','precision','%.2f');
    dlmwrite(sprintf('%s%s%d%s',file_save_path,str3,i_var,txt_format),file3_var,'delimiter',' ','precision','%.2f');
    dlmwrite(sprintf('%s%s%d%s',file_save_path,str4,i_var,txt_format),file4_var,'delimiter',' ','precision','%.2f');
    dlmwrite(sprintf('%s%s%d%s',file_save_path,str5,i_var,txt_format),file5_var,'delimiter',' ','precision','%.2f');
    dlmwrite(sprintf('%s%s%d%s',file_save_path,str6,i_var,txt_format),file6_var,'delimiter',' ','precision','%.2f');
end