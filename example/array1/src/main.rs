use std :: vec;
fn main() {
    // 数组定义
    let  number  = [1,3,5,7,9,11,1,4];
    let sdf = number[3];
    println!("sdf is {}",sdf);
    // 创建数组的方式
    let mut v = Vec::new();
    v.push(1);
    v.push(23);
    v.push(87);
    println!("v:{:?}",v);
    let mut v2 = Vec::with_capacity(20);
    v2.push(23);
    v2.push(12);
    v2.push(25);
    v2.push(18);
    v2.push(21);
    v2.push(15);
    v2.push(27);
    v2.push(10);
    v2.push(16);
    v2.push(23);
    v2.push(12);
    println!("v2:{:?}",v2);
    v2.pop();
    println!("v2:{:?}",v2);
   // v2.remove(v2[2]);
    println!("v3:{:?}",v2);


}