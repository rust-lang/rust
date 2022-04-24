// 结构体代码实现
struct Student{
    // 属性功能字段，name有生命周期static
    name: &'static str,
    socre :i32
}
// 定义结构体三个类型对于应的变量
struct Color(i32,i32,i32);
fn main() {
    // 初始化结构体，传入实参数
    let b_b = Color(23,45,78);
    // 直接给结构体初始化，给name,socre赋予值
let socre =43;
let name = "lk";

// 结构体实力话
let mut s  = Student{
    name,
    socre
};
// s.socre = 98;
// 引用name,socre
let s1 = s.name;
let s2 = s.socre;
 println!("name:{},socre:{}",s1,s2)
}
