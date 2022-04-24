fn main() {
    // 闭包的使用
    //let add = |a:i32,b:i32| ->i32{
        let add =|a,b|{
              a+b
            };
    let x = add(1,3);
    println!("x:{}",x);
    let add2 = |a1,b2,c3,d4|{
        a1+b2+c3-d4
    };
    let y = add2(9,7,12,3);
    println!("y:{}",y);
}
