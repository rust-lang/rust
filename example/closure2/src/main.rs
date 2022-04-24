fn main() {
    let x = 1_234;
    let add_x = |a| x+a;
    let result = add_x(19);
    println!("result:{}",result);
    let mut f= String::from("lello world!");
    let  make_adder = ||{
        f.push_str("leikang");
    };
  make_adder();
 println!("cvbcbcb:{}",f);
}   
