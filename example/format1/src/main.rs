fn main() {
format!("Hello, world!");
format!("Hello");                 // => "Hello"
format!("Hello, {}!", "world");   // => "Hello, world!"
format!("The number is {}", 1);   // => "The number is 1"
format!("{:?}", (3, 4));          // => "(3, 4)"
format!("{value}", value=4);      // => "4"
format!("{} {}", 1, 2);           // => "1 2"
format!("{:04}", 42);             // => "0042" with leading zeros
format!("{:#?}", (100, 200));  
let  func:fn(i32,i32) ->i32 = add;
#[warn(dead_code)]
fn leikang() -> !{
    panic!("this is leikang");
}
}
fn add(x:i32,y:i32) ->i32 {
    x +y
}
