fn main() {
    let text = String::from("123");
    let number1 = text as i32; //~ ERROR non-primitive cast

    let slice: &str = "123";
    let number2 = slice as u16; //~ ERROR casting `&str` as `u16` is invalid

    let number3 = "8080" as i16; //~ ERROR casting `&'static str` as `i16` is invalid
    let number4 = &text as i32; //~ ERROR casting `&String` as `i32` is invalid
}