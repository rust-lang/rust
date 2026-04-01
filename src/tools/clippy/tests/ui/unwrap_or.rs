#![warn(clippy::or_fun_call)]
#![allow(clippy::unnecessary_literal_unwrap)]

fn main() {
    let s = Some(String::from("test string")).unwrap_or("Fail".to_string()).len();
    //~^ or_fun_call
}

fn new_lines() {
    let s = Some(String::from("test string")).unwrap_or("Fail".to_string()).len();
    //~^ or_fun_call
}
