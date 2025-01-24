#![warn(clippy::all, clippy::or_fun_call)]
#![allow(clippy::unnecessary_literal_unwrap)]

fn main() {
    let s = Some(String::from("test string")).unwrap_or("Fail".to_string()).len();
    //~^ ERROR: function call inside of `unwrap_or`
    //~| NOTE: `-D clippy::or-fun-call` implied by `-D warnings`
}

fn new_lines() {
    let s = Some(String::from("test string")).unwrap_or("Fail".to_string()).len();
    //~^ ERROR: function call inside of `unwrap_or`
}
