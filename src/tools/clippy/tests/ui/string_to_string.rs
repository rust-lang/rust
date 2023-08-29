#![warn(clippy::string_to_string)]
#![allow(clippy::redundant_clone)]

fn main() {
    let mut message = String::from("Hello");
    let mut v = message.to_string();
    //~^ ERROR: `to_string()` called on a `String`
}
