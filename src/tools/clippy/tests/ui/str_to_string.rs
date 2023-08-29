#![warn(clippy::str_to_string)]

fn main() {
    let hello = "hello world".to_string();
    //~^ ERROR: `to_string()` called on a `&str`
    let msg = &hello[..];
    msg.to_string();
    //~^ ERROR: `to_string()` called on a `&str`
}
