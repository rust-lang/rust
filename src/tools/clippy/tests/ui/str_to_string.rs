#![warn(clippy::str_to_string)]

fn main() {
    let hello = "hello world".to_string();
    //~^ str_to_string

    let msg = &hello[..];
    msg.to_string();
    //~^ str_to_string
}
