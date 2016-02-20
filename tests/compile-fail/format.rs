#![feature(plugin)]
#![plugin(clippy)]
#![deny(useless_format)]

fn main() {
    format!("foo"); //~ERROR useless use of `format!`
    format!("foo {}", 42);

    println!("foo");
    println!("foo {}", 42);
}
