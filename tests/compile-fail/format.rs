#![feature(plugin)]
#![plugin(clippy)]
#![deny(useless_format)]

fn main() {
    format!("foo"); //~ERROR useless use of `format!`
    format!("{}", 42); //~ERROR useless use of `format!`
    format!("{:?}", 42); // we only want to warn about `{}`
    format!("{:+}", 42); // we only want to warn about `{}`
    format!("foo {}", 42);
    format!("{} bar", 42);

    println!("foo");
    println!("foo {}", 42);
}
