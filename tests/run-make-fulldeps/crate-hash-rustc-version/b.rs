extern crate a;

use a::foo;

fn main() {
    let x = String::from("Hello");
    println!("{}", foo(x));
}
