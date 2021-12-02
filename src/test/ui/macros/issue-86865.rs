use std::fmt::Write;

fn main() {
    println!(b"foo");
    //~^ ERROR format argument must be a string literal
    //~| HELP consider removing the leading `b`
    let mut s = String::new();
    write!(s, b"foo{}", "bar");
    //~^ ERROR format argument must be a string literal
    //~| HELP consider removing the leading `b`
}
