// run-pass
use std::fmt;

struct Foo;
impl fmt::Debug for Foo {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        println!("<Foo as Debug>::fmt()");

        write!(fmt, "")
    }
}

fn test1() {
    let foo_str = format!("{:?}", Foo);

    println!("{}", foo_str);
}

fn test2() {
    println!("{:?}", Foo);
}

fn main() {
    // This works fine
    test1();

    // This fails
    test2();
}
