use std::fmt::{self, Display};

fn main() {
    let a = Foo;

    if a.to_string() != "bar" {
        //~^ cmp_owned
        println!("foo");
    }

    if "bar" != a.to_string() {
        //~^ cmp_owned
        println!("foo");
    }
}

struct Foo;

impl Display for Foo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "foo")
    }
}

impl PartialEq<&str> for Foo {
    fn eq(&self, other: &&str) -> bool {
        "foo" == *other
    }
}
