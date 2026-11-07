// Check that this program doesn't cause the compiler to error without output.

trait Foo {
    type Assoc3<T>;
}

struct Bar;

impl Foo for Bar {
    type Assoc3<T> = Vec<T> where T: Iterator;
    //~^ ERROR impl has stricter requirements than trait
}

fn main() {}
