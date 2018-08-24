// edition:2018

#![feature(uniform_paths)]

enum Foo { A, B }

struct std;

fn main() {
    enum Foo {}
    use Foo::*;
    //~^ ERROR `Foo` import is ambiguous

    let _ = (A, B);

    fn std() {}
    enum std {}
    use std as foo;
    //~^ ERROR `std` import is ambiguous
    //~| ERROR `std` import is ambiguous
}
