// edition:2018

#![allow(non_camel_case_types)]

enum Foo {}

struct std;

fn main() {
    enum Foo { A, B }
    use Foo::*;
    //~^ ERROR `Foo` is ambiguous

    let _ = (A, B);

    fn std() {}
    enum std {}
    use std as foo;
    //~^ ERROR `std` is ambiguous
    //~| ERROR `std` is ambiguous
}
