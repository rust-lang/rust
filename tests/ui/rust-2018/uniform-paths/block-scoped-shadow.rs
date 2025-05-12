//@ check-pass
//@ edition:2018

#![allow(non_camel_case_types)]

enum Foo {}

struct std;

fn main() {
    enum Foo { A, B }
    use Foo::*; // OK

    let _ = (A, B);

    fn std() {}
    enum std {}
    use std as foo; // OK
}
