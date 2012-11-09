// aux-build:pub_use_xcrate1.rs
// aux-build:pub_use_xcrate2.rs

extern mod pub_use_xcrate2;

use pub_use_xcrate2::Foo;

fn main() {
    let foo: Foo = Foo {
        name: 0
    };
}

