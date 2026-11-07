//@ check-pass
//
// regression test for #98702

trait Foo {
    type Assoc<T>;
}

impl Foo for () {
    type Assoc<T> = [T; 2*2];
}

fn main() {}
