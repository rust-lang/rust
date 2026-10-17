//@ aux-build:ctor-stability.rs
//@ check-pass

extern crate ctor_stability;

fn main() {
    let _ = ctor_stability::Foo::A;
}
