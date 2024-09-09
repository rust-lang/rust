//@ known-bug: rust-lang/rust#129075
//@ compile-flags: -Zvalidate-mir -Zinline-mir=yes

struct Foo<T>([T; 2]);

impl<T: Default + Copy> Default for Foo<T> {
    fn default(&mut self) -> Self {
        Foo([Default::default(); 2])
    }
}

fn field_array() {
    let a: i32;
    let b;
    Foo([a, b]) = Default::default();
}
