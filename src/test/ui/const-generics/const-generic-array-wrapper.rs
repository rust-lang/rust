// run-pass
// revisions: full min

#![cfg_attr(full, feature(const_generics))]
#![cfg_attr(full, allow(incomplete_features))]
#![cfg_attr(min, feature(min_const_generics))]

struct Foo<T, const N: usize>([T; N]);

impl<T, const N: usize> Foo<T, N> {
    fn foo(&self) -> usize {
        N
    }
}

fn main() {
    let foo = Foo([0u32; 21]);
    assert_eq!(foo.0, [0u32; 21]);
    assert_eq!(foo.foo(), 21);
}
