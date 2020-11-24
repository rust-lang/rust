// run-pass
#![feature(in_band_lifetimes, const_generics)]
#![allow(incomplete_features)]

struct Foo<T>(T);

impl Foo<[u8; { let _: &'a (); 3 }]> {
    fn test() -> Self {
        Foo([0; 3])
    }
}

fn main() {
    assert_eq!(Foo::test().0, [0; 3]);
}
