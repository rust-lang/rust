// Test that the `'a` from the impl doesn't
// prevent us from creating a `'a` parameter
// on the `blah` function.
//
// check-pass

#![feature(in_band_lifetimes)]

struct Foo<'a> {
    x: &'a u32

}

impl Foo<'a> {
    fn method(&self) {
        fn blah(f: Foo<'a>) { }
    }
}

fn main() { }
