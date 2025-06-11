// Test that we catch that the reference outlives the referent and we
// successfully emit a diagnostic. Regression test for issue #114714.

#![feature(generic_const_items)]
#![expect(incomplete_features)]

struct S<'a>(&'a ());

impl<'a> S<'a> {
    const K<'b>: &'a &'b () = &&(); //~ ERROR reference has a longer lifetime than the data it references
}

const Q<'a, 'b>: () = {
    let _: &'a &'b () = &&(); //~ ERROR lifetime may not live long enough
};

fn main() {}
