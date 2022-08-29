// check-pass

// compile-flags: --emit=mir,link -O

// There is an ICE somewhere in type normalization, and we are hitting it during
// the MIR inlining pass on this code.
//
// Long term, we should fix that ICE and change the compile-flags for this test
// to explicitly enable MIR inlining.
//
// Short term, we are diabling MIR inlining for Rust 1.64-beta, so that we avoid
// this ICE in this instance.

pub trait Trait {
    type Associated;
}
impl<T> Trait for T {
    type Associated = T;
}

pub struct Struct<T>(<T as Trait>::Associated);

pub fn foo<T>() -> Struct<T>
where
    T: Trait,
{
    bar()
}

#[inline]
fn bar<T>() -> Struct<T> {
    Struct(baz())
}

fn baz<T>() -> T {
    unimplemented!()
}

fn main() { }
