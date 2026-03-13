//@check-pass
//@ revisions: next old
//@[next] compile-flags: -Znext-solver

#![feature(try_as_dyn)]

use std::any::try_as_dyn;

trait HasAssoc<'a> {
    type Assoc;
}
struct Dummy;
impl<'a> HasAssoc<'a> for Dummy {
    // Changing this to &'a i64 makes try_as_dyn succeed
    type Assoc = &'static i64;
}

trait Trait {}
impl Trait for i32 where for<'a> Dummy: HasAssoc<'a, Assoc = &'a i64> {}

const _: () = {
    let x = 1i32;
    assert!(try_as_dyn::<_, dyn Trait>(&x).is_none());
};

fn main() {}
