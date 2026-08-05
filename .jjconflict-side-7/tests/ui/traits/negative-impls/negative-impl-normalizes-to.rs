//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@ check-pass

// Check that negative impls for traits with associated types
// do not result in an ICE when trying to normalize.
#![feature(negative_impls)]
trait Trait {
    type Assoc;
}

struct Local<T>(T);
impl !Trait for Local<u32> {}
impl Trait for Local<i32> {
    type Assoc = i32;
}

trait NoOverlap {}
impl<T: Trait<Assoc = u32>> NoOverlap for T {}
impl<T> NoOverlap for Local<T> {}

fn main() {}
