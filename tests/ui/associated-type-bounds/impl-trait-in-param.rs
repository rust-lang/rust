//@ run-rustfix
// Issue #97583.
#![allow(unused)]

trait Super<T> {
    type Assoc;
}
trait Trait: Super<u32> + Super<i32> {}

fn foo<T: Trait<Assoc = u32>>() {} //~ ERROR ambiguous associated type `Assoc` in bounds of `Trait`
// Should suggest:
fn bar<T: Trait + Super<u32, Assoc = u32> + Super<i32, Assoc = u32>>() {}

fn main() {}
