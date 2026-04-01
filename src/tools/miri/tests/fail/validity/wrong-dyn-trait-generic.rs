use std::mem;

// Make sure we notice the mismatch also if the difference is "only" in the generic
// parameters of the trait.

trait Trait<T> {}
impl<T> Trait<T> for T {}

fn main() {
    let x: &dyn Trait<i32> = &0;
    let _y: *const dyn Trait<u32> = unsafe { mem::transmute(x) }; //~ERROR: wrong trait
}
