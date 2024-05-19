//! Check that we do sometimes reuse addresses.
use std::collections::HashSet;

fn main() {
    let count = 100;
    let mut addrs = HashSet::<usize>::new();
    for _ in 0..count {
        // We make a `Box` with a layout that's hopefully not used by tons of things inside the
        // allocator itself, so that we are more likely to get reuse. (With `i32` or `usize`, on
        // Windows the reuse chances are very low.)
        let b = Box::new([42usize; 4]);
        addrs.insert(&*b as *const [usize; 4] as usize);
    }
    // dbg!(addrs.len());
    assert!(addrs.len() > 1 && addrs.len() < count);
}
