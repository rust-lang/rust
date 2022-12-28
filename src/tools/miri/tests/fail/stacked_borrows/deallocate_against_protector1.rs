//@error-pattern: /deallocating while item \[Unique for .*\] is protected/
use std::alloc::{dealloc, Layout};

fn inner(x: &mut i32, f: fn(&mut i32)) {
    // `f` may mutate, but it may not deallocate!
    f(x)
}

fn main() {
    inner(Box::leak(Box::new(0)), |x| {
        let raw = x as *mut i32 as *mut u8;
        drop(unsafe { dealloc(raw, Layout::new::<i32>()) });
    });
}
