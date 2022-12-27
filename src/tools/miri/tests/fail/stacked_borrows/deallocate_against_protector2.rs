//@error-pattern: /deallocating while item \[SharedReadWrite for .*\] is strongly protected/
use std::alloc::{dealloc, Layout};
use std::marker::PhantomPinned;

pub struct NotUnpin(i32, PhantomPinned);

fn inner(x: &mut NotUnpin, f: fn(&mut NotUnpin)) {
    // `f` may mutate, but it may not deallocate!
    f(x)
}

fn main() {
    inner(Box::leak(Box::new(NotUnpin(0, PhantomPinned))), |x| {
        let raw = x as *mut NotUnpin as *mut u8;
        drop(unsafe { dealloc(raw, Layout::new::<NotUnpin>()) });
    });
}
