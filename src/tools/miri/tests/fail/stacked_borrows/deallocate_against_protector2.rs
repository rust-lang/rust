//@error-pattern: /deallocating while item \[SharedReadWrite for .*\] is protected/
use std::marker::PhantomPinned;

pub struct NotUnpin(i32, PhantomPinned);

fn inner(x: &mut NotUnpin, f: fn(&mut NotUnpin)) {
    // `f` may mutate, but it may not deallocate!
    f(x)
}

fn main() {
    inner(Box::leak(Box::new(NotUnpin(0, PhantomPinned))), |x| {
        let raw = x as *mut _;
        drop(unsafe { Box::from_raw(raw) });
    });
}
