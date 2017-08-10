#![allow(unused_variables)]

mod safe {
    pub(crate) fn safe(x: *mut u32) {
        unsafe { *x = 42; } //~ ERROR: in conflict with lock WriteLock
    }
}

fn main() {
    let target = &mut 42u32;
    let target2 = target as *mut _;
    drop(&mut *target); // reborrow
    // Now make sure we still got the lock
    safe::safe(target2);
}
