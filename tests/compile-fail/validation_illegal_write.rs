#![allow(unused_variables)]

mod safe {
    pub(crate) fn safe(x: &u32) {
        let x : &mut u32 = unsafe { &mut *(x as *const _ as *mut _) };
        *x = 42; //~ ERROR: in conflict with lock ReadLock
    }
}

fn main() {
    let target = &mut 42;
    let target_ref = &target;
    // do a reborrow, but we keep the lock
    safe::safe(&*target);
}
