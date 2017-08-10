#![allow(unused_variables)]

mod safe {
    // This makes a ref that was passed to us via &mut alias with things it should not alias with
    pub(crate) fn safe(x: &mut &u32, target: &mut u32) {
        unsafe { *x = &mut *(target as *mut _); }
    }
}

fn main() {
    let target = &mut 42;
    let mut target_alias = &42; // initial dummy value
    safe::safe(&mut target_alias, target); //~ ERROR: in conflict with lock ReadLock
}
