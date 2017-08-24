#![allow(unused_variables)]

// For some reason, the error location is different when using fullmir
// error-pattern: in conflict with lock WriteLock

mod safe {
    use std::slice::from_raw_parts_mut;

    pub fn as_mut_slice<T>(self_: &Vec<T>) -> &mut [T] {
        unsafe {
            from_raw_parts_mut(self_.as_ptr() as *mut T, self_.len())
        }
    }
}

fn main() {
    let v = vec![0,1,2];
    let v1_ = safe::as_mut_slice(&v);
    let v2_ = safe::as_mut_slice(&v);
}
