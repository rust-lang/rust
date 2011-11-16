// error-pattern: assigning to immutable box

use std;

fn main() {
    unsafe fn f(&&v: *const int) {
        // This shouldn't be possible
        *v = 1
    }

    unsafe {
        let a = 0;
        let v = std::ptr::mut_addr_of(a);
        f(v);
    }
}
