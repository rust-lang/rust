// error-pattern: mismatched types

use std;

fn main() {
    let a = [0];
    let v: *mut [int] = ptr::mut_addr_of(a);

    fn f(&&v: *mut [const int]) {
        unsafe {
            *v = [mut 3]
        }
    }

    f(v);
}
