// error-pattern: mismatched types

use std;

fn main() {
    let a = [0];
    let v: *mutable [int] = ptr::mut_addr_of(a);

    fn f(&&v: *mutable [const int]) {
        unsafe {
            *v = [mutable 3]
        }
    }

    f(v);
}
