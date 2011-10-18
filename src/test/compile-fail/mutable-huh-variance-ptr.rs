// error-pattern: mismatched types

use std;

fn main() {
    let a = [0];
    let v: *mutable [int] = std::ptr::addr_of(a);

    fn f(&&v: *mutable [mutable? int]) {
        unsafe {
            *v = [mutable 3]
        }
    }

    f(v);
}
