use std;

fn main() {
    unsafe fn f(&&v: *const int) {
        *v = 1 //! ERROR assigning to dereference of const pointer
    }

    unsafe {
        let a = 0;
        let v = ptr::mut_addr_of(a);
        f(v);
    }
}
