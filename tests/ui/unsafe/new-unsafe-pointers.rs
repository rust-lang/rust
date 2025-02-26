//@ run-pass

fn main() {
    let _a: *const isize = 3 as *const isize;
    let _a: *mut isize = 3 as *mut isize;
}
