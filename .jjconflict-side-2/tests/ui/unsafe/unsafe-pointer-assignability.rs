//@ run-pass

fn f(x: *const isize) {
    unsafe {
        assert_eq!(*x, 3);
    }
}

pub fn main() {
    f(&3);
}
