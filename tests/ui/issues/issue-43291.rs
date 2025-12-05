//@ run-pass
pub fn main() {
    assert_eq!(!0usize as *const (), foo(0, 1));
    assert_eq!(!0usize as *const (), (0i8 - 1) as *const ());
}

pub fn foo(a: i8, b: i8) -> *const () {
    (a - b) as *const ()
}
