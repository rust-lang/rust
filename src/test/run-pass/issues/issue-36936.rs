// run-pass
// check that casts are not being treated as lexprs.

fn main() {
    let mut a = 0i32;
    let b = &(a as i32);
    a = 1;
    assert_ne!(&a as *const i32, b as *const i32);
    assert_eq!(*b, 0);

    assert_eq!(issue_36936(), 1);
}


struct A(u32);

impl Drop for A {
    fn drop(&mut self) {
        self.0 = 0;
    }
}

fn issue_36936() -> u32 {
    let a = &(A(1) as A);
    a.0
}
