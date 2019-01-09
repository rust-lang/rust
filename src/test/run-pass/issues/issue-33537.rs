// run-pass

const fn foo() -> *const i8 {
    b"foo" as *const _ as *const i8
}

const fn bar() -> i32 {
    *&{(1, 2, 3).1}
}

fn main() {
    assert_eq!(foo(), b"foo" as *const _ as *const i8);
    assert_eq!(bar(), 2);
}
