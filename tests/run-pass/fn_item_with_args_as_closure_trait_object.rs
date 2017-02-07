fn foo(i: i32) {
    assert_eq!(i, 42);
}

fn main() {
    let f: &Fn(i32) = &foo;
    f(42);
}
