fn foo(i: i32, j: i32) {
    assert_eq!(i, 42);
    assert_eq!(j, 55);
}

fn bar(i: i32, j: i32, k: f32) {
    assert_eq!(i, 42);
    assert_eq!(j, 55);
    assert_eq!(k, 3.14159)
}


fn main() {
    let f: &Fn(i32, i32) = &foo;
    f(42, 55);
    let f: &Fn(i32, i32, f32) = &bar;
    f(42, 55, 3.14159);
}
