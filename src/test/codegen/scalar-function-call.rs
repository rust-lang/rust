fn foo(x: int) -> int {
    x * x
}

#[no_mangle]
fn test() {
    let x = foo(10);
}
