fn foo() {
    if true {}
    loop {}
    match () {}
    while true {}
    for _ in () {}
    {}
    {}
    macro_rules! test {
         () => {}
    }
    test!{}
}
