fn foo() {
    match () {
        _ if foo => (),
        _ if let foo = bar => (),
    }
}
