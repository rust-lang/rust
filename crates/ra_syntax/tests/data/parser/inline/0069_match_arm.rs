fn foo() {
    match () {
        _ => (),
        X | Y if Z => (),
        | X | Y if Z => (),
        | X => (),
    };
}
