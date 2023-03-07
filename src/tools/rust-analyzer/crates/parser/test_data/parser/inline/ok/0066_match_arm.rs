fn foo() {
    match () {
        _ => (),
        _ if Test > Test{field: 0} => (),
        X | Y if Z => (),
        | X | Y if Z => (),
        | X => (),
    };
}
