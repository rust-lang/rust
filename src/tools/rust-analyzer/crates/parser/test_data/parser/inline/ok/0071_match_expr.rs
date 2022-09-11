fn foo() {
    match () { };
    match S {};
    match { } { _ => () };
    match { S {} } {};
}
