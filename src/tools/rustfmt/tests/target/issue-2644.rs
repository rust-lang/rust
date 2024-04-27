// rustfmt-max_width: 80
fn foo(e: Enum) {
    match e {
        Enum::Var { element1, element2 } => {
            return;
        }
    }
}
