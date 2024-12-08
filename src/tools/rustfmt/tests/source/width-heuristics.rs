// rustfmt-max_width: 120

// elems on multiple lines for max_width 100, but same line for max_width 120
fn foo(e: Enum) {
    match e {
        Enum::Var {
            elem1,
            elem2,
            elem3,
        } => {
            return;
        }
    }
}

// elems not on same line for either max_width 100 or 120
fn bar(e: Enum) {
    match e {
        Enum::Var {
            elem1,
            elem2,
            elem3,
            elem4,
        } => {
            return;
        }
    }
}
