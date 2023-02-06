fn main() {}
fn foo() -> bool {
    &  //~ ERROR 3:5: 5:36: mismatched types [E0308]
    mut
    if true { true } else { false }
}

fn bar() -> bool {
    &  //~ ERROR 9:5: 10:40: mismatched types [E0308]
    mut if true { true } else { false }
}

fn baz() -> bool {
    & mut //~ ERROR 14:5: 15:36: mismatched types [E0308]
    if true { true } else { false }
}
