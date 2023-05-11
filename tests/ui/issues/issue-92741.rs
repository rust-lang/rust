// run-rustfix
fn main() {}
fn _foo() -> bool {
    &  //~ ERROR 4:5: 6:36: mismatched types [E0308]
    mut
    if true { true } else { false }
}

fn _bar() -> bool {
    &  //~ ERROR 10:5: 11:40: mismatched types [E0308]
    mut if true { true } else { false }
}

fn _baz() -> bool {
    & mut //~ ERROR 15:5: 16:36: mismatched types [E0308]
    if true { true } else { false }
}
