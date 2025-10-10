// https://github.com/rust-lang/rust/issues/92741
//@ run-rustfix
fn main() {}
fn _foo() -> bool {
    &  //~ ERROR mismatched types [E0308]
    mut
    if true { true } else { false }
}

fn _bar() -> bool {
    &  //~ ERROR mismatched types [E0308]
    mut if true { true } else { false }
}

fn _baz() -> bool {
    & mut //~ ERROR mismatched types [E0308]
    if true { true } else { false }
}
