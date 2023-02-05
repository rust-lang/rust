fn main() {}
fn foo() -> bool {
    &  //~ ERROR 3:5: 5:36: mismatched types [E0308]
    mut
    if true { true } else { false }
}
