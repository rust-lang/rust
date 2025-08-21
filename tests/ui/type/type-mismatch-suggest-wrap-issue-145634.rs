// We should not suggest wrapping rhs with `Some`
// when the found type is an unresolved type variable
//
// See https://github.com/rust-lang/rust/issues/145634

fn main() {
    let foo = Some(&(1, 2));
    assert!(matches!(foo, &Some((1, 2)))); //~ ERROR mismatched types [E0308]
}
