//@ run-rustfix
fn main() {
    let value = [7u8];
    while Some(0) = value.get(0) {} //~ ERROR invalid left-hand side of assignment
}
