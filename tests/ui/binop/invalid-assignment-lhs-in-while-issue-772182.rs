//@ run-rustfix
fn main() {
    let value = [7u8];
    while Some(0) = value.get(0) { //~ ERROR invalid left-hand side of assignment
    }
}

// https://github.com/rust-lang/rust/issues/772182
