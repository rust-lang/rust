// Test that we do some basic error correction in the tokeniser (and don't spew
// too many bogus errors).

fn foo() -> usize {
    3
}

fn bar() {
    foo() //~ ERROR mismatched types
}

fn main() {
    bar()
}
