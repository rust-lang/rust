// Test that we do some basic error correction in the tokeniser (and don't ICE).

fn main() {
    if foo {
    ) //~ ERROR: mismatched closing delimiter: `)`
}
