// Test that we do some basic error correction in the tokeniser.

fn main() {
    foo(bar(;
}
//~^ ERROR: mismatched closing delimiter: `}`

fn foo(_: usize) {}
