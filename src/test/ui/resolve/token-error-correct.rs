// Test that we do some basic error correction in the tokeniser.

fn main() {
    foo(bar(;
    //~^ ERROR cannot find function `bar` in this scope
}
//~^ ERROR: incorrect close delimiter: `}`

fn foo(_: usize) {}
