// Test that we do some basic error correction in the tokeniser.

fn main() {
    foo(bar(;
    //~^ ERROR: expected expression, found `;`
}
//~^ ERROR: incorrect close delimiter: `}`
