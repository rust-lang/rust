// error-pattern:mismatched types: expected `()` but found `bool`

fn main() {
    for vec::each([0]) {|_i|
        true
    }
}