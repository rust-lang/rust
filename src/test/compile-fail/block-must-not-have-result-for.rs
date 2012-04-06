// error-pattern:mismatched types: expected `()` but found `bool`

fn main() {
    for vec::iter([0]) {|_i|
        true
    }
}