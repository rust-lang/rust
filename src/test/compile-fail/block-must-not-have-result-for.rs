// error-pattern:mismatched types: expected `()` but found `bool`

fn main() {
    for i in [0] {
        true
    }
}