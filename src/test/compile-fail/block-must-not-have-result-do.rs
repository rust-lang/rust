// error-pattern:mismatched types: expected `()` but found `bool`

fn main() {
    do {
        true
    } while true;
}