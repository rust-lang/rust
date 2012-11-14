// error-pattern:mismatched types: expected `()` but found `bool`

struct r {}

impl r : Drop {
    fn finalize() {
        true
    }
}

fn main() {
}
