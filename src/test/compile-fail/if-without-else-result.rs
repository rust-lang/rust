// error-pattern:mismatched types: expected `()` but found `bool`

fn main() {
    let a = if true { true };
    log(debug, a);
}