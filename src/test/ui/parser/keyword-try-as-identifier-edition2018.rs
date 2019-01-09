// compile-flags: --edition 2018

fn main() {
    let try = "foo"; //~ error: expected pattern, found reserved keyword `try`
}
