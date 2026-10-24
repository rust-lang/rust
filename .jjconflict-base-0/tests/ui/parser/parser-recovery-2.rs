// Test that we can recover from mismatched braces in the parser.

trait Foo {
    fn bar() {
        let x = foo();
    ) //~ ERROR mismatched closing delimiter: `)`
}

fn main() {
    let x = y.;
}
