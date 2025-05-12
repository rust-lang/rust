// Test that we can recover from missing braces in the parser.

trait Foo {
    fn bar() {
        let x = foo();
}

fn main() {
    let x = y.;
} //~ ERROR this file contains an unclosed delimiter
