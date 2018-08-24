// compile-flags: -Z continue-parse-after-error

// Test that we can recover from mismatched braces in the parser.

trait Foo {
    fn bar() {
        let x = foo(); //~ ERROR cannot find function `foo` in this scope
    ) //~ ERROR incorrect close delimiter: `)`
}

fn main() {
    let x = y.;  //~ ERROR unexpected token
                 //~^ ERROR cannot find value `y` in this scope
}
