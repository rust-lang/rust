// compile-flags: -Z continue-parse-after-error

// Test that the parser is error correcting missing idents. Despite a parsing
// error (or two), we still run type checking (and don't get extra errors there).

fn main() {
    let y = 42;
    let x = y.;  //~ ERROR unexpected token
    let x = y.();  //~ ERROR unexpected token
                   //~^ ERROR expected function, found `{integer}`
    let x = y.foo; //~ ERROR `{integer}` is a primitive type and therefore doesn't have fields [E061
}
