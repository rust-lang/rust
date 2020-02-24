#![feature(generators)]
// edition:2018
// Regression test for #67158.
fn main() {
    async { yield print!(":C") }; //~ ERROR `async` generators are not yet supported
}
