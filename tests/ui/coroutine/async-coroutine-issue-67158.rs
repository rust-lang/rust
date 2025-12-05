#![feature(coroutines)]
//@ edition:2018
// Regression test for #67158.
fn main() {
    async { yield print!(":C") }; //~ ERROR `async` coroutines are not yet supported
}
