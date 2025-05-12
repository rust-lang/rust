// This is a regression test for #61933
// Verify that the vec![] macro may not be used in patterns
// and that the resulting diagnostic is actually helpful.

fn main() {
    match Some(vec![42]) {
        Some(vec![43]) => {} //~ ERROR expected a pattern, found a function call
        //~| ERROR found associated function
        //~| ERROR usage of qualified paths in this context is experimental
        _ => {}
    }
}
