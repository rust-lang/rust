//! Regression test for an unclosed delimiter whose block begins with `&&`/`||`
//! should hint that the user may have meant to continue an if-let chain.
fn main() {
    if let Some(x) = Some(42) {
        && x == 42
    {
    }
} //~ ERROR this file contains an unclosed delimiter
