// Regression test for the ICE described in issue #91334.

#![feature(coroutines)]

//~vv ERROR mismatched closing delimiter: `)`
//~v ERROR this file contains an unclosed delimiter
fn f(){||yield(((){),
