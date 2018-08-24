// aux-build:lint_output_format.rs

#![feature(unstable_test_feature)]
#![feature(rustc_attrs)]

extern crate lint_output_format;
use lint_output_format::{foo, bar};
//~^ WARNING use of deprecated item 'lint_output_format::foo': text

#[rustc_error]
fn main() { //~ ERROR: compilation successful
    let _x = foo();
    //~^ WARNING use of deprecated item 'lint_output_format::foo': text
    let _y = bar();
}
