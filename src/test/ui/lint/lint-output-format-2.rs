// aux-build:lint_output_format.rs

#![feature(unstable_test_feature)]
// build-pass (FIXME(62277): could be check-pass?)

extern crate lint_output_format;
use lint_output_format::{foo, bar};
//~^ WARNING use of deprecated item 'lint_output_format::foo': text


fn main() {
    let _x = foo();
    //~^ WARNING use of deprecated item 'lint_output_format::foo': text
    let _y = bar();
}
