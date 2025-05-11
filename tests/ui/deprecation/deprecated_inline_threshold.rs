//@ revisions: good_val bad_val no_val
//
//@[good_val] compile-flags: -Cinline-threshold=666
//@[good_val] check-pass
//@[bad_val] compile-flags: -Cinline-threshold=asd
//@[no_val] compile-flags: -Cinline-threshold

fn main() {}

//[good_val]~? WARN `-C inline-threshold`: this option is deprecated and does nothing
//[bad_val]~? WARN `-C inline-threshold`: this option is deprecated and does nothing
//[bad_val]~? ERROR incorrect value `asd` for codegen option `inline-threshold`
//[no_val]~? WARN `-C inline-threshold`: this option is deprecated and does nothing
//[no_val]~? ERROR codegen option `inline-threshold` requires a number
