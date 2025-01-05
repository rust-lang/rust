//@ revisions: good_val bad_val no_val
//
//@[good_val] compile-flags: -Cinline-threshold=666
//@[good_val] check-pass
//@[bad_val] compile-flags: -Cinline-threshold=asd
//@[no_val] compile-flags: -Cinline-threshold

fn main() {}
