//@ known-bug: #137084
#![feature(min_generic_const_args)]
fn a<const b: i32>() {}
fn d(e: &String) {
    a::<d>
}
