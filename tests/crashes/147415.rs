//@ known-bug: rust-lang/rust#147415
#![feature(min_generic_const_args)]

fn foo<T>() {
    [0; size_of::<*mut T>()]
}
