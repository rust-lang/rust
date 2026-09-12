//@ known-bug: #142229
#![feature(super_let)]

const _: *const i32 = {
    super let x = 1;
    &raw const x
};
