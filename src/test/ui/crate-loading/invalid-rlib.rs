// compile-flags: --crate-type lib --extern foo={{src-base}}/crate-loading/auxiliary/libfoo.rlib
// edition:2018
#![no_std]
use ::foo; //~ ERROR invalid metadata files for crate `foo`
//~| NOTE memory map must have a non-zero length
