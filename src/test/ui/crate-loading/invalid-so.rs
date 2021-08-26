// compile-flags: --crate-type lib --extern bar={{src-base}}/crate-loading/auxiliary/libbar.so
// edition:2018
use ::bar; //~ ERROR invalid metadata files for crate `bar`
//~| NOTE invalid metadata version
