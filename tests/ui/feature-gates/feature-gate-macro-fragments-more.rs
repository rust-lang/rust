#![crate_type = "lib"]

macro_rules! use_fn { ($f:fn) => {} }
//~^ ERROR macro `:fn` and `:adt` fragments are unstable
