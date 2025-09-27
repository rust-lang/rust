#![crate_type = "lib"]

macro_rules! use_fn { ($f:fn) => {} }
//~^ ERROR macro `:fn` and `:adt` fragments are unstable

macro_rules! use_adt { ($f:adt) => {} }
//~^ ERROR macro `:fn` and `:adt` fragments are unstable
