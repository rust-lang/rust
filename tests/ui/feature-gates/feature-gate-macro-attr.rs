#![crate_type = "lib"]

macro_rules! myattr { attr() {} => {} }
//~^ ERROR `macro_rules!` attributes are unstable
