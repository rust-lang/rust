#![crate_type = "lib"]

macro_rules! MyDerive { derive() {} => {} }
//~^ ERROR `macro_rules!` derives are unstable
