#![crate_type = "lib"]

pub extern "fictional" fn lol() {} //~ ERROR: invalid ABI
