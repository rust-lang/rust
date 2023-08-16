//@error-in-other-file: cannot mix `proc-macro` crate type with others

// force-host
// no-prefer-dynamic

#![crate_type = "proc-macro"]
#![crate_type = "rlib"]
