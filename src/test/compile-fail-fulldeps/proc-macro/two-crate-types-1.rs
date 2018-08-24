// error-pattern: cannot mix `proc-macro` crate type with others

#![crate_type = "proc-macro"]
#![crate_type = "rlib"]
