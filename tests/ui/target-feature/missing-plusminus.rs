//@ compile-flags: -Ctarget-feature=banana --crate-type=rlib
//@ build-pass

//~? WARN unknown feature specified for `-Ctarget-feature`: `banana`
