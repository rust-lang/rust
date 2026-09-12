//@ compile-flags: -Ctarget-feature=banana --crate-type=rlib
//@ build-pass

//~? WARN ignoring feature with missing prefix in `-Ctarget-feature`: `banana`
