//@ revisions: regparm0 regparm1 regparm2 regparm3 regparm4

//@ needs-llvm-components: x86
//@ compile-flags: --target i686-unknown-linux-gnu -Zunstable-options

//@[regparm0] check-pass
//@[regparm0] compile-flags: -Tregparm=0

//@[regparm1] check-pass
//@[regparm1] compile-flags: -Tregparm=1

//@[regparm2] check-pass
//@[regparm2] compile-flags: -Tregparm=2

//@[regparm3] check-pass
//@[regparm3] compile-flags: -Tregparm=3

//@[regparm4] check-fail
//@[regparm4] compile-flags: -Tregparm=4
//@ ignore-backends: gcc

#![feature(no_core)]
#![no_core]
#![no_main]

//[regparm4]~? ERROR `-Tregparm=4` is unsupported (valid values 0-3)
