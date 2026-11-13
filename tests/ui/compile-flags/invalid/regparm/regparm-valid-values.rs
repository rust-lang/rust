//@ revisions: regparm0 regparm1 regparm2 regparm3 regparm4

//@ needs-llvm-components: x86
//@ compile-flags: --target i686-unknown-linux-gnu

//@[regparm0] check-pass
//@[regparm0] compile-flags: -Zregparm=0

//@[regparm1] check-pass
//@[regparm1] compile-flags: -Zregparm=1

//@[regparm2] check-pass
//@[regparm2] compile-flags: -Zregparm=2

//@[regparm3] check-pass
//@[regparm3] compile-flags: -Zregparm=3

//@[regparm4] check-fail
//@[regparm4] compile-flags: -Zregparm=4

#![feature(no_core)]
#![no_core]
#![no_main]

//[regparm4]~? ERROR `-Zregparm=4` is unsupported (valid values 0-3)
