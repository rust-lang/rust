// This test verifies the expected behavior of various options passed to
// `-Zdwarf-version`: 2 - 5 (valid) with all other options being invalid.

//@ revisions: zero one two three four five six

//@[zero] compile-flags: -Zdwarf-version=0

//@[one] compile-flags: -Zdwarf-version=1
//@[one] error-pattern: requested DWARF version 1 is not supported

//@[two] compile-flags: -Zdwarf-version=2
//@[two] check-pass

//@[three] compile-flags: -Zdwarf-version=3
//@[three] check-pass

//@[four] compile-flags: -Zdwarf-version=4
//@[four] check-pass

//@[five] compile-flags: -Zdwarf-version=5
//@[five] check-pass

//@[six] compile-flags: -Zdwarf-version=6

//@ compile-flags: -g --target x86_64-unknown-linux-gnu --crate-type cdylib
//@ needs-llvm-components: x86

#![feature(no_core, lang_items)]

#![no_core]
#![no_std]

#[lang = "sized"]
pub trait Sized {}

pub fn foo() {}

//[zero]~? ERROR requested DWARF version 0 is not supported
//[one]~? ERROR requested DWARF version 1 is not supported
//[six]~? ERROR requested DWARF version 6 is not supported
