//@ revisions: one two three four five six

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
//@[six] error-pattern: requested DWARF version 6 is not supported

//@ compile-flags: -g --target x86_64-unknown-linux-gnu --crate-type cdylib
//@ needs-llvm-components: x86

// This test verifies the expected behavior of various options passed
// to `-Zdwarf-version`: 1 & 6 (not supported), 2 - 5 (valid)

#![feature(no_core, lang_items)]

#![no_core]
#![no_std]

#[lang = "sized"]
pub trait Sized {}

pub fn foo() {}
