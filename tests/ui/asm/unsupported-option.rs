//@ needs-asm-support
//@ run-rustfix

use std::arch::global_asm;

fn main() {}

global_asm!("", options(nomem, readonly, noreturn, raw));
//~^ ERROR the `nomem` option cannot be used with `global_asm!`
//~| ERROR the `readonly` option cannot be used with `global_asm!`
//~| ERROR the `noreturn` option cannot be used with `global_asm!`
