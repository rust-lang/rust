//@ only-x86_64

#![feature(asm_unwind, asm_goto)]

use std::arch::{asm, global_asm};

fn main() {
    let mut foo = 0;
    unsafe {
        asm!("", options(nomem, readonly));
        //~^ ERROR the `nomem` and `readonly` options are mutually exclusive
        asm!("", options(pure, nomem, noreturn));
        //~^ ERROR the `pure` and `noreturn` options are mutually exclusive
        //~^^ ERROR asm with the `pure` option must have at least one output
        asm!("{}", in(reg) foo, options(pure, nomem));
        //~^ ERROR asm with the `pure` option must have at least one output
        asm!("{}", out(reg) foo, options(noreturn));
        //~^ ERROR asm outputs are not allowed with the `noreturn` option
        asm!("{}", label {}, options(may_unwind));
        //~^ ERROR asm labels are not allowed with the `may_unwind` option
    }

    unsafe {
        asm!("", clobber_abi("foo"));
        //~^ ERROR invalid ABI for `clobber_abi`
        asm!("{}", out(reg) foo, clobber_abi("C"));
        //~^ ERROR asm with `clobber_abi` must specify explicit registers for outputs
        asm!("{}", out(reg) foo, clobber_abi("C"), clobber_abi("C"));
        //~^ ERROR asm with `clobber_abi` must specify explicit registers for outputs
        //~| ERROR `C` ABI specified multiple times
        asm!("", out("eax") foo, clobber_abi("C"));
    }
}

global_asm!("", options(nomem));
//~^ ERROR the `nomem` option cannot be used with `global_asm!`
global_asm!("", options(readonly));
//~^ ERROR the `readonly` option cannot be used with `global_asm!`
global_asm!("", options(noreturn));
//~^ ERROR the `noreturn` option cannot be used with `global_asm!`
global_asm!("", options(pure));
//~^ ERROR the `pure` option cannot be used with `global_asm!`
global_asm!("", options(nostack));
//~^ ERROR the `nostack` option cannot be used with `global_asm!`
global_asm!("", options(preserves_flags));
//~^ ERROR the `preserves_flags` option cannot be used with `global_asm!`
