// only-x86_64
// run-rustfix

#![feature(asm, global_asm)]

fn main() {
    unsafe {
        asm!("", options(nomem, nomem));
        //~^ ERROR the `nomem` option was already provided
        asm!("", options(att_syntax, att_syntax));
        //~^ ERROR the `att_syntax` option was already provided
        asm!("", options(nostack, att_syntax), options(nostack));
        //~^ ERROR the `nostack` option was already provided
        asm!("", options(nostack, nostack), options(nostack), options(nostack));
        //~^ ERROR the `nostack` option was already provided
        //~| ERROR the `nostack` option was already provided
        //~| ERROR the `nostack` option was already provided
        asm!(
            "",
            options(nomem, noreturn),
            options(att_syntax, noreturn), //~ ERROR the `noreturn` option was already provided
            options(nomem, nostack), //~ ERROR the `nomem` option was already provided
            options(noreturn), //~ ERROR the `noreturn` option was already provided
        );
    }
}

global_asm!("", options(att_syntax, att_syntax));
//~^ ERROR the `att_syntax` option was already provided
