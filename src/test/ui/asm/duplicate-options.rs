// only-x86_64

#![feature(asm)]

fn main() {
    unsafe {
        asm!("", options(nomem, nomem));
        //~^ ERROR the `nomem` option was already provided
        //~| HELP remove this option
        asm!("", options(att_syntax, att_syntax));
        //~^ ERROR the `att_syntax` option was already provided
        //~| HELP remove this option
        asm!("", options(nostack, att_syntax), options(nostack));
        //~^ ERROR the `nostack` option was already provided
        //~| HELP remove this option
        asm!("", options(nostack, nostack), options(nostack), options(nostack));
        //~^ ERROR the `nostack` option was already provided
        //~| HELP remove this option
        //~| ERROR the `nostack` option was already provided
        //~| HELP remove this option
        //~| ERROR the `nostack` option was already provided
        //~| HELP remove this option
    }
}
