// only-x86_64
// build-pass

#![feature(asm)]

fn main() {
    unsafe {
        asm!("", options(nomem, nomem));
        //~^ WARNING the `nomem` option was already provided
        asm!("", options(att_syntax, att_syntax));
        //~^ WARNING the `att_syntax` option was already provided
        asm!("", options(nostack, att_syntax), options(nostack));
        //~^ WARNING the `nostack` option was already provided
        asm!("", options(nostack, nostack), options(nostack), options(nostack));
        //~^ WARNING the `nostack` option was already provided
        //~| WARNING the `nostack` option was already provided
        //~| WARNING the `nostack` option was already provided
    }
}
