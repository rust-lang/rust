// only-aarch64
// run-rustfix

#![feature(asm, global_asm)]

fn main() {
    unsafe {
        asm!("", options(nomem, nomem));
        //~^ ERROR the `nomem` option was already provided
        asm!("", options(preserves_flags, preserves_flags));
        //~^ ERROR the `preserves_flags` option was already provided
        asm!("", options(nostack, preserves_flags), options(nostack));
        //~^ ERROR the `nostack` option was already provided
        asm!("", options(nostack, nostack), options(nostack), options(nostack));
        //~^ ERROR the `nostack` option was already provided
        //~| ERROR the `nostack` option was already provided
        //~| ERROR the `nostack` option was already provided
        asm!(
            "",
            options(nomem, noreturn),
            options(preserves_flags, noreturn), //~ ERROR the `noreturn` option was already provided
            options(nomem, nostack), //~ ERROR the `nomem` option was already provided
            options(noreturn), //~ ERROR the `noreturn` option was already provided
        );
    }
}
