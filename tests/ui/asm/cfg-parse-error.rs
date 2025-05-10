//@ needs-asm-support
#![feature(asm_cfg)]

use std::arch::asm;

fn main() {
    unsafe {
        // Templates are not allowed after operands (even if the operands are configured out).
        asm!(
            "",
            #[cfg(false)]
            clobber_abi("C"),
            #[cfg(false)]
            options(att_syntax),
            #[cfg(false)]
            a = out(reg) x,
            "",
            //~^ ERROR expected one of `clobber_abi`, `const`
        );
        asm!(
            #[cfg(false)]
            "",
            #[cfg(false)]
            const {
                5
            },
            "", //~ ERROR expected one of `clobber_abi`, `const`
        );

        // This is currently accepted because `const { 5 }` parses as an expression.
        asm!(
            #[cfg(false)]
            const {
                5
            },
            "",
        );
        // This is not accepted because `a = out(reg) x` is not a valid expresion.
        asm!(
            #[cfg(false)]
            a = out(reg) x, //~ ERROR expected token: `,`
            "",
        );
    }
}
