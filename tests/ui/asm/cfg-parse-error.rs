//@ needs-asm-support
#![feature(asm_cfg)]

use std::arch::asm;

fn main() {
    unsafe {
        asm!(
            "",
            #[cfg(false)]
            clobber_abi("C"),
            #[cfg(false)]
            options(att_syntax),
            #[cfg(false)]
            a = out(reg) x,
            "",
            //~^ ERROR expected one of `#`, `clobber_abi`, `const`, `in`, `inlateout`, `inout`, `label`, `lateout`, `options`, `out`, or `sym`, found `""`
        );
        asm!(
            #[cfg(false)]
            "",
            #[cfg(false)]
            const {
                5
            },
            "",
            //~^ ERROR expected one of `#`, `clobber_abi`, `const`, `in`, `inlateout`, `inout`, `label`, `lateout`, `options`, `out`, or `sym`, found `""`
        );

        asm!(
            #[cfg_attr(true, cfg(false))]
            const {
                5
            },
            "",
        );

        // This is not accepted because `a = out(reg) x` is not a valid expression.
        asm!(
            #[cfg(false)]
            a = out(reg) x, //~ ERROR expected token: `,`
            "",
        );

        // For now, any non-cfg attributes are rejected
        asm!(
            #[rustfmt::skip] //~ ERROR this attribute is not supported on assembly
            "",
        );

        // For now, any non-cfg attributes are rejected
        asm!(
            #![rustfmt::skip] //~ ERROR an inner attribute is not permitted in this context
            "",
        );
    }
}
