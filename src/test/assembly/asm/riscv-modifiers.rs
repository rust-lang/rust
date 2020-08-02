// no-system-llvm
// assembly-output: emit-asm
// compile-flags: -O
// compile-flags: --target riscv64gc-unknown-linux-gnu
// compile-flags: -C target-feature=+f
// needs-llvm-components: riscv

#![feature(no_core, lang_items, rustc_attrs)]
#![crate_type = "rlib"]
#![no_core]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}
#[rustc_builtin_macro]
macro_rules! concat {
    () => {};
}
#[rustc_builtin_macro]
macro_rules! stringify {
    () => {};
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

impl Copy for f32 {}

macro_rules! check {
    ($func:ident $modifier:literal $reg:ident $mov:literal) => {
        // -O and extern "C" guarantee that the selected register is always r0/s0/d0/q0
        #[no_mangle]
        pub unsafe extern "C" fn $func() -> f32 {
            // Hack to avoid function merging
            extern "Rust" {
                fn dont_merge(s: &str);
            }
            dont_merge(stringify!($func));

            let y;
            asm!(concat!($mov, " {0:", $modifier, "}, {0:", $modifier, "}"), out($reg) y);
            y
        }
    };
}

// CHECK-LABEL: reg:
// CHECK: #APP
// CHECK: mv a0, a0
// CHECK: #NO_APP
check!(reg "" reg "mv");

// CHECK-LABEL: freg:
// CHECK: #APP
// CHECK: fmv.s fa0, fa0
// CHECK: #NO_APP
check!(freg "" freg "fmv.s");
