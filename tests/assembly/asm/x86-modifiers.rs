// revisions: x86_64 i686
// assembly-output: emit-asm
// compile-flags: -O
//[x86_64] compile-flags: --target x86_64-unknown-linux-gnu
//[x86_64] needs-llvm-components: x86
//[i686] compile-flags: --target i686-unknown-linux-gnu
//[i686] needs-llvm-components: x86
// compile-flags: -C llvm-args=--x86-asm-syntax=intel
// compile-flags: -C target-feature=+avx512bw

#![feature(no_core, lang_items, rustc_attrs)]
#![crate_type = "rlib"]
#![no_core]
#![allow(asm_sub_register)]

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

impl Copy for i32 {}

macro_rules! check {
    ($func:ident $modifier:literal $reg:ident $mov:literal) => {
        // -O and extern "C" guarantee that the selected register is always ax/xmm0
        #[no_mangle]
        pub unsafe extern "C" fn $func() -> i32 {
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
// x86_64: mov rax, rax
// i686: mov eax, eax
// CHECK: #NO_APP
check!(reg "" reg "mov");

// x86_64-LABEL: reg_l:
// x86_64: #APP
// x86_64: mov al, al
// x86_64: #NO_APP
#[cfg(x86_64)]
check!(reg_l "l" reg "mov");

// CHECK-LABEL: reg_x:
// CHECK: #APP
// CHECK: mov ax, ax
// CHECK: #NO_APP
check!(reg_x "x" reg "mov");

// CHECK-LABEL: reg_e:
// CHECK: #APP
// CHECK: mov eax, eax
// CHECK: #NO_APP
check!(reg_e "e" reg "mov");

// x86_64-LABEL: reg_r:
// x86_64: #APP
// x86_64: mov rax, rax
// x86_64: #NO_APP
#[cfg(x86_64)]
check!(reg_r "r" reg "mov");

// CHECK-LABEL: reg_abcd:
// CHECK: #APP
// x86_64: mov rax, rax
// i686: mov eax, eax
// CHECK: #NO_APP
check!(reg_abcd "" reg_abcd "mov");

// CHECK-LABEL: reg_abcd_l:
// CHECK: #APP
// CHECK: mov al, al
// CHECK: #NO_APP
check!(reg_abcd_l "l" reg_abcd "mov");

// CHECK-LABEL: reg_abcd_h:
// CHECK: #APP
// CHECK: mov ah, ah
// CHECK: #NO_APP
check!(reg_abcd_h "h" reg_abcd "mov");

// CHECK-LABEL: reg_abcd_x:
// CHECK: #APP
// CHECK: mov ax, ax
// CHECK: #NO_APP
check!(reg_abcd_x "x" reg_abcd "mov");

// CHECK-LABEL: reg_abcd_e:
// CHECK: #APP
// CHECK: mov eax, eax
// CHECK: #NO_APP
check!(reg_abcd_e "e" reg_abcd "mov");

// x86_64-LABEL: reg_abcd_r:
// x86_64: #APP
// x86_64: mov rax, rax
// x86_64: #NO_APP
#[cfg(x86_64)]
check!(reg_abcd_r "r" reg_abcd "mov");

// CHECK-LABEL: xmm_reg
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check!(xmm_reg "" xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_x
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check!(xmm_reg_x "x" xmm_reg "movaps");

// CHECK-LABEL: xmm_reg_y
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check!(xmm_reg_y "y" xmm_reg "vmovaps");

// CHECK-LABEL: xmm_reg_z
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check!(xmm_reg_z "z" xmm_reg "vmovaps");

// CHECK-LABEL: ymm_reg
// CHECK: #APP
// CHECK: movaps ymm0, ymm0
// CHECK: #NO_APP
check!(ymm_reg "" ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_x
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check!(ymm_reg_x "x" ymm_reg "movaps");

// CHECK-LABEL: ymm_reg_y
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check!(ymm_reg_y "y" ymm_reg "vmovaps");

// CHECK-LABEL: ymm_reg_z
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check!(ymm_reg_z "z" ymm_reg "vmovaps");

// CHECK-LABEL: zmm_reg
// CHECK: #APP
// CHECK: movaps zmm0, zmm0
// CHECK: #NO_APP
check!(zmm_reg "" zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_x
// CHECK: #APP
// CHECK: movaps xmm0, xmm0
// CHECK: #NO_APP
check!(zmm_reg_x "x" zmm_reg "movaps");

// CHECK-LABEL: zmm_reg_y
// CHECK: #APP
// CHECK: vmovaps ymm0, ymm0
// CHECK: #NO_APP
check!(zmm_reg_y "y" zmm_reg "vmovaps");

// CHECK-LABEL: zmm_reg_z
// CHECK: #APP
// CHECK: vmovaps zmm0, zmm0
// CHECK: #NO_APP
check!(zmm_reg_z "z" zmm_reg "vmovaps");

// Note: we don't have any way of ensuring that k1 is actually the register
// chosen by the register allocator, so this check may fail if a different
// register is chosen.

// CHECK-LABEL: kreg:
// CHECK: #APP
// CHECK: kmovb k1, k1
// CHECK: #NO_APP
check!(kreg "" kreg "kmovb");
