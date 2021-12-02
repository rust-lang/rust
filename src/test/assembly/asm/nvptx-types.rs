// assembly-output: emit-asm
// compile-flags: --target nvptx64-nvidia-cuda
// compile-flags: --crate-type cdylib
// needs-llvm-components: nvptx

#![feature(no_core, lang_items, rustc_attrs, asm_sym, asm_experimental_arch)]
#![no_core]

#[rustc_builtin_macro]
macro_rules! asm {
    () => {};
}
#[rustc_builtin_macro]
macro_rules! concat {
    () => {};
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}

type ptr = *mut u8;

impl Copy for i8 {}
impl Copy for i16 {}
impl Copy for i32 {}
impl Copy for f32 {}
impl Copy for i64 {}
impl Copy for f64 {}
impl Copy for ptr {}

// NVPTX does not support static variables
#[no_mangle]
fn extern_func() {}

// CHECK-LABEL: .visible .func sym_fn()
// CHECK: // begin inline asm
// CHECK: call extern_func;
// CHECK: // end inline asm
#[no_mangle]
pub unsafe fn sym_fn() {
    asm!("call {};", sym extern_func);
}

macro_rules! check {
    ($func:ident $ty:ident $class:ident $mov:literal) => {
        #[no_mangle]
        pub unsafe fn $func(x: $ty) -> $ty {
            let y;
            asm!(concat!($mov, " {}, {};"), out($class) y, in($class) x);
            y
        }
    };
}

// CHECK-LABEL: .visible .func (.param .b32 func_retval0) reg16_i8
// CHECK: // begin inline asm
// CHECK: mov.i16 %{{[a-z0-9]+}}, %{{[a-z0-9]+}};
// CHECK: // end inline asm
check!(reg16_i8 i8 reg16 "mov.i16");

// CHECK-LABEL: .visible .func (.param .b32 func_retval0) reg16_i16
// CHECK: // begin inline asm
// CHECK: mov.i16 %{{[a-z0-9]+}}, %{{[a-z0-9]+}};
// CHECK: // end inline asm
check!(reg16_i16 i16 reg16 "mov.i16");

// CHECK-LABEL: .visible .func (.param .b32 func_retval0) reg32_i8
// CHECK: // begin inline asm
// CHECK: mov.i32 %{{[a-z0-9]+}}, %{{[a-z0-9]+}};
// CHECK: // end inline asm
check!(reg32_i8 i8 reg32 "mov.i32");

// CHECK-LABEL: .visible .func (.param .b32 func_retval0) reg32_i16
// CHECK: // begin inline asm
// CHECK: mov.i32 %{{[a-z0-9]+}}, %{{[a-z0-9]+}};
// CHECK: // end inline asm
check!(reg32_i16 i16 reg32 "mov.i32");

// CHECK-LABEL: .visible .func (.param .b32 func_retval0) reg32_i32
// CHECK: // begin inline asm
// CHECK: mov.i32 %{{[a-z0-9]+}}, %{{[a-z0-9]+}};
// CHECK: // end inline asm
check!(reg32_i32 i32 reg32 "mov.i32");

// CHECK-LABEL: .visible .func (.param .b32 func_retval0) reg32_f32
// CHECK: // begin inline asm
// CHECK: mov.i32 %{{[a-z0-9]+}}, %{{[a-z0-9]+}};
// CHECK: // end inline asm
check!(reg32_f32 f32 reg32 "mov.i32");

// CHECK-LABEL: .visible .func (.param .b32 func_retval0) reg64_i8
// CHECK: // begin inline asm
// CHECK: mov.i64 %{{[a-z0-9]+}}, %{{[a-z0-9]+}};
// CHECK: // end inline asm
check!(reg64_i8 i8 reg64 "mov.i64");

// CHECK-LABEL: .visible .func (.param .b32 func_retval0) reg64_i16
// CHECK: // begin inline asm
// CHECK: mov.i64 %{{[a-z0-9]+}}, %{{[a-z0-9]+}};
// CHECK: // end inline asm
check!(reg64_i16 i16 reg64 "mov.i64");

// CHECK-LABEL: .visible .func (.param .b32 func_retval0) reg64_i32
// CHECK: // begin inline asm
// CHECK: mov.i64 %{{[a-z0-9]+}}, %{{[a-z0-9]+}};
// CHECK: // end inline asm
check!(reg64_i32 i32 reg64 "mov.i64");

// CHECK-LABEL: .visible .func (.param .b32 func_retval0) reg64_f32
// CHECK: // begin inline asm
// CHECK: mov.i64 %{{[a-z0-9]+}}, %{{[a-z0-9]+}};
// CHECK: // end inline asm
check!(reg64_f32 f32 reg64 "mov.i64");

// CHECK-LABEL: .visible .func (.param .b64 func_retval0) reg64_i64
// CHECK: // begin inline asm
// CHECK: mov.i64 %{{[a-z0-9]+}}, %{{[a-z0-9]+}};
// CHECK: // end inline asm
check!(reg64_i64 i64 reg64 "mov.i64");

// CHECK-LABEL: .visible .func (.param .b64 func_retval0) reg64_f64
// CHECK: // begin inline asm
// CHECK: mov.i64 %{{[a-z0-9]+}}, %{{[a-z0-9]+}};
// CHECK: // end inline asm
check!(reg64_f64 f64 reg64 "mov.i64");

// CHECK-LABEL: .visible .func (.param .b64 func_retval0) reg64_ptr
// CHECK: // begin inline asm
// CHECK: mov.i64 %{{[a-z0-9]+}}, %{{[a-z0-9]+}};
// CHECK: // end inline asm
check!(reg64_ptr ptr reg64 "mov.i64");
