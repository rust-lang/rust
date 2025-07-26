#[cfg(all(unix, feature = "jit"))]
use std::ffi::c_int;
#[cfg(feature = "jit")]
use std::ffi::c_void;

// FIXME replace with core::ffi::c_size_t once stabilized
#[allow(non_camel_case_types)]
#[cfg(feature = "jit")]
type size_t = usize;

macro_rules! builtin_functions {
    (
        $register:ident;
        $(
            $(#[$attr:meta])?
            fn $name:ident($($arg_name:ident: $arg_ty:ty),*) -> $ret_ty:ty;
        )*
    ) => {
        #[cfg(feature = "jit")]
        #[allow(improper_ctypes)]
        extern "C" {
            $(
                $(#[$attr])?
                fn $name($($arg_name: $arg_ty),*) -> $ret_ty;
            )*
        }

        #[cfg(feature = "jit")]
        pub(crate) fn $register(builder: &mut cranelift_jit::JITBuilder) {
            for (name, val) in [$($(#[$attr])? (stringify!($name), $name as *const u8)),*] {
                builder.symbol(name, val);
            }
        }
    };
}

builtin_functions! {
    register_functions_for_jit;

    // integers
    fn __muloti4(n: i128, d: i128, oflow: &mut i32) -> i128;
    fn __udivti3(n: u128, d: u128) -> u128;
    fn __divti3(n: i128, d: i128) -> i128;
    fn __umodti3(n: u128, d: u128) -> u128;
    fn __modti3(n: i128, d: i128) -> i128;
    fn __rust_u128_mulo(a: u128, b: u128, oflow: &mut i32) -> u128;
    fn __rust_i128_mulo(a: i128, b: i128, oflow: &mut i32) -> i128;

    // integer -> float
    fn __floattisf(i: i128) -> f32;
    fn __floattidf(i: i128) -> f64;
    fn __floatsitf(i: i32) -> f128;
    fn __floatditf(i: i64) -> f128;
    fn __floattitf(i: i128) -> f128;
    fn __floatuntisf(i: u128) -> f32;
    fn __floatuntidf(i: u128) -> f64;
    fn __floatunsitf(i: u32) -> f128;
    fn __floatunditf(i: u64) -> f128;
    fn __floatuntitf(i: u128) -> f128;
    // float -> integer
    fn __fixsfti(f: f32) -> i128;
    fn __fixdfti(f: f64) -> i128;
    fn __fixtfsi(f: f128) -> i32;
    fn __fixtfdi(f: f128) -> i64;
    fn __fixtfti(f: f128) -> i128;
    fn __fixunssfti(f: f32) -> u128;
    fn __fixunsdfti(f: f64) -> u128;
    fn __fixunstfsi(f: f128) -> u32;
    fn __fixunstfdi(f: f128) -> u64;
    fn __fixunstfti(f: f128) -> u128;
    // float -> float
    fn __extendhfsf2(f: f16) -> f32;
    fn __extendhftf2(f: f16) -> f128;
    fn __extendsftf2(f: f32) -> f128;
    fn __extenddftf2(f: f64) -> f128;
    fn __trunctfdf2(f: f128) -> f64;
    fn __trunctfsf2(f: f128) -> f32;
    fn __trunctfhf2(f: f128) -> f16;
    fn __truncdfhf2(f: f64) -> f16;
    fn __truncsfhf2(f: f32) -> f16;
    // float binops
    fn __addtf3(a: f128, b: f128) -> f128;
    fn __subtf3(a: f128, b: f128) -> f128;
    fn __multf3(a: f128, b: f128) -> f128;
    fn __divtf3(a: f128, b: f128) -> f128;
    fn fmodf(a: f32, b: f32) -> f32;
    fn fmod(a: f64, b: f64) -> f64;
    fn fmodf128(a: f128, b: f128) -> f128;
    // float comparison
    fn __eqtf2(a: f128, b: f128) -> i32;
    fn __netf2(a: f128, b: f128) -> i32;
    fn __lttf2(a: f128, b: f128) -> i32;
    fn __letf2(a: f128, b: f128) -> i32;
    fn __gttf2(a: f128, b: f128) -> i32;
    fn __getf2(a: f128, b: f128) -> i32;
    fn fminimumf128(a: f128, b: f128) -> f128;
    fn fmaximumf128(a: f128, b: f128) -> f128;
    // Cranelift float libcalls
    fn fmaf(a: f32, b: f32, c: f32) -> f32;
    fn fma(a: f64, b: f64, c: f64) -> f64;
    fn floorf(f: f32) -> f32;
    fn floor(f: f64) -> f64;
    fn ceilf(f: f32) -> f32;
    fn ceil(f: f64) -> f64;
    fn truncf(f: f32) -> f32;
    fn trunc(f: f64) -> f64;
    fn nearbyintf(f: f32) -> f32;
    fn nearbyint(f: f64) -> f64;
    // float intrinsics
    fn __powisf2(a: f32, b: i32) -> f32;
    fn __powidf2(a: f64, b: i32) -> f64;
    // FIXME(f16_f128): `compiler-builtins` doesn't currently support `__powitf2` on MSVC.
    // fn __powitf2(a: f128, b: i32) -> f128;
    fn powf(a: f32, b: f32) -> f32;
    fn pow(a: f64, b: f64) -> f64;
    fn expf(f: f32) -> f32;
    fn exp(f: f64) -> f64;
    fn exp2f(f: f32) -> f32;
    fn exp2(f: f64) -> f64;
    fn logf(f: f32) -> f32;
    fn log(f: f64) -> f64;
    fn log2f(f: f32) -> f32;
    fn log2(f: f64) -> f64;
    fn log10f(f: f32) -> f32;
    fn log10(f: f64) -> f64;
    fn sinf(f: f32) -> f32;
    fn sin(f: f64) -> f64;
    fn cosf(f: f32) -> f32;
    fn cos(f: f64) -> f64;
    fn fmaf128(a: f128, b: f128, c: f128) -> f128;
    fn floorf16(f: f16) -> f16;
    fn floorf128(f: f128) -> f128;
    fn ceilf16(f: f16) -> f16;
    fn ceilf128(f: f128) -> f128;
    fn truncf16(f: f16) -> f16;
    fn truncf128(f: f128) -> f128;
    fn rintf16(f: f16) -> f16;
    fn rintf128(f: f128) -> f128;
    fn sqrtf16(f: f16) -> f16;
    fn sqrtf128(f: f128) -> f128;
    // FIXME(f16_f128): Add other float intrinsics as compiler-builtins gains support (meaning they
    // are available on all targets).

    // allocator
    // NOTE: These need to be mentioned here despite not being part of compiler_builtins because
    // newer glibc resolve dlsym("malloc") to libc.so despite the override in the rustc binary to
    // use jemalloc. Libraries opened with dlopen still get the jemalloc version, causing multiple
    // allocators to be mixed, resulting in a crash.
    fn calloc(nobj: size_t, size: size_t) -> *mut c_void;
    #[cfg(unix)]
    fn posix_memalign(memptr: *mut *mut c_void, align: size_t, size: size_t) -> c_int;
    fn malloc(size: size_t) -> *mut c_void;
    fn realloc(p: *mut c_void, size: size_t) -> *mut c_void;
    fn free(p: *mut c_void) -> ();
}
