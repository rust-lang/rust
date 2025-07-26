use alloc::vec::Vec;
use core::cell::RefCell;

use compiler_builtins::float::Float;

/// Fuzz with these many items to ensure equal functions
pub const CHECK_ITER_ITEMS: u32 = 10_000;
/// Benchmark with this many items to get a variety
pub const BENCH_ITER_ITEMS: u32 = 500;

/// Still run benchmarks/tests but don't check correctness between compiler-builtins and
/// builtin system functions functions
pub fn skip_sys_checks(test_name: &str) -> bool {
    const ALWAYS_SKIPPED: &[&str] = &[
        // FIXME(f16_f128): system symbols have incorrect results
        // <https://github.com/rust-lang/compiler-builtins/issues/617>
        "extend_f16_f32",
        "trunc_f32_f16",
        "trunc_f64_f16",
        // FIXME(#616): re-enable once fix is in nightly
        // <https://github.com/rust-lang/compiler-builtins/issues/616>
        "mul_f32",
        "mul_f64",
    ];

    // FIXME(f16_f128): error on LE ppc64. There are more tests that are cfg-ed out completely
    // in their benchmark modules due to runtime panics.
    // <https://github.com/rust-lang/compiler-builtins/issues/617#issuecomment-2125914639>
    const PPC64LE_SKIPPED: &[&str] = &["extend_f32_f128"];

    // FIXME(f16_f128): system symbols have incorrect results
    // <https://github.com/rust-lang/compiler-builtins/issues/617#issuecomment-2125914639>
    const X86_NO_SSE_SKIPPED: &[&str] = &[
        "add_f128", "sub_f128", "mul_f128", "div_f128", "powi_f32", "powi_f64",
    ];

    // FIXME(f16_f128): Wide multiply carry bug in `compiler-rt`, re-enable when nightly no longer
    // uses `compiler-rt` version.
    // <https://github.com/llvm/llvm-project/issues/91840>
    const AARCH64_SKIPPED: &[&str] = &["mul_f128", "div_f128"];

    // FIXME(llvm): system symbols have incorrect results on Windows
    // <https://github.com/rust-lang/compiler-builtins/issues/617#issuecomment-2121359807>
    const WINDOWS_SKIPPED: &[&str] = &[
        "conv_f32_u128",
        "conv_f32_i128",
        "conv_f64_u128",
        "conv_f64_i128",
    ];

    if cfg!(target_arch = "arm") {
        // The Arm symbols need a different ABI that our macro doesn't handle, just skip it
        return true;
    }

    if ALWAYS_SKIPPED.contains(&test_name) {
        return true;
    }

    if cfg!(all(target_arch = "powerpc64", target_endian = "little"))
        && PPC64LE_SKIPPED.contains(&test_name)
    {
        return true;
    }

    if cfg!(all(target_arch = "x86", not(target_feature = "sse")))
        && X86_NO_SSE_SKIPPED.contains(&test_name)
    {
        return true;
    }

    if cfg!(target_arch = "aarch64") && AARCH64_SKIPPED.contains(&test_name) {
        return true;
    }

    if cfg!(target_family = "windows") && WINDOWS_SKIPPED.contains(&test_name) {
        return true;
    }

    false
}

/// Still run benchmarks/tests but don't check correctness between compiler-builtins and
/// assembly functions
pub fn skip_asm_checks(_test_name: &str) -> bool {
    // Nothing to skip at this time
    false
}

/// Create a comparison of the system symbol, compiler_builtins, and optionally handwritten
/// assembly.
///
/// # Safety
///
/// The signature must be correct and any assembly must be sound.
#[macro_export]
macro_rules! float_bench {
    (
        // Name of this benchmark
        name: $name:ident,
        // The function signature to be tested
        sig: ($($arg:ident: $arg_ty:ty),*) -> $ret_ty:ty,
        // Path to the crate in compiler_builtins
        crate_fn: $crate_fn:path,
        // Optional alias on ppc
        $( crate_fn_ppc: $crate_fn_ppc:path, )?
        // Name of the system symbol
        sys_fn: $sys_fn:ident,
        // Optional alias on ppc
        $( sys_fn_ppc: $sys_fn_ppc:path, )?
        // Meta saying whether the system symbol is available
        sys_available: $sys_available:meta,
        // An optional function to validate the results of two functions are equal, if not
        // just `$ret_ty::check_eq`
        $( output_eq: $output_eq:expr, )?
        // Assembly implementations, if any.
        asm: [
            $(
                #[cfg($asm_meta:meta)] {
                    $($asm_tt:tt)*
                }
            );*
            $(;)?
        ]
        $(,)?
    ) => {paste::paste! {
        // SAFETY: macro invocation must use the correct signature
        #[cfg($sys_available)]
        unsafe extern "C" {
            /// Binding for the system function
            #[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
            fn $sys_fn($($arg: $arg_ty),*) -> $ret_ty;


            #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
            float_bench! { @coalesce_fn $($sys_fn_ppc)? =>
                fn $sys_fn($($arg: $arg_ty),*) -> $ret_ty;
            }
        }

        fn $name(c: &mut Criterion) {
            use core::hint::black_box;
            use compiler_builtins::float::Float;
            use $crate::bench::TestIO;

            #[inline(never)] // equalize with external calls
            fn crate_fn($($arg: $arg_ty),*) -> $ret_ty {
                #[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
                let target_crate_fn = $crate_fn;

                // On PPC, use an alias if specified
                #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
                let target_crate_fn = float_bench!(@coalesce $($crate_fn_ppc)?, $crate_fn);

                target_crate_fn( $($arg),* )
            }

            #[inline(always)] // already a branch
            #[cfg($sys_available)]
            fn sys_fn($($arg: $arg_ty),*) -> $ret_ty {
                #[cfg(not(any(target_arch = "powerpc", target_arch = "powerpc64")))]
                let target_sys_fn = $sys_fn;

                // On PPC, use an alias if specified
                #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
                let target_sys_fn = float_bench!(@coalesce $($sys_fn_ppc)?, $sys_fn);

                unsafe { target_sys_fn( $($arg),* ) }
            }

            #[inline(never)] // equalize with external calls
            #[cfg(any( $($asm_meta),* ))]
            fn asm_fn($(mut $arg: $arg_ty),*) -> $ret_ty {
                use core::arch::asm;
                $(
                    #[cfg($asm_meta)]
                    unsafe { $($asm_tt)* }
                )*
            }

            let testvec = <($($arg_ty),*)>::make_testvec($crate::bench::CHECK_ITER_ITEMS);
            let benchvec = <($($arg_ty),*)>::make_testvec($crate::bench::BENCH_ITER_ITEMS);
            let test_name = stringify!($name);
            let check_eq = float_bench!(@coalesce $($output_eq)?, $ret_ty::check_eq);

            // Verify math lines up. We run the crate functions even if we don't validate the
            // output here to make sure there are no panics or crashes.

            #[cfg($sys_available)]
            for ($($arg),*) in testvec.iter().copied() {
                let crate_res = crate_fn($($arg),*);
                let sys_res = sys_fn($($arg),*);

                if $crate::bench::skip_sys_checks(test_name) {
                    continue;
                }

                assert!(
                    check_eq(crate_res, sys_res),
                    "{test_name}{:?}: crate: {crate_res:?}, sys: {sys_res:?}",
                    ($($arg),* ,)
                );
            }

            #[cfg(any( $($asm_meta),* ))]
            {
                for ($($arg),*) in testvec.iter().copied() {
                    let crate_res = crate_fn($($arg),*);
                    let asm_res = asm_fn($($arg),*);

                    if $crate::bench::skip_asm_checks(test_name) {
                        continue;
                    }

                    assert!(
                        check_eq(crate_res, asm_res),
                        "{test_name}{:?}: crate: {crate_res:?}, asm: {asm_res:?}",
                        ($($arg),* ,)
                    );
                }
            }

            let mut group = c.benchmark_group(test_name);
            group.bench_function("compiler-builtins", |b| b.iter(|| {
                for ($($arg),*) in benchvec.iter().copied() {
                    black_box(crate_fn( $(black_box($arg)),* ));
                }
            }));

            #[cfg($sys_available)]
            group.bench_function("system", |b| b.iter(|| {
                for ($($arg),*) in benchvec.iter().copied() {
                    black_box(sys_fn( $(black_box($arg)),* ));
                }
            }));

            #[cfg(any( $($asm_meta),* ))]
            group.bench_function(&format!(
                "assembly ({} {})", std::env::consts::ARCH, std::env::consts::FAMILY
            ), |b| b.iter(|| {
                for ($($arg),*) in benchvec.iter().copied() {
                    black_box(asm_fn( $(black_box($arg)),* ));
                }
            }));

            group.finish();
        }
    }};

    // Allow overriding a default
    (@coalesce $specified:expr, $default:expr) => { $specified };
    (@coalesce, $default:expr) => { $default };

    // Allow overriding a function name
    (@coalesce_fn $specified:ident => fn $default_name:ident $($tt:tt)+) => {
        fn $specified $($tt)+
    };
    (@coalesce_fn => fn $default_name:ident $($tt:tt)+) => {
        fn $default_name $($tt)+
    };
}

/// A type used as either an input or output to/from a benchmark function.
pub trait TestIO: Sized {
    fn make_testvec(len: u32) -> Vec<Self>;
    fn check_eq(a: Self, b: Self) -> bool;
}

macro_rules! impl_testio {
    (float $($f_ty:ty),+) => {$(
        impl TestIO for $f_ty {
            fn make_testvec(len: u32) -> Vec<Self> {
                // refcell because fuzz_* takes a `Fn`
                let ret = RefCell::new(Vec::new());
                crate::fuzz_float(len, |a| ret.borrow_mut().push(a));
                ret.into_inner()
            }

            fn check_eq(a: Self, b: Self) -> bool {
                Float::eq_repr(a, b)
            }
        }

        impl TestIO for ($f_ty, $f_ty) {
            fn make_testvec(len: u32) -> Vec<Self> {
                // refcell because fuzz_* takes a `Fn`
                let ret = RefCell::new(Vec::new());
                crate::fuzz_float_2(len, |a, b| ret.borrow_mut().push((a, b)));
                ret.into_inner()
            }

            fn check_eq(_a: Self, _b: Self) -> bool {
                unimplemented!()
            }
        }
    )*};

    (int $($i_ty:ty),+) => {$(
        impl TestIO for $i_ty {
            fn make_testvec(len: u32) -> Vec<Self> {
                // refcell because fuzz_* takes a `Fn`
                let ret = RefCell::new(Vec::new());
                crate::fuzz(len, |a| ret.borrow_mut().push(a));
                ret.into_inner()
            }

            fn check_eq(a: Self, b: Self) -> bool {
                a == b
            }
        }

        impl TestIO for ($i_ty, $i_ty) {
            fn make_testvec(len: u32) -> Vec<Self> {
                // refcell because fuzz_* takes a `Fn`
                let ret = RefCell::new(Vec::new());
                crate::fuzz_2(len, |a, b| ret.borrow_mut().push((a, b)));
                ret.into_inner()
            }

            fn check_eq(_a: Self, _b: Self) -> bool {
                unimplemented!()
            }
        }
    )*};

    ((float, int) ($f_ty:ty, $i_ty:ty)) => {
        impl TestIO for ($f_ty, $i_ty) {
            fn make_testvec(len: u32) -> Vec<Self> {
                // refcell because fuzz_* takes a `Fn`
                let ivec = RefCell::new(Vec::new());
                let fvec = RefCell::new(Vec::new());

                crate::fuzz(len.isqrt(), |a| ivec.borrow_mut().push(a));
                crate::fuzz_float(len.isqrt(), |a| fvec.borrow_mut().push(a));

                let mut ret = Vec::new();
                let ivec = ivec.into_inner();
                let fvec = fvec.into_inner();

                for f in fvec {
                    for i in &ivec {
                        ret.push((f, *i));
                    }
                }

                ret
            }

            fn check_eq(_a: Self, _b: Self) -> bool {
                unimplemented!()
            }
        }
    }
}

#[cfg(f16_enabled)]
impl_testio!(float f16);
impl_testio!(float f32, f64);
#[cfg(f128_enabled)]
impl_testio!(float f128);
impl_testio!(int i8, i16, i32, i64, i128, isize);
impl_testio!(int u8, u16, u32, u64, u128, usize);
impl_testio!((float, int)(f32, i32));
impl_testio!((float, int)(f64, i32));
#[cfg(f128_enabled)]
impl_testio!((float, int)(f128, i32));
