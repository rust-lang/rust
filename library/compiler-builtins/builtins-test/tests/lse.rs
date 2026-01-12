#![feature(decl_macro)] // so we can use pub(super)
#![feature(macro_metavar_expr_concat)]
#![cfg(target_arch = "aarch64")]

use std::sync::Mutex;

use compiler_builtins::aarch64_outline_atomics::{get_have_lse_atomics, set_have_lse_atomics};
use compiler_builtins::int::{Int, MinInt};
use compiler_builtins::{foreach_bytes, foreach_ordering};

#[track_caller]
fn with_maybe_lse_atomics(use_lse: bool, f: impl FnOnce()) {
    // Ensure tests run in parallel don't interleave global settings
    static LOCK: Mutex<()> = Mutex::new(());
    let _g = LOCK.lock().unwrap();
    let old = get_have_lse_atomics();
    // safety: as the caller of the unsafe fn `set_have_lse_atomics`, we
    // have to ensure the CPU supports LSE. This is why we make this assertion.
    if use_lse || old {
        assert!(std::arch::is_aarch64_feature_detected!("lse"));
    }
    unsafe { set_have_lse_atomics(use_lse) };
    f();
    unsafe { set_have_lse_atomics(old) };
}

pub fn run_fuzz_tests_with_lse_variants<I: Int, F: Fn(I, I) + Copy>(n: u32, f: F)
where
    <I as MinInt>::Unsigned: Int,
{
    // We use `fuzz_2` because our subject function `f` requires two inputs
    let test_fn = || {
        builtins_test::fuzz_2(n, f);
    };
    // Always run without LSE
    with_maybe_lse_atomics(false, test_fn);

    // Conditionally run with LSE
    if std::arch::is_aarch64_feature_detected!("lse") {
        with_maybe_lse_atomics(true, test_fn);
    }
}

/// Translate a byte size to a Rust type.
macro int_ty {
    (1) => { u8 },
    (2) => { u16 },
    (4) => { u32 },
    (8) => { u64 },
    (16) => { u128 }
}

mod cas {
    pub(super) macro test($_ordering:ident, $bytes:tt, $name:ident) {
        #[test]
        fn $name() {
            crate::run_fuzz_tests_with_lse_variants(10000, |expected: super::int_ty!($bytes), new| {
                let mut target = expected.wrapping_add(10);
                let ret: super::int_ty!($bytes) = unsafe {
                    compiler_builtins::aarch64_outline_atomics::$name::$name(
                        expected,
                        new,
                        &mut target,
                    )
                };
                assert_eq!(
                    ret,
                    expected.wrapping_add(10),
                    "return value should always be the previous value",
                );
                assert_eq!(
                    target,
                    expected.wrapping_add(10),
                    "shouldn't have changed target"
                );

                target = expected;
                let ret: super::int_ty!($bytes) = unsafe {
                    compiler_builtins::aarch64_outline_atomics::$name::$name(
                        expected,
                        new,
                        &mut target,
                    )
                };
                assert_eq!(
                    ret,
                    expected,
                    "the new return value should always be the previous value (i.e. the first parameter passed to the function)",
                );
                assert_eq!(target, new, "should have updated target");
            });
        }
    }
}

macro test_cas16($_ordering:ident, $name:ident) {
    cas::test!($_ordering, 16, $name);
}

mod swap {
    pub(super) macro test($_ordering:ident, $bytes:tt, $name:ident) {
        #[test]
        fn $name() {
            crate::run_fuzz_tests_with_lse_variants(
                10000,
                |left: super::int_ty!($bytes), mut right| {
                    let orig_right = right;
                    assert_eq!(
                        unsafe {
                            compiler_builtins::aarch64_outline_atomics::$name::$name(
                                left, &mut right,
                            )
                        },
                        orig_right
                    );
                    assert_eq!(left, right);
                },
            );
        }
    }
}

macro_rules! test_op {
    ($mod:ident, $( $op:tt )* ) => {
        mod $mod {
            pub(super) macro test {
                ($_ordering:ident, $bytes:tt, $name:ident) => {
                    #[test]
                    fn $name() {
                        crate::run_fuzz_tests_with_lse_variants(10000, |old, val| {
                            let mut target = old;
                            let op: fn(super::int_ty!($bytes), super::int_ty!($bytes)) -> _ = $($op)*;
                            let expected = op(old, val);
                            assert_eq!(old, unsafe { compiler_builtins::aarch64_outline_atomics::$name::$name(val, &mut target) }, "{} should return original value", stringify!($name));
                            assert_eq!(expected, target, "{} should store to target", stringify!($name));
                        });
                    }
                }
            }
        }
    };
}

test_op!(add, |left, right| left.wrapping_add(right));
test_op!(clr, |left, right| left & !right);
test_op!(xor, std::ops::BitXor::bitxor);
test_op!(or, std::ops::BitOr::bitor);
compiler_builtins::foreach_cas!(cas::test);
compiler_builtins::foreach_cas16!(test_cas16);
compiler_builtins::foreach_swp!(swap::test);
compiler_builtins::foreach_ldadd!(add::test);
compiler_builtins::foreach_ldclr!(clr::test);
compiler_builtins::foreach_ldeor!(xor::test);
compiler_builtins::foreach_ldset!(or::test);
