#![feature(decl_macro)] // so we can use pub(super)
#![feature(macro_metavar_expr_concat)]
#![cfg(all(target_arch = "aarch64", target_os = "linux", not(feature = "no-asm")))]

/// Translate a byte size to a Rust type.
macro int_ty {
    (1) => { i8 },
    (2) => { i16 },
    (4) => { i32 },
    (8) => { i64 },
    (16) => { i128 }
}

mod cas {
    pub(super) macro test($_ordering:ident, $bytes:tt, $name:ident) {
        #[test]
        fn $name() {
            builtins_test::fuzz_2(10000, |expected: super::int_ty!($bytes), new| {
                let mut target = expected.wrapping_add(10);
                assert_eq!(
                    unsafe {
                        compiler_builtins::aarch64_linux::$name::$name(expected, new, &mut target)
                    },
                    expected.wrapping_add(10),
                    "return value should always be the previous value",
                );
                assert_eq!(
                    target,
                    expected.wrapping_add(10),
                    "shouldn't have changed target"
                );

                target = expected;
                assert_eq!(
                    unsafe {
                        compiler_builtins::aarch64_linux::$name::$name(expected, new, &mut target)
                    },
                    expected
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
            builtins_test::fuzz_2(10000, |left: super::int_ty!($bytes), mut right| {
                let orig_right = right;
                assert_eq!(
                    unsafe { compiler_builtins::aarch64_linux::$name::$name(left, &mut right) },
                    orig_right
                );
                assert_eq!(left, right);
            });
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
                        builtins_test::fuzz_2(10000, |old, val| {
                            let mut target = old;
                            let op: fn(super::int_ty!($bytes), super::int_ty!($bytes)) -> _ = $($op)*;
                            let expected = op(old, val);
                            assert_eq!(old, unsafe { compiler_builtins::aarch64_linux::$name::$name(val, &mut target) }, "{} should return original value", stringify!($name));
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
use compiler_builtins::{foreach_bytes, foreach_ordering};
compiler_builtins::foreach_cas!(cas::test);
compiler_builtins::foreach_cas16!(test_cas16);
compiler_builtins::foreach_swp!(swap::test);
compiler_builtins::foreach_ldadd!(add::test);
compiler_builtins::foreach_ldclr!(clr::test);
compiler_builtins::foreach_ldeor!(xor::test);
compiler_builtins::foreach_ldset!(or::test);
