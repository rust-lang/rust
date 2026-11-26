/// Version of `assert_matches` that ignores fancy runtime printing in const context and uses structural equality.
#[allow(unused_macros)]
macro_rules! assert_eq_const_safe {
    ($t:ty: $left:expr, $right:expr) => {
        assert_eq_const_safe!($t: $left, $right, concat!(stringify!($left), " == ", stringify!($right)));
    };
    ($t:ty: $left:expr, $right:expr$(, $($arg:tt)+)?) => {
        {
            fn runtime() {
                assert_eq!($left, $right, $($($arg)*),*);
            }
            const fn compiletime() {
                const PAT: $t = $right;
                assert!(matches!($left, PAT), $($($arg)*),*);
            }
            core::intrinsics::const_eval_select((), compiletime, runtime)
        }
    };
}

/// Creates a test for runtime and a test for constant-time.
#[allow(unused_macros)]
macro_rules! test_runtime_and_compiletime {
    ($(
        $(#[$attr:meta])*
        fn $test:ident() $block:block
    )*) => {
        $(
            $(#[$attr])*
            #[test]
            fn $test() $block
            $(#[$attr])*
            const _: () = $block;
        )*
    }
}

/// Copied from `std::test_helpers::test_rng`, see that function for rationale.
#[track_caller]
#[allow(dead_code)] // Not used in all configurations.
pub(crate) fn test_rng() -> rand_xorshift::XorShiftRng {
    use core::hash::{BuildHasher, Hash, Hasher};
    let mut hasher = std::hash::RandomState::new().build_hasher();
    core::panic::Location::caller().hash(&mut hasher);
    let hc64 = hasher.finish();
    let seed_vec = hc64.to_le_bytes().into_iter().chain(0u8..8).collect::<Vec<u8>>();
    let seed: [u8; 16] = seed_vec.as_slice().try_into().unwrap();
    rand::SeedableRng::from_seed(seed)
}
