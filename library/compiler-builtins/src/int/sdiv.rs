use int::udiv::*;

macro_rules! sdivmod {
    (
        $unsigned_fn:ident, // name of the unsigned division function
        $signed_fn:ident, // name of the signed division function
        $uX:ident, // unsigned integer type for the inputs and outputs of `$unsigned_name`
        $iX:ident, // signed integer type for the inputs and outputs of `$signed_name`
        $($attr:tt),* // attributes
    ) => {
        intrinsics! {
            $(
                #[$attr]
            )*
            /// Returns `n / d` and sets `*rem = n % d`
            pub extern "C" fn $signed_fn(a: $iX, b: $iX, rem: &mut $iX) -> $iX {
                let a_neg = a < 0;
                let b_neg = b < 0;
                let mut a = a;
                let mut b = b;
                if a_neg {
                    a = a.wrapping_neg();
                }
                if b_neg {
                    b = b.wrapping_neg();
                }
                let mut r = *rem as $uX;
                let t = $unsigned_fn(a as $uX, b as $uX, Some(&mut r)) as $iX;
                let mut r = r as $iX;
                if a_neg {
                    r = r.wrapping_neg();
                }
                *rem = r;
                if a_neg != b_neg {
                    t.wrapping_neg()
                } else {
                    t
                }
            }
        }
    }
}

macro_rules! sdiv {
    (
        $unsigned_fn:ident, // name of the unsigned division function
        $signed_fn:ident, // name of the signed division function
        $uX:ident, // unsigned integer type for the inputs and outputs of `$unsigned_name`
        $iX:ident, // signed integer type for the inputs and outputs of `$signed_name`
        $($attr:tt),* // attributes
    ) => {
        intrinsics! {
            $(
                #[$attr]
            )*
            /// Returns `n / d`
            pub extern "C" fn $signed_fn(a: $iX, b: $iX) -> $iX {
                let a_neg = a < 0;
                let b_neg = b < 0;
                let mut a = a;
                let mut b = b;
                if a_neg {
                    a = a.wrapping_neg();
                }
                if b_neg {
                    b = b.wrapping_neg();
                }
                let t = $unsigned_fn(a as $uX, b as $uX) as $iX;
                if a_neg != b_neg {
                    t.wrapping_neg()
                } else {
                    t
                }
            }
        }
    }
}

macro_rules! smod {
    (
        $unsigned_fn:ident, // name of the unsigned division function
        $signed_fn:ident, // name of the signed division function
        $uX:ident, // unsigned integer type for the inputs and outputs of `$unsigned_name`
        $iX:ident, // signed integer type for the inputs and outputs of `$signed_name`
        $($attr:tt),* // attributes
    ) => {
        intrinsics! {
            $(
                #[$attr]
            )*
            /// Returns `n % d`
            pub extern "C" fn $signed_fn(a: $iX, b: $iX) -> $iX {
                let a_neg = a < 0;
                let b_neg = b < 0;
                let mut a = a;
                let mut b = b;
                if a_neg {
                    a = a.wrapping_neg();
                }
                if b_neg {
                    b = b.wrapping_neg();
                }
                let r = $unsigned_fn(a as $uX, b as $uX) as $iX;
                if a_neg {
                    r.wrapping_neg()
                } else {
                    r
                }
            }
        }
    }
}

sdivmod!(
    __udivmodsi4,
    __divmodsi4,
    u32,
    i32,
    maybe_use_optimized_c_shim
);
// The `#[arm_aeabi_alias = __aeabi_idiv]` attribute cannot be made to work with `intrinsics!` in macros
intrinsics! {
    #[maybe_use_optimized_c_shim]
    #[arm_aeabi_alias = __aeabi_idiv]
    /// Returns `n / d`
    pub extern "C" fn __divsi3(a: i32, b: i32) -> i32 {
        let a_neg = a < 0;
        let b_neg = b < 0;
        let mut a = a;
        let mut b = b;
        if a_neg {
            a = a.wrapping_neg();
        }
        if b_neg {
            b = b.wrapping_neg();
        }
        let t = __udivsi3(a as u32, b as u32) as i32;
        if a_neg != b_neg {
            t.wrapping_neg()
        } else {
            t
        }
    }
}
smod!(__umodsi3, __modsi3, u32, i32, maybe_use_optimized_c_shim);

sdivmod!(
    __udivmoddi4,
    __divmoddi4,
    u64,
    i64,
    maybe_use_optimized_c_shim
);
sdiv!(__udivdi3, __divdi3, u64, i64, maybe_use_optimized_c_shim);
smod!(__umoddi3, __moddi3, u64, i64, maybe_use_optimized_c_shim);

// LLVM does not currently have a `__divmodti4` function, but GCC does
sdivmod!(
    __udivmodti4,
    __divmodti4,
    u128,
    i128,
    maybe_use_optimized_c_shim
);
sdiv!(__udivti3, __divti3, u128, i128, win64_128bit_abi_hack);
smod!(__umodti3, __modti3, u128, i128, win64_128bit_abi_hack);
