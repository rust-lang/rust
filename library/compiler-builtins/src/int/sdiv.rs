use int::Int;

intrinsics! {
    #[maybe_use_optimized_c_shim]
    #[arm_aeabi_alias = __aeabi_idiv]
    /// Returns `n / d`
    pub extern "C" fn __divsi3(a: i32, b: i32) -> i32 {
        i32_div_rem(a, b).0
    }

    #[maybe_use_optimized_c_shim]
    /// Returns `n % d`
    pub extern "C" fn __modsi3(a: i32, b: i32) -> i32 {
        i32_div_rem(a, b).1
    }

    #[maybe_use_optimized_c_shim]
    /// Returns `n / d` and sets `*rem = n % d`
    pub extern "C" fn __divmodsi4(a: i32, b: i32, rem: &mut i32) -> i32 {
        let quo_rem = i32_div_rem(a, b);
        *rem = quo_rem.1;
        quo_rem.0
    }

    #[maybe_use_optimized_c_shim]
    /// Returns `n / d`
    pub extern "C" fn __divdi3(a: i64, b: i64) -> i64 {
        i64_div_rem(a, b).0
    }

    #[maybe_use_optimized_c_shim]
    /// Returns `n % d`
    pub extern "C" fn __moddi3(a: i64, b: i64) -> i64 {
        i64_div_rem(a, b).1
    }

    #[maybe_use_optimized_c_shim]
    /// Returns `n / d` and sets `*rem = n % d`
    pub extern "C" fn __divmoddi4(a: i64, b: i64, rem: &mut i64) -> i64 {
        let quo_rem = i64_div_rem(a, b);
        *rem = quo_rem.1;
        quo_rem.0
    }

    #[win64_128bit_abi_hack]
    /// Returns `n / d`
    pub extern "C" fn __divti3(a: i128, b: i128) -> i128 {
        i128_div_rem(a, b).0
    }

    #[win64_128bit_abi_hack]
    /// Returns `n % d`
    pub extern "C" fn __modti3(a: i128, b: i128) -> i128 {
        i128_div_rem(a, b).1
    }

    // LLVM does not currently have a `__divmodti4` function
}
