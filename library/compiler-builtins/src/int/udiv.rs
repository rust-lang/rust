use int::{Int, LargeInt};

intrinsics! {
    #[maybe_use_optimized_c_shim]
    #[arm_aeabi_alias = __aeabi_uidiv]
    /// Returns `n / d`
    pub extern "C" fn __udivsi3(n: u32, d: u32) -> u32 {
        u32_div_rem(n, d).0
    }

    #[maybe_use_optimized_c_shim]
    /// Returns `n % d`
    pub extern "C" fn __umodsi3(n: u32, d: u32) -> u32 {
        u32_div_rem(n, d).1
    }

    #[maybe_use_optimized_c_shim]
    /// Returns `n / d` and sets `*rem = n % d`
    pub extern "C" fn __udivmodsi4(n: u32, d: u32, rem: Option<&mut u32>) -> u32 {
        let quo_rem = u32_div_rem(n, d);
        if let Some(rem) = rem {
            *rem = quo_rem.1;
        }
        quo_rem.0
    }

    #[maybe_use_optimized_c_shim]
    /// Returns `n / d`
    pub extern "C" fn __udivdi3(n: u64, d: u64) -> u64 {
        u64_div_rem(n, d).0
    }

    #[maybe_use_optimized_c_shim]
    /// Returns `n % d`
    pub extern "C" fn __umoddi3(n: u64, d: u64) -> u64 {
        u64_div_rem(n, d).1
    }

    #[maybe_use_optimized_c_shim]
    /// Returns `n / d` and sets `*rem = n % d`
    pub extern "C" fn __udivmoddi4(n: u64, d: u64, rem: Option<&mut u64>) -> u64 {
        let quo_rem = u64_div_rem(n, d);
        if let Some(rem) = rem {
            *rem = quo_rem.1;
        }
        quo_rem.0
    }

    #[win64_128bit_abi_hack]
    /// Returns `n / d`
    pub extern "C" fn __udivti3(n: u128, d: u128) -> u128 {
        u128_div_rem(n, d).0
    }

    #[win64_128bit_abi_hack]
    /// Returns `n % d`
    pub extern "C" fn __umodti3(n: u128, d: u128) -> u128 {
        u128_div_rem(n, d).1
    }

    #[win64_128bit_abi_hack]
    /// Returns `n / d` and sets `*rem = n % d`
    pub extern "C" fn __udivmodti4(n: u128, d: u128, rem: Option<&mut u128>) -> u128 {
        let quo_rem = u128_div_rem(n, d);
        if let Some(rem) = rem {
            *rem = quo_rem.1;
        }
        quo_rem.0
    }
}
