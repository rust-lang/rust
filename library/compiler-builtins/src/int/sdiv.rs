use int::Int;

trait Div: Int {
    /// Returns `a / b`
    fn div(self, other: Self) -> Self {
        let s_a = self >> (Self::BITS - 1);
        let s_b = other >> (Self::BITS - 1);
        // NOTE it's OK to overflow here because of the `.unsigned()` below.
        // This whole operation is computing the absolute value of the inputs
        // So some overflow will happen when dealing with e.g. `i64::MIN`
        // where the absolute value is `(-i64::MIN) as u64`
        let a = (self ^ s_a).wrapping_sub(s_a);
        let b = (other ^ s_b).wrapping_sub(s_b);
        let s = s_a ^ s_b;

        let r = a.unsigned().aborting_div(b.unsigned());
        (Self::from_unsigned(r) ^ s) - s
    }
}

impl Div for i32 {}
impl Div for i64 {}
impl Div for i128 {}

trait Mod: Int {
    /// Returns `a % b`
    fn mod_(self, other: Self) -> Self {
        let s = other >> (Self::BITS - 1);
        // NOTE(wrapping_sub) see comment in the `div`
        let b = (other ^ s).wrapping_sub(s);
        let s = self >> (Self::BITS - 1);
        let a = (self ^ s).wrapping_sub(s);

        let r = a.unsigned().aborting_rem(b.unsigned());
        (Self::from_unsigned(r) ^ s) - s
    }
}

impl Mod for i32 {}
impl Mod for i64 {}
impl Mod for i128 {}

trait Divmod: Int {
    /// Returns `a / b` and sets `*rem = n % d`
    fn divmod<F>(self, other: Self, rem: &mut Self, div: F) -> Self
    where
        F: Fn(Self, Self) -> Self,
    {
        let r = div(self, other);
        // NOTE won't overflow because it's using the result from the
        // previous division
        *rem = self - r.wrapping_mul(other);
        r
    }
}

impl Divmod for i32 {}
impl Divmod for i64 {}


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

    #[aapcs_on_arm]
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
