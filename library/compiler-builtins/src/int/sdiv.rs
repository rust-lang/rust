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
        where F: Fn(Self, Self) -> Self,
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
    #[arm_aeabi_alias = __aeabi_idiv]
    pub extern "C" fn __divsi3(a: i32, b: i32) -> i32 {
        a.div(b)
    }

    #[use_c_shim_if(all(target_arch = "x86", not(target_env = "msvc")))]
    pub extern "C" fn __divdi3(a: i64, b: i64) -> i64 {
        a.div(b)
    }

    #[win64_128bit_abi_hack]
    pub extern "C" fn __divti3(a: i128, b: i128) -> i128 {
        a.div(b)
    }

    #[use_c_shim_if(all(target_arch = "arm", not(target_os = "ios")))]
    pub extern "C" fn __modsi3(a: i32, b: i32) -> i32 {
        a.mod_(b)
    }

    #[use_c_shim_if(all(target_arch = "x86", not(target_env = "msvc")))]
    pub extern "C" fn __moddi3(a: i64, b: i64) -> i64 {
        a.mod_(b)
    }

    #[win64_128bit_abi_hack]
    pub extern "C" fn __modti3(a: i128, b: i128) -> i128 {
        a.mod_(b)
    }

    #[use_c_shim_if(all(target_arch = "arm", not(target_os = "ios")))]
    pub extern "C" fn __divmodsi4(a: i32, b: i32, rem: &mut i32) -> i32 {
        a.divmod(b, rem, |a, b| __divsi3(a, b))
    }

    #[aapcs_on_arm]
    pub extern "C" fn __divmoddi4(a: i64, b: i64, rem: &mut i64) -> i64 {
        a.divmod(b, rem, |a, b| __divdi3(a, b))
    }
}

#[cfg_attr(not(stage0), lang = "i128_div")]
pub fn rust_i128_div(a: i128, b: i128) -> i128 {
    __divti3(a, b)
}
#[cfg_attr(not(stage0), lang = "i128_rem")]
pub fn rust_i128_rem(a: i128, b: i128) -> i128 {
    __modti3(a, b)
}
