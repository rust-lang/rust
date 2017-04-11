use int::Int;

macro_rules! div {
    ($intrinsic:ident: $ty:ty, $uty:ty) => {
        div!($intrinsic: $ty, $uty, $ty, |i| {i});
    };
    ($intrinsic:ident: $ty:ty, $uty:ty, $tyret:ty, $conv:expr) => {
        /// Returns `a / b`
        #[cfg_attr(not(test), no_mangle)]
        pub extern "C" fn $intrinsic(a: $ty, b: $ty) -> $tyret {
            let s_a = a >> (<$ty>::bits() - 1);
            let s_b = b >> (<$ty>::bits() - 1);
            // NOTE it's OK to overflow here because of the `as $uty` cast below
            // This whole operation is computing the absolute value of the inputs
            // So some overflow will happen when dealing with e.g. `i64::MIN`
            // where the absolute value is `(-i64::MIN) as u64`
            let a = (a ^ s_a).wrapping_sub(s_a);
            let b = (b ^ s_b).wrapping_sub(s_b);
            let s = s_a ^ s_b;

            let r = udiv!(a as $uty, b as $uty);
            ($conv)((r as $ty ^ s) - s)
        }
    }
}

macro_rules! mod_ {
    ($intrinsic:ident: $ty:ty, $uty:ty) => {
        mod_!($intrinsic: $ty, $uty, $ty, |i| {i});
    };
    ($intrinsic:ident: $ty:ty, $uty:ty, $tyret:ty, $conv:expr) => {
        /// Returns `a % b`
        #[cfg_attr(not(test), no_mangle)]
        pub extern "C" fn $intrinsic(a: $ty, b: $ty) -> $tyret {
            let s = b >> (<$ty>::bits() - 1);
            // NOTE(wrapping_sub) see comment in the `div` macro
            let b = (b ^ s).wrapping_sub(s);
            let s = a >> (<$ty>::bits() - 1);
            let a = (a ^ s).wrapping_sub(s);

            let r = urem!(a as $uty, b as $uty);
            ($conv)((r as $ty ^ s) - s)
        }
    }
}

macro_rules! divmod {
    ($abi:tt, $intrinsic:ident, $div:ident: $ty:ty) => {
        /// Returns `a / b` and sets `*rem = n % d`
        #[cfg_attr(not(test), no_mangle)]
        pub extern $abi fn $intrinsic(a: $ty, b: $ty, rem: &mut $ty) -> $ty {
            #[cfg(all(feature = "c", any(target_arch = "x86")))]
            extern {
                fn $div(a: $ty, b: $ty) -> $ty;
            }

            let r = match () {
                #[cfg(not(all(feature = "c", any(target_arch = "x86"))))]
                () => $div(a, b),
                #[cfg(all(feature = "c", any(target_arch = "x86")))]
                () => unsafe { $div(a, b) },
            };
            // NOTE won't overflow because it's using the result from the
            // previous division
            *rem = a - r.wrapping_mul(b);
            r
        }
    }
}

#[cfg(not(all(feature = "c", target_arch = "arm", not(target_os = "ios"), not(thumbv6m))))]
div!(__divsi3: i32, u32);

#[cfg(not(all(feature = "c", target_arch = "x86")))]
div!(__divdi3: i64, u64);

#[cfg(not(all(windows, target_pointer_width="64")))]
div!(__divti3: i128, u128);

#[cfg(all(windows, target_pointer_width="64"))]
div!(__divti3: i128, u128, ::U64x2, ::sconv);

#[cfg(not(all(feature = "c", target_arch = "arm", not(target_os = "ios"))))]
mod_!(__modsi3: i32, u32);

#[cfg(not(all(feature = "c", target_arch = "x86")))]
mod_!(__moddi3: i64, u64);

#[cfg(not(all(windows, target_pointer_width="64")))]
mod_!(__modti3: i128, u128);

#[cfg(all(windows, target_pointer_width="64"))]
mod_!(__modti3: i128, u128, ::U64x2, ::sconv);

#[cfg(not(all(feature = "c", target_arch = "arm", not(target_os = "ios"))))]
divmod!("C", __divmodsi4, __divsi3: i32);

#[cfg(target_arch = "arm")]
divmod!("aapcs", __divmoddi4, __divdi3: i64);

#[cfg(not(target_arch = "arm"))]
divmod!("C", __divmoddi4, __divdi3: i64);
