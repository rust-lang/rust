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
            let a = (a ^ s_a) - s_a;
            let b = (b ^ s_b) - s_b;
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
            let b = (b ^ s) - s;
            let s = a >> (<$ty>::bits() - 1);
            let a = (a ^ s) - s;

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
            *rem = a - (r * b);
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

#[cfg(test)]
mod tests {
    use qc::{U32, U64};

    check! {
        fn __divdi3(f: extern fn(i64, i64) -> i64, n: U64, d: U64) -> Option<i64> {
            let (n, d) = (n.0 as i64, d.0 as i64);
            if d == 0 {
                None
            } else {
                Some(f(n, d))
            }
        }

        fn __moddi3(f: extern fn(i64, i64) -> i64, n: U64, d: U64) -> Option<i64> {
            let (n, d) = (n.0 as i64, d.0 as i64);
            if d == 0 {
                None
            } else {
                Some(f(n, d))
            }
        }

        fn __divmoddi4(f: extern fn(i64, i64, &mut i64) -> i64,
                       n: U64,
                       d: U64) -> Option<(i64, i64)> {
            let (n, d) = (n.0 as i64, d.0 as i64);
            if d == 0 {
                None
            } else {
                let mut r = 0;
                let q = f(n, d, &mut r);
                Some((q, r))
            }
        }

        fn __divsi3(f: extern fn(i32, i32) -> i32,
                    n: U32,
                    d: U32) -> Option<i32> {
            let (n, d) = (n.0 as i32, d.0 as i32);
            if d == 0 {
                None
            } else {
                Some(f(n, d))
            }
        }

        fn __modsi3(f: extern fn(i32, i32) -> i32,
                    n: U32,
                    d: U32) -> Option<i32> {
            let (n, d) = (n.0 as i32, d.0 as i32);
            if d == 0 {
                None
            } else {
                Some(f(n, d))
            }
        }

        fn __divmodsi4(f: extern fn(i32, i32, &mut i32) -> i32,
                       n: U32,
                       d: U32) -> Option<(i32, i32)> {
            let (n, d) = (n.0 as i32, d.0 as i32);
            if d == 0 {
                None
            } else {
                let mut r = 0;
                let q = f(n, d, &mut r);
                Some((q, r))
            }
        }
    }
}

#[cfg(test)]
#[cfg(all(not(windows),
          not(target_arch = "mips64"),
          not(target_arch = "mips64el"),
          target_pointer_width="64"))]
mod tests_i128 {
    use qc::U128;
    check! {

        fn __divti3(f: extern fn(i128, i128) -> i128, n: U128, d: U128) -> Option<i128> {
            let (n, d) = (n.0 as i128, d.0 as i128);
            if d == 0 {
                None
            } else {
                Some(f(n, d))
            }
        }

        fn __modti3(f: extern fn(i128, i128) -> i128, n: U128, d: U128) -> Option<i128> {
            let (n, d) = (n.0 as i128, d.0 as i128);
            if d == 0 {
                None
            } else {
                Some(f(n, d))
            }
        }
    }
}
