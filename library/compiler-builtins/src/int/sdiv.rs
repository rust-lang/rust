use int::Int;

macro_rules! div {
    ($intrinsic:ident: $ty:ty, $uty:ty) => {
        /// Returns `a / b`
        #[cfg_attr(not(test), no_mangle)]
        pub extern "C" fn $intrinsic(a: $ty, b: $ty) -> $ty {
            let s_a = a >> (<$ty>::bits() - 1);
            let s_b = b >> (<$ty>::bits() - 1);
            let a = (a ^ s_a) - s_a;
            let b = (b ^ s_b) - s_b;
            let s = s_a ^ s_b;
            let r = (a as $uty) / (b as $uty);
            (r as $ty ^ s) - s
        }
    }
}

macro_rules! mod_ {
    ($intrinsic:ident: $ty:ty, $uty:ty) => {
        /// Returns `a % b`
        #[cfg_attr(not(test), no_mangle)]
        pub extern "C" fn $intrinsic(a: $ty, b: $ty) -> $ty {
            let s = b >> (<$ty>::bits() - 1);
            let b = (b ^ s) - s;
            let s = a >> (<$ty>::bits() - 1);
            let a = (a ^ s) - s;
            let r = (a as $uty) % (b as $uty);
            (r as $ty ^ s) - s
        }
    }
}

macro_rules! divmod {
    ($intrinsic:ident, $div:ident: $ty:ty) => {
        /// Returns `a / b` and sets `*rem = n % d`
        #[cfg_attr(not(test), no_mangle)]
        pub extern "C" fn $intrinsic(a: $ty, b: $ty, rem: &mut $ty) -> $ty {
            let r = $div(a, b);
            *rem = a - (r * b);
            r
        }
    }
}

div!(__divsi3: i32, u32);
div!(__divdi3: i64, u64);
mod_!(__modsi3: i32, u32);
mod_!(__moddi3: i64, u64);
divmod!(__divmodsi4, __divsi3: i32);
divmod!(__divmoddi4, __divdi3: i64);

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
