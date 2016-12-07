// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(compiler_builtins)]
#![no_std]
#![compiler_builtins]
#![unstable(feature = "compiler_builtins_lib",
            reason = "internal implementation detail of rustc right now",
            issue = "0")]
#![crate_name = "compiler_builtins"]
#![crate_type = "rlib"]
#![feature(staged_api)]
#![cfg_attr(any(target_pointer_width="32", target_pointer_width="16", target_os="windows",
            target_arch="mips64"),
            feature(core_intrinsics, core_float))]
#![feature(associated_consts)]
#![cfg_attr(not(stage0), feature(i128_type))]

#![allow(non_camel_case_types, unused_variables)]

#[cfg(any(target_pointer_width="32", target_pointer_width="16", target_os="windows",
          target_arch="mips64"))]
pub mod reimpls {

    #![allow(unused_comparisons)]

    use core::intrinsics::unchecked_div;
    use core::intrinsics::unchecked_rem;
    use core::ptr;

    // C API is expected to tolerate some amount of size mismatch in ABI. Hopefully the amount of
    // handling is sufficient for bootstrapping.
    #[cfg(stage0)]
    type u128_ = u64;
    #[cfg(stage0)]
    type i128_ = i64;
    #[cfg(not(stage0))]
    type u128_ = u128;
    #[cfg(not(stage0))]
    type i128_ = i128;

    fn unimplemented() -> ! {
        unsafe { ::core::intrinsics::abort() }
    }

    macro_rules! ashl {
        ($a:expr, $b:expr, $ty:ty) => {{
            let (a, b) = ($a, $b);
            let bits = (::core::mem::size_of::<$ty>() * 8) as $ty;
            let half_bits = bits / 2;
            if b & half_bits != 0 {
                <$ty>::from_parts(0, a.low() << (b - half_bits))
            } else if b == 0 {
                a
            } else {
                <$ty>::from_parts(a.low() << b, (a.high() << b) | (a.low() >> (half_bits - b)))
            }
        }}
    }

    #[export_name="__ashlti3"]
    pub extern "C" fn shl(a: u128_, b: u128_) -> u128_ {
        ashl!(a, b, u128_)
    }

    macro_rules! ashr {
        ($a: expr, $b: expr, $ty:ty) => {{
            let (a, b) = ($a, $b);
            let bits = (::core::mem::size_of::<$ty>() * 8) as $ty;
            let half_bits = bits >> 1;
            if b & half_bits != 0 {
                <$ty>::from_parts((a.high() >> (b - half_bits)) as <$ty as LargeInt>::LowHalf,
                                  a.high() >> (half_bits - 1))
            } else if b == 0 {
                a
            } else {
                let high_unsigned = a.high() as <$ty as LargeInt>::LowHalf;
                <$ty>::from_parts((high_unsigned << (half_bits - b)) | (a.low() >> b),
                                  a.high() >> b)
            }
        }}
    }

    #[export_name="__ashrti3"]
    pub extern "C" fn shr(a: i128_, b: i128_) -> i128_ {
        ashr!(a, b, i128_)
    }

    macro_rules! lshr {
        ($a: expr, $b: expr, $ty:ty) => {{
            let (a, b) = ($a, $b);
            let bits = (::core::mem::size_of::<$ty>() * 8) as $ty;
            let half_bits = bits >> 1;
            if b & half_bits != 0 {
                <$ty>::from_parts(a.high() >> (b - half_bits), 0)
            } else if b == 0 {
                a
            } else {
                <$ty>::from_parts((a.high() << (half_bits - b)) | (a.low() >> b), a.high() >> b)
            }
        }}
    }


    #[export_name="__lshrti3"]
    pub extern "C" fn lshr(a: u128_, b: u128_) -> u128_ {
        lshr!(a, b, u128_)
    }

    #[cfg(stage0)]
    #[export_name="__udivmodti4"]
    pub extern "C" fn u128_div_mod(n: u128_, d: u128_, rem: *mut u128_) -> u128_ {
        unsafe {
        if !rem.is_null() {
            *rem = unchecked_rem(n, d);
        }
        unchecked_div(n, d)
        }
    }

    #[cfg(not(stage0))]
    #[export_name="__udivmodti4"]
    pub extern "C" fn u128_div_mod(n: u128_, d: u128_, rem: *mut u128_) -> u128_ {
        // Translated from Figure 3-40 of The PowerPC Compiler Writer's Guide
        unsafe {
        // special cases, X is unknown, K != 0
        if n.high() == 0 {
            if d.high() == 0 {
                // 0 X
                // ---
                // 0 X
                if !rem.is_null() {
                    *rem = u128::from(unchecked_rem(n.low(), d.low()));
                }
                return u128::from(unchecked_div(n.low(), d.low()));
            } else {
                // 0 X
                // ---
                // K X
                if !rem.is_null() {
                    *rem = n;
                }
                return 0;
            };
        }

        let mut sr;
        let mut q;
        let mut r;

        if d.low() == 0 {
            if d.high() == 0 {
                // K X
                // ---
                // 0 0
                if !rem.is_null() {
                    *rem = u128::from(unchecked_rem(n.high(), d.low()));
                }
                return u128::from(unchecked_div(n.high(), d.low()));
            }

            if n.low() == 0 {
                // K 0
                // ---
                // K 0
                if !rem.is_null() {
                    *rem = u128::from_parts(0, unchecked_rem(n.high(), d.high()));
                }
                return u128::from(unchecked_div(n.high(), d.high()));
            }

            // K K
            // ---
            // K 0

            if d.high().is_power_of_two() {
                if !rem.is_null() {
                    *rem = u128::from_parts(n.low(), n.high() & (d.high() - 1));
                }
                return u128::from(n.high() >> d.high().trailing_zeros());
            }

            // K K
            // ---
            // K 0
            sr = d.high().leading_zeros().wrapping_sub(n.high().leading_zeros());

            // D > N
            if sr > 64 - 2 {
                if !rem.is_null() {
                    *rem = n;
                }
                return 0;
            }

            sr += 1;

            // 1 <= sr <= u64::bits() - 1
            q = n << (64 - sr);
            r = n >> sr;
        } else {
            if d.high() == 0 {
                // K X
                // ---
                // 0 K
                if d.low().is_power_of_two() {
                    if !rem.is_null() {
                        *rem = u128::from(n.low() & (d.low() - 1));
                    }

                    if d.low() == 1 {
                        return n;
                    } else {
                        let sr = d.low().trailing_zeros();
                        return n >> sr;
                    };
                }

                sr = 1 + 64 + d.low().leading_zeros() - n.high().leading_zeros();

                // 2 <= sr <= u64::bits() - 1
                q = n << (128 - sr);
                r = n >> sr;
                // FIXME the C compiler-rt implementation has something here
                // that looks like a speed optimisation.
                // It would be worth a try to port it to Rust too and
                // compare the speed.
            } else {
                // K X
                // ---
                // K K
                sr = d.high().leading_zeros().wrapping_sub(n.high().leading_zeros());

                // D > N
                if sr > 64 - 1 {
                    if !rem.is_null() {
                        *rem = n;
                    }
                    return 0;
                }

                sr += 1;

                // 1 <= sr <= u32::bits()
                q = n << (128 - sr);
                r = n >> sr;
            }
        }

        // Not a special case
        // q and r are initialized with
        // q = n << (u64::bits() - sr)
        // r = n >> sr
        // 1 <= sr <= u64::bits() - 1
        let mut carry = 0;

        // FIXME: replace this with a for loop
        // (atm not doable as this generates call to
        // eh_personality when optimisations are turned off,
        // which in turn gives a linker error in later
        // compilation steps)
        while sr > 0 {
            // r:q = ((r:q) << 1) | carry
            r = (r << 1) | (q >> (128 - 1));
            q = (q << 1) | carry as u128;

            // carry = 0
            // if r >= d {
            //     r -= d;
            //     carry = 1;
            // }
            let s = (d.wrapping_sub(r).wrapping_sub(1)) as i128 >> (128 - 1);
            carry = (s & 1) as u64;
            r -= d & s as u128;
            sr -= 1;
        }

        if !rem.is_null() {
            *rem = r;
        }
        (q << 1) | carry as u128
        }
    }

    #[export_name="__umodti3"]
    pub extern "C" fn u128_mod(a: u128_, b: u128_) -> u128_ {
        unsafe {
            let mut r = ::core::mem::zeroed();
            u128_div_mod(a, b, &mut r);
            r
        }
    }

    #[export_name="__modti3"]
    pub extern "C" fn i128_mod(a: i128_, b: i128_) -> i128_ {
        let b = b.uabs();
        let sa = a.signum();
        let a = a.uabs();
        unsafe {
            let mut r = ::core::mem::zeroed();
            u128_div_mod(a, b, &mut r);
            if sa == -1 { -(r as i128_) } else { r as i128_ }
        }
    }

    #[export_name="__divti3"]
    pub extern "C" fn i128_div(a: i128_, b: i128_) -> i128_ {
        let sa = a.signum();
        let sb = b.signum();
        let a = a.uabs();
        let b = b.uabs();
        let sr = sa * sb; // sign of quotient
        if sr == -1 {
            -(u128_div_mod(a, b, ptr::null_mut()) as i128_)
        } else {
            u128_div_mod(a, b, ptr::null_mut()) as i128_
        }
    }

    #[export_name="__udivti3"]
    pub extern "C" fn u128_div(a: u128_, b: u128_) -> u128_ {
        u128_div_mod(a, b, ptr::null_mut())
    }

    macro_rules! mulo {
        ($a:expr, $b:expr, $o: expr, $ty: ty) => {{
            let (a, b, overflow) = ($a, $b, $o);
            *overflow = 0;
            let result = a.wrapping_mul(b);
            if a == <$ty>::min_value() {
                if b != 0 && b != 1 {
                    *overflow = 1;
                }
                return result;
            }
            if b == <$ty>::min_value() {
                if a != 0 && a != 1 {
                    *overflow = 1;
                }
                return result;
            }

            let sa = a.signum();
            let abs_a = a.iabs();
            let sb = b.signum();
            let abs_b = b.iabs();
            if abs_a < 2 || abs_b < 2 {
                return result;
            }
            unsafe {
            if sa == sb {
                if abs_a > unchecked_div(<$ty>::max_value(), abs_b) {
                    *overflow = 1;
                }
            } else {
                if abs_a > unchecked_div(<$ty>::min_value(), -abs_b) {
                    *overflow = 1;
                }
            }
            }
            result
        }}
    }

    // FIXME: i32 here should be c_int.
    #[export_name="__muloti4"]
    pub extern "C" fn i128_mul_oflow(a: i128_, b: i128_, o: &mut i32) -> i128_ {
        mulo!(a, b, o, i128_)
    }

    pub trait LargeInt {
        type LowHalf;
        type HighHalf;

        fn low(self) -> Self::LowHalf;
        fn high(self) -> Self::HighHalf;
        fn from_parts(low: Self::LowHalf, high: Self::HighHalf) -> Self;
    }
    impl LargeInt for u64 {
        type LowHalf = u32;
        type HighHalf = u32;

        fn low(self) -> u32 {
            self as u32
        }
        fn high(self) -> u32 {
            (self >> 32) as u32
        }
        fn from_parts(low: u32, high: u32) -> u64 {
            low as u64 | ((high as u64) << 32)
        }
    }
    impl LargeInt for i64 {
        type LowHalf = u32;
        type HighHalf = i32;

        fn low(self) -> u32 {
            self as u32
        }
        fn high(self) -> i32 {
            (self >> 32) as i32
        }
        fn from_parts(low: u32, high: i32) -> i64 {
            low as i64 | ((high as i64) << 32)
        }
    }
    #[cfg(not(stage0))]
    impl LargeInt for u128 {
        type LowHalf = u64;
        type HighHalf = u64;

        fn low(self) -> u64 {
            self as u64
        }
        fn high(self) -> u64 {
            unsafe { *(&self as *const u128 as *const u64).offset(1) }
        }
        fn from_parts(low: u64, high: u64) -> u128 {
            #[repr(C, packed)] struct Parts(u64, u64);
            unsafe { ::core::mem::transmute(Parts(low, high)) }
        }
    }
    #[cfg(not(stage0))]
    impl LargeInt for i128 {
        type LowHalf = u64;
        type HighHalf = i64;

        fn low(self) -> u64 {
            self as u64
        }
        fn high(self) -> i64 {
            unsafe { *(&self as *const i128 as *const i64).offset(1) }
        }
        fn from_parts(low: u64, high: i64) -> i128 {
            u128::from_parts(low, high as u64) as i128
        }
    }

    macro_rules! mul {
        ($a:expr, $b:expr, $ty: ty, $tyh: ty) => {{
            let (a, b) = ($a, $b);
            let half_bits = (::core::mem::size_of::<$tyh>() * 8) / 2;
            let lower_mask = !0 >> half_bits;
            let mut low = (a.low() & lower_mask) * (b.low() & lower_mask);
            let mut t = low >> half_bits;
            low &= lower_mask;
            t += (a.low() >> half_bits) * (b.low() & lower_mask);
            low += (t & lower_mask) << half_bits;
            let mut high = (t >> half_bits) as $tyh;
            t = low >> half_bits;
            low &= lower_mask;
            t += (b.low() >> half_bits) * (a.low() & lower_mask);
            low += (t & lower_mask) << half_bits;
            high += (t >> half_bits) as $tyh;
            high += ((a.low() >> half_bits) * (b.low() >> half_bits)) as $tyh;
            high = high
                .wrapping_add(a.high()
                .wrapping_mul(b.low() as $tyh))
                .wrapping_add((a.low() as $tyh)
                .wrapping_mul(b.high()));
            <$ty>::from_parts(low, high)
        }}
    }

    #[cfg(stage0)]
    #[export_name="__multi3"]
    pub extern "C" fn u128_mul(a: i128_, b: i128_) -> i128_ {
        (a as i64 * b as i64) as i128_
    }

    #[cfg(not(stage0))]
    #[export_name="__multi3"]
    pub extern "C" fn u128_mul(a: i128_, b: i128_) -> i128_ {
        mul!(a, b, i128_, i64)
    }

    trait AbsExt: Sized {
        fn uabs(self) -> u128_;
        fn iabs(self) -> i128_;
    }

    impl AbsExt for i128_ {
        fn uabs(self) -> u128_ {
            self.iabs() as u128_
        }
        fn iabs(self) -> i128_ {
            ((self ^ self).wrapping_sub(self))
        }
    }

    trait FloatStuff: Sized {
        type ToBytes;

        const MANTISSA_BITS: u32;
        const MAX_EXP: i32;
        const EXP_MASK: Self::ToBytes;
        const MANTISSA_MASK: Self::ToBytes;
        const MANTISSA_LEAD_BIT: Self::ToBytes;

        fn to_bytes(self) -> Self::ToBytes;
        fn get_exponent(self) -> i32;
    }

    impl FloatStuff for f32 {
        type ToBytes = u32;
        const MANTISSA_BITS: u32 = 23;
        const MAX_EXP: i32 = 127;
        const EXP_MASK: u32 = 0x7F80_0000;
        const MANTISSA_MASK: u32 = 0x007F_FFFF;
        const MANTISSA_LEAD_BIT: u32 = 0x0080_0000;

        fn to_bytes(self) -> u32 { unsafe { ::core::mem::transmute(self) } }
        fn get_exponent(self) -> i32 {
            (((self.to_bytes() & Self::EXP_MASK) >> Self::MANTISSA_BITS) as i32) - Self::MAX_EXP
        }
    }

    impl FloatStuff for f64 {
        type ToBytes = u64;
        const MANTISSA_BITS: u32 = 52;
        const MAX_EXP: i32 = 1023;
        const EXP_MASK: u64 = 0x7FF0_0000_0000_0000;
        const MANTISSA_MASK: u64 = 0x000F_FFFF_FFFF_FFFF;
        const MANTISSA_LEAD_BIT: u64 = 0x0010_0000_0000_0000;

        fn to_bytes(self) -> u64 { unsafe { ::core::mem::transmute(self) } }
        fn get_exponent(self) -> i32 {
            (((self.to_bytes() & Self::EXP_MASK) >> Self::MANTISSA_BITS) as i32) - Self::MAX_EXP
        }
    }

    macro_rules! float_as_unsigned {
        ($from: expr, $fromty: ty, $outty: ty) => { {
            use core::num::Float;
            let repr = $from.to_bytes();
            let sign = $from.signum();
            let exponent = $from.get_exponent();
            let mantissa_fraction = repr & <$fromty as FloatStuff>::MANTISSA_MASK;
            let mantissa = mantissa_fraction | <$fromty as FloatStuff>::MANTISSA_LEAD_BIT;
            if sign == -1.0 || exponent < 0 { return 0; }
            if exponent > ::core::mem::size_of::<$outty>() as i32 * 8 {
                return !0;
            }
            if exponent < (<$fromty as FloatStuff>::MANTISSA_BITS) as i32 {
                mantissa as $outty >> (<$fromty as FloatStuff>::MANTISSA_BITS as i32 - exponent)
            } else {
                mantissa as $outty << (exponent - <$fromty as FloatStuff>::MANTISSA_BITS as i32)
            }
        } }
    }

    #[export_name="__fixunsdfti"]
    pub extern "C" fn f64_as_u128(a: f64) -> u128_ {
        float_as_unsigned!(a, f64, u128_)
    }

    #[export_name="__fixunssfti"]
    pub extern "C" fn f32_as_u128(a: f32) -> u128_ {
        float_as_unsigned!(a, f32, u128_)
    }

    macro_rules! float_as_signed {
        ($from: expr, $fromty: ty, $outty: ty) => {{
            use core::num::Float;
            let repr = $from.to_bytes();
            let sign = $from.signum();
            let exponent = $from.get_exponent();
            let mantissa_fraction = repr & <$fromty as FloatStuff>::MANTISSA_MASK;
            let mantissa = mantissa_fraction | <$fromty as FloatStuff>::MANTISSA_LEAD_BIT;

            if exponent < 0 { return 0; }
            if exponent > ::core::mem::size_of::<$outty>() as i32 * 8 {
                return if sign > 0.0 { <$outty>::max_value() } else { <$outty>::min_value() };
            }
            let r = if exponent < (<$fromty as FloatStuff>::MANTISSA_BITS) as i32 {
                mantissa as $outty >> (<$fromty as FloatStuff>::MANTISSA_BITS as i32 - exponent)
            } else {
                mantissa as $outty << (exponent - <$fromty as FloatStuff>::MANTISSA_BITS as i32)
            };
            if sign >= 0.0 { r } else { -r }
        }}
    }

    #[export_name="__fixdfti"]
    pub extern "C" fn f64_as_i128(a: f64) -> i128_ {
        float_as_signed!(a, f64, i128_)
    }

    #[export_name="__fixsfti"]
    pub extern "C" fn f32_as_i128(a: f32) -> i128_ {
        float_as_signed!(a, f32, i128_)
    }

    #[export_name="__floattidf"]
    pub extern "C" fn i128_as_f64(a: i128_) -> f64 {
        match a.signum() {
            1 => u128_as_f64(a.uabs()),
            0 => 0.0,
            -1 => -u128_as_f64(a.uabs()),
            _ => unimplemented()
        }
    }

    #[export_name="__floattisf"]
    pub extern "C" fn i128_as_f32(a: i128_) -> f32 {
        match a.signum() {
            1 => u128_as_f32(a.uabs()),
            0 => 0.0,
            -1 => -u128_as_f32(a.uabs()),
            _ => unimplemented()
        }
    }

    #[export_name="__floatuntidf"]
    pub extern "C" fn u128_as_f64(mut a: u128_) -> f64 {
        use ::core::f64::MANTISSA_DIGITS;
        if a == 0 { return 0.0; }
        let sd = 128 - a.leading_zeros();
        let mut e = sd - 1;
        const MD1 : u32 = MANTISSA_DIGITS + 1;
        const MD2 : u32 = MANTISSA_DIGITS + 2;

        if sd > MANTISSA_DIGITS {
            a = match sd {
                MD1 => a << 1,
                MD2 => a,
                _ => (a >> (sd - (MANTISSA_DIGITS + 2))) |
                     (if (a & (!0 >> (128 + MANTISSA_DIGITS + 2) - sd)) == 0 { 0 } else { 1 })
            };
            a |= if (a & 4) == 0 { 0 } else { 1 };
            a += 1;
            a >>= 2;
            if a & (1 << MANTISSA_DIGITS) != 0 {
                a >>= 1;
                e += 1;
            }
        } else {
            a <<= MANTISSA_DIGITS - sd;
        }
        unsafe {
            ::core::mem::transmute(((e as u64 + 1023) << 52) | (a as u64 & 0x000f_ffff_ffff_ffff))
        }
    }

    #[export_name="__floatuntisf"]
    pub extern "C" fn u128_as_f32(mut a: u128_) -> f32 {
        use ::core::f32::MANTISSA_DIGITS;
        if a == 0 { return 0.0; }
        let sd = 128 - a.leading_zeros();
        let mut e = sd - 1;
        const MD1 : u32 = MANTISSA_DIGITS + 1;
        const MD2 : u32 = MANTISSA_DIGITS + 2;

        if sd > MANTISSA_DIGITS {
            a = match sd {
                MD1 => a << 1,
                MD2 => a,
                _ => (a >> (sd - (MANTISSA_DIGITS + 2))) |
                     (if (a & (!0 >> (128 + MANTISSA_DIGITS + 2) - sd)) == 0 { 0 } else { 1 })
            };
            a |= if (a & 4) == 0 { 0 } else { 1 };
            a += 1;
            a >>= 2;
            if a & (1 << MANTISSA_DIGITS) != 0 {
                a >>= 1;
                e += 1;
            }
        } else {
            a <<= MANTISSA_DIGITS - sd;
        }
        unsafe {
            ::core::mem::transmute(((e + 127) << 23) | (a as u32 & 0x007f_ffff))
        }
    }
}
