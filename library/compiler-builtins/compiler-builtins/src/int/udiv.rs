#[cfg(not(feature = "unstable-public-internals"))]
pub(crate) use crate::int::specialized_div_rem::*;
#[cfg(feature = "unstable-public-internals")]
pub use crate::int::specialized_div_rem::*;

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
}

#[cfg(not(target_arch = "avr"))]
intrinsics! {
    #[maybe_use_optimized_c_shim]
    /// Returns `n / d` and sets `*rem = n % d`
    pub extern "C" fn __udivmodsi4(n: u32, d: u32, rem: Option<&mut u32>) -> u32 {
        let quo_rem = u32_div_rem(n, d);
        if let Some(rem) = rem {
            *rem = quo_rem.1;
        }
        quo_rem.0
    }
}

#[cfg(target_arch = "avr")]
intrinsics! {
    /// Returns `n / d` and `n % d` packed together.
    ///
    /// Ideally we'd use `-> (u32, u32)` or some kind of a packed struct, but
    /// both force a stack allocation, while our result has to be in R18:R26.
    pub extern "C" fn __udivmodsi4(n: u32, d: u32) -> u64 {
        let (div, rem) = u32_div_rem(n, d);

        ((rem as u64) << 32) | (div as u64)
    }

    #[unsafe(naked)]
    pub unsafe extern "C" fn __udivmodqi4() {
        // compute unsigned 8-bit `n / d` and `n % d`.
        //
        // Note: GCC implements a [non-standard calling convention](https://gcc.gnu.org/wiki/avr-gcc#Exceptions_to_the_Calling_Convention) for this function.
        // Inputs:
        //     R24: dividend
        //     R22: divisor
        // Outputs:
        //     R24: quotient  (dividend / divisor)
        //     R25: remainder (dividend % divisor)
        // Clobbers:
        //     R23: loop counter
        core::arch::naked_asm!(
            // This assembly routine implements the [long division](https://en.wikipedia.org/wiki/Division_algorithm#Long_division) algorithm.
            // Bits shift out of the dividend and into the quotient, so R24 is used for both.
            "clr R25",      // remainder = 0

            "ldi R23, 8",   // for each bit
            "1:",
            "lsl R24",      //     shift the dividend MSb
            "rol R25",      //     into the remainder LSb

            "cp  R25, R22", //     if remainder >= divisor
            "brlo 2f",
            "sub R25, R22", //         remainder -= divisor
            "sbr R24, 1",   //         quotient |= 1
            "2:",

            "dec R23",      // end loop
            "brne 1b",
            "ret",
        );
    }

    #[unsafe(naked)]
    pub unsafe extern "C" fn __udivmodhi4() {
        // compute unsigned 16-bit `n / d` and `n % d`.
        //
        // Note: GCC implements a [non-standard calling convention](https://gcc.gnu.org/wiki/avr-gcc#Exceptions_to_the_Calling_Convention) for this function.
        // Inputs:
        //     R24: dividend [low]
        //     R25: dividend [high]
        //     R22: divisor [low]
        //     R23: divisor [high]
        // Outputs:
        //     R22: quotient [low]  (dividend / divisor)
        //     R23: quotient [high]
        //     R24: remainder [low] (dividend % divisor)
        //     R25: remainder [high]
        // Clobbers:
        //     R21: loop counter
        //     R26: divisor [low]
        //     R27: divisor [high]
        core::arch::naked_asm!(
            // This assembly routine implements the [long division](https://en.wikipedia.org/wiki/Division_algorithm#Long_division) algorithm.
            // Bits shift out of the dividend and into the quotient, so R24+R25 are used for both.
            "mov R26, R22",     // move divisor to make room for quotient
            "mov R27, R23",
            "mov R22, R24",     // move dividend to output location (becomes quotient)
            "mov R23, R25",
            "clr R24",          // remainder = 0
            "clr R25",

            "ldi R21, 16",      // for each bit
            "1:",
            "lsl R22",          //     shift the dividend MSb
            "rol R23",
            "rol R24",          //     into the remainder LSb
            "rol R25",

            "cp  R24, R26",     //     if remainder >= divisor
            "cpc R25, R27",
            "brlo 2f",
            "sub R24, R26",     //         remainder -= divisor
            "sbc R25, R27",
            "sbr R22, 1",       //         quotient |= 1
            "2:",

            "dec R21",          // end loop
            "brne 1b",
            "ret",
        );
    }

}

intrinsics! {
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

    // Note: we use block configuration and not `if cfg!(...)`, because we need to entirely disable
    // the existence of `u128_div_rem` to get 32-bit SPARC to compile, see `u128_divide_sparc` docs.

    /// Returns `n / d`
    pub extern "C" fn __udivti3(n: u128, d: u128) -> u128 {
        #[cfg(not(any(target_arch = "sparc", target_arch = "sparc64")))] {
            u128_div_rem(n, d).0
        }
        #[cfg(any(target_arch = "sparc", target_arch = "sparc64"))] {
            u128_divide_sparc(n, d, &mut 0)
        }
    }

    /// Returns `n % d`
    pub extern "C" fn __umodti3(n: u128, d: u128) -> u128 {
        #[cfg(not(any(target_arch = "sparc", target_arch = "sparc64")))] {
            u128_div_rem(n, d).1
        }
        #[cfg(any(target_arch = "sparc", target_arch = "sparc64"))] {
            let mut rem = 0;
            u128_divide_sparc(n, d, &mut rem);
            rem
        }
    }

    /// Returns `n / d` and sets `*rem = n % d`
    pub extern "C" fn __udivmodti4(n: u128, d: u128, rem: Option<&mut u128>) -> u128 {
        #[cfg(not(any(target_arch = "sparc", target_arch = "sparc64")))] {
            let quo_rem = u128_div_rem(n, d);
            if let Some(rem) = rem {
                *rem = quo_rem.1;
            }
            quo_rem.0
        }
        #[cfg(any(target_arch = "sparc", target_arch = "sparc64"))] {
            let mut tmp = 0;
            let quo = u128_divide_sparc(n, d, &mut tmp);
            if let Some(rem) = rem {
                *rem = tmp;
            }
            quo
        }
    }
}
