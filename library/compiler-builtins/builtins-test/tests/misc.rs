// makes configuration easier
#![allow(unused_macros)]

use builtins_test::*;

/// Make sure that the the edge case tester and randomized tester don't break, and list examples of
/// fuzz values for documentation purposes.
#[test]
fn fuzz_values() {
    const VALS: [u16; 47] = [
        0b0, // edge cases
        0b1111111111111111,
        0b1111111111111110,
        0b1111111111111100,
        0b1111111110000000,
        0b1111111100000000,
        0b1110000000000000,
        0b1100000000000000,
        0b1000000000000000,
        0b111111111111111,
        0b111111111111110,
        0b111111111111100,
        0b111111110000000,
        0b111111100000000,
        0b110000000000000,
        0b100000000000000,
        0b11111111111111,
        0b11111111111110,
        0b11111111111100,
        0b11111110000000,
        0b11111100000000,
        0b10000000000000,
        0b111111111,
        0b111111110,
        0b111111100,
        0b110000000,
        0b100000000,
        0b11111111,
        0b11111110,
        0b11111100,
        0b10000000,
        0b111,
        0b110,
        0b100,
        0b11,
        0b10,
        0b1,
        0b1010110100000, // beginning of random fuzzing
        0b1100011001011010,
        0b1001100101001111,
        0b1101010100011010,
        0b100010001,
        0b1000000000000000,
        0b1100000000000101,
        0b1100111101010101,
        0b1100010111111111,
        0b1111110101111111,
    ];
    let mut i = 0;
    fuzz(10, |x: u16| {
        assert_eq!(x, VALS[i]);
        i += 1;
    });
}

#[test]
fn leading_zeros() {
    use compiler_builtins::int::leading_zeros::{leading_zeros_default, leading_zeros_riscv};
    {
        use compiler_builtins::int::leading_zeros::__clzsi2;
        fuzz(N, |x: u32| {
            if x == 0 {
                return; // undefined value for an intrinsic
            }
            let lz = x.leading_zeros() as usize;
            let lz0 = __clzsi2(x);
            let lz1 = leading_zeros_default(x);
            let lz2 = leading_zeros_riscv(x);
            if lz0 != lz {
                panic!("__clzsi2({x}): std: {lz}, builtins: {lz0}");
            }
            if lz1 != lz {
                panic!("leading_zeros_default({x}): std: {lz}, builtins: {lz1}");
            }
            if lz2 != lz {
                panic!("leading_zeros_riscv({x}): std: {lz}, builtins: {lz2}");
            }
        });
    }

    {
        use compiler_builtins::int::leading_zeros::__clzdi2;
        fuzz(N, |x: u64| {
            if x == 0 {
                return; // undefined value for an intrinsic
            }
            let lz = x.leading_zeros() as usize;
            let lz0 = __clzdi2(x);
            let lz1 = leading_zeros_default(x);
            let lz2 = leading_zeros_riscv(x);
            if lz0 != lz {
                panic!("__clzdi2({x}): std: {lz}, builtins: {lz0}");
            }
            if lz1 != lz {
                panic!("leading_zeros_default({x}): std: {lz}, builtins: {lz1}");
            }
            if lz2 != lz {
                panic!("leading_zeros_riscv({x}): std: {lz}, builtins: {lz2}");
            }
        });
    }

    {
        use compiler_builtins::int::leading_zeros::__clzti2;
        fuzz(N, |x: u128| {
            if x == 0 {
                return; // undefined value for an intrinsic
            }
            let lz = x.leading_zeros() as usize;
            let lz0 = __clzti2(x);
            if lz0 != lz {
                panic!("__clzti2({x}): std: {lz}, builtins: {lz0}");
            }
        });
    }
}

#[test]
fn trailing_zeros() {
    use compiler_builtins::int::trailing_zeros::{__ctzdi2, __ctzsi2, __ctzti2, trailing_zeros};
    fuzz(N, |x: u32| {
        if x == 0 {
            return; // undefined value for an intrinsic
        }
        let tz = x.trailing_zeros() as usize;
        let tz0 = __ctzsi2(x);
        let tz1 = trailing_zeros(x);
        if tz0 != tz {
            panic!("__ctzsi2({x}): std: {tz}, builtins: {tz0}");
        }
        if tz1 != tz {
            panic!("trailing_zeros({x}): std: {tz}, builtins: {tz1}");
        }
    });
    fuzz(N, |x: u64| {
        if x == 0 {
            return; // undefined value for an intrinsic
        }
        let tz = x.trailing_zeros() as usize;
        let tz0 = __ctzdi2(x);
        let tz1 = trailing_zeros(x);
        if tz0 != tz {
            panic!("__ctzdi2({x}): std: {tz}, builtins: {tz0}");
        }
        if tz1 != tz {
            panic!("trailing_zeros({x}): std: {tz}, builtins: {tz1}");
        }
    });
    fuzz(N, |x: u128| {
        if x == 0 {
            return; // undefined value for an intrinsic
        }
        let tz = x.trailing_zeros() as usize;
        let tz0 = __ctzti2(x);
        if tz0 != tz {
            panic!("__ctzti2({x}): std: {tz}, builtins: {tz0}");
        }
    });
}

#[test]
fn bswap() {
    use compiler_builtins::int::bswap::{__bswapdi2, __bswapsi2};
    fuzz(N, |x: u32| {
        assert_eq!(x.swap_bytes(), __bswapsi2(x));
    });
    fuzz(N, |x: u64| {
        assert_eq!(x.swap_bytes(), __bswapdi2(x));
    });

    assert_eq!(__bswapsi2(0x12345678u32), 0x78563412u32);
    assert_eq!(__bswapsi2(0x00000001u32), 0x01000000u32);
    assert_eq!(__bswapdi2(0x123456789ABCDEF0u64), 0xF0DEBC9A78563412u64);
    assert_eq!(__bswapdi2(0x0200000001000000u64), 0x0000000100000002u64);

    #[cfg(any(target_pointer_width = "32", target_pointer_width = "64"))]
    {
        use compiler_builtins::int::bswap::__bswapti2;
        fuzz(N, |x: u128| {
            assert_eq!(x.swap_bytes(), __bswapti2(x));
        });

        assert_eq!(
            __bswapti2(0x123456789ABCDEF013579BDF02468ACEu128),
            0xCE8A4602DF9B5713F0DEBC9A78563412u128
        );
        assert_eq!(
            __bswapti2(0x04000000030000000200000001000000u128),
            0x00000001000000020000000300000004u128
        );
    }
}
