//@ run-pass

#![feature(step_trait)]
#![feature(ascii_char)]

use std::iter::Step;
use std::num::{NonZeroU8, NonZeroU16, NonZeroU32, NonZeroU64, NonZeroUsize};
use std::ascii::Char as AsciiChar;
use std::net::{Ipv4Addr, Ipv6Addr};

// Pick various step sizes to exercise different overflow cases.
#[rustfmt::skip]
const STEPS: &[usize] = match usize::BITS {
    64 => &[
        1, 2, 5, 13, 29, 61, 127, 128, 255, 256, 511, 512,
        1087, 4079, 17919, 32767, 32768, 65535, 65536,
        (1 << 17) - 1, 1 << 17, (1 << 31) - 1, 1 << 31,
        (1 << 32) - 1, 1 << 32, (1 << 63) - 1, 1 << 63,
        usize::MAX,
    ],
    32 => &[
        1, 2, 5, 13, 29, 61, 127, 128, 255, 256, 511, 512,
        1087, 4079, 17919, 32767, 32768, 65535, 65536,
        (1 << 17) - 1, 1 << 17, (1 << 31) - 1, 1 << 31,
        usize::MAX,
    ],
    16 => &[
        1, 2, 5, 13, 29, 61, 127, 128, 255, 256, 511, 512,
        1087, 4079, 17919, 32767, 32768, usize::MAX
    ],
    _ => &[
        1, 2, 5, 13, 29, 61, 127, 128, usize::MAX
    ],
};

fn main() {
    // Exhaustively test the smaller integers
    for &step_size in STEPS {
        let mut n = u8::MIN;
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(
            Step::forward_overflowing(n, step_size),
            (n.wrapping_add(step_size as u8), true)
        );

        let mut n = u8::MAX;
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(
            Step::backward_overflowing(n, step_size),
            (n.wrapping_sub(step_size as u8), true)
        );
    }
    for &step_size in STEPS {
        let mut n = i8::MIN;
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(
            Step::forward_overflowing(n, step_size),
            (n.wrapping_add_unsigned(step_size as u8), true)
        );

        let mut n = i8::MAX;
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(
            Step::backward_overflowing(n, step_size),
            (n.wrapping_sub_unsigned(step_size as u8), true)
        );
    }

    for &step_size in STEPS {
        let mut n = u16::MIN;
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(
            Step::forward_overflowing(n, step_size),
            (n.wrapping_add(step_size as u16), true)
        );

        let mut n = u16::MAX;
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(
            Step::backward_overflowing(n, step_size),
            (n.wrapping_sub(step_size as u16), true)
        );
    }
    for &step_size in STEPS {
        let mut n = i16::MIN;
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(
            Step::forward_overflowing(n, step_size),
            (n.wrapping_add_unsigned(step_size as u16), true)
        );

        let mut n = i16::MAX;
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(
            Step::backward_overflowing(n, step_size),
            (n.wrapping_sub_unsigned(step_size as u16), true)
        );
    }

    for &step_size in STEPS {
        let mut n = NonZeroU8::MIN;
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(Step::forward_overflowing(n, step_size), (NonZeroU8::MAX, true));

        let mut n = NonZeroU8::MAX;
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(Step::backward_overflowing(n, step_size), (NonZeroU8::MIN, true));
    }

    for &step_size in STEPS {
        let mut n = NonZeroU16::MIN;
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(Step::forward_overflowing(n, step_size), (NonZeroU16::MAX, true));

        let mut n = NonZeroU16::MAX;
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(Step::backward_overflowing(n, step_size), (NonZeroU16::MIN, true));
    }

    // Test larger integers starting at MAX-2^16 and MIN+2^16
    for &step_size in STEPS {
        let mut n = u32::MAX - (1 << 16);
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(
            Step::forward_overflowing(n, step_size),
            (n.wrapping_add(step_size as u32), true)
        );

        let mut n = u32::MIN + (1 << 16);
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(
            Step::backward_overflowing(n, step_size),
            (n.wrapping_sub(step_size as u32), true)
        );
    }
    for &step_size in STEPS {
        let mut n = i32::MAX - (1 << 16);
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(
            Step::forward_overflowing(n, step_size),
            (n.wrapping_add_unsigned(step_size as u32), true)
        );

        let mut n = i32::MIN + (1 << 16);
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(
            Step::backward_overflowing(n, step_size),
            (n.wrapping_sub_unsigned(step_size as u32), true)
        );
    }

    for &step_size in STEPS {
        let mut n = u64::MAX - (1 << 16);
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(
            Step::forward_overflowing(n, step_size),
            (n.wrapping_add(step_size as u64), true)
        );

        let mut n = u64::MIN + (1 << 16);
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(
            Step::backward_overflowing(n, step_size),
            (n.wrapping_sub(step_size as u64), true)
        );
    }
    for &step_size in STEPS {
        let mut n = i64::MAX - (1 << 16);
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(
            Step::forward_overflowing(n, step_size),
            (n.wrapping_add_unsigned(step_size as u64), true)
        );

        let mut n = i64::MIN + (1 << 16);
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(
            Step::backward_overflowing(n, step_size),
            (n.wrapping_sub_unsigned(step_size as u64), true)
        );
    }

    for &step_size in STEPS {
        let mut n = u128::MAX - (1 << 16);
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(
            Step::forward_overflowing(n, step_size),
            (n.wrapping_add(step_size as u128), true)
        );

        let mut n = u128::MIN + (1 << 16);
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(
            Step::backward_overflowing(n, step_size),
            (n.wrapping_sub(step_size as u128), true)
        );
    }
    for &step_size in STEPS {
        let mut n = i128::MAX - (1 << 16);
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(
            Step::forward_overflowing(n, step_size),
            (n.wrapping_add_unsigned(step_size as u128), true)
        );

        let mut n = i128::MIN + (1 << 16);
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(
            Step::backward_overflowing(n, step_size),
            (n.wrapping_sub_unsigned(step_size as u128), true)
        );
    }

    for &step_size in STEPS {
        let mut n = Step::backward(NonZeroU32::MAX, 1 << 16);
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(Step::forward_overflowing(n, step_size), (NonZeroU32::MAX, true));

        let mut n = Step::forward(NonZeroU32::MIN, 1 << 16);
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(Step::backward_overflowing(n, step_size), (NonZeroU32::MIN, true));
    }

    for &step_size in STEPS {
        let mut n = Step::backward(NonZeroU64::MAX, 1 << 16);
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(Step::forward_overflowing(n, step_size), (NonZeroU64::MAX, true));

        let mut n = Step::forward(NonZeroU64::MIN, 1 << 16);
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(Step::backward_overflowing(n, step_size), (NonZeroU64::MIN, true));
    }

    for &step_size in STEPS {
        let mut n = if usize::BITS > 16 {
            usize::MAX - (1 << 16)
        } else {
            usize::MIN
        };
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(Step::forward_overflowing(n, step_size), (n.wrapping_add(step_size), true));

        let mut n = if usize::BITS > 16 {
            usize::MIN + (1 << 16)
        } else {
            usize::MAX
        };
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(Step::backward_overflowing(n, step_size), (n.wrapping_sub(step_size), true));
    }

    for &step_size in STEPS {
        let mut n = if usize::BITS > 16 {
            Step::backward(NonZeroUsize::MAX, 1 << 16)
        } else {
            NonZeroUsize::MIN
        };
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(Step::forward_overflowing(n, step_size), (NonZeroUsize::MAX, true));

        let mut n = if usize::BITS > 16 {
            Step::forward(NonZeroUsize::MIN, 1 << 16)
        } else {
            NonZeroUsize::MAX
        };
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(Step::backward_overflowing(n, step_size), (NonZeroUsize::MIN, true));
    }

    for &step_size in STEPS {
        let mut n = Step::backward(char::MAX, 1 << 16);
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(Step::forward_overflowing(n, step_size), (char::MAX, true));

        let mut n = Step::forward(char::MIN, 1 << 16);
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(Step::backward_overflowing(n, step_size), (char::MIN, true));
    }

    for &step_size in STEPS {
        let mut n = AsciiChar::MIN;
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(
            Step::forward_overflowing(n, step_size),
            (AsciiChar::from_u8(
                (n as u8).wrapping_add(step_size as u8) & (AsciiChar::MAX as u8)
            ).unwrap(), true)
        );

        let mut n = AsciiChar::MAX;
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(
            Step::backward_overflowing(n, step_size),
            (AsciiChar::from_u8(
                (n as u8).wrapping_sub(step_size as u8) & (AsciiChar::MAX as u8)
            ).unwrap(), true)
        );
    }

    for &step_size in STEPS {
        let mut n = Step::backward(Ipv4Addr::from_bits(u32::MAX), 1 << 16);
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(
            Step::forward_overflowing(n, step_size),
            (Ipv4Addr::from_bits(n.to_bits().wrapping_add(step_size as u32)), true)
        );

        let mut n = Step::forward(Ipv4Addr::from_bits(u32::MIN), 1 << 16);
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(
            Step::backward_overflowing(n, step_size),
            (Ipv4Addr::from_bits(n.to_bits().wrapping_sub(step_size as u32)), true)
        );
    }
    for &step_size in STEPS {
        let mut n = Step::backward(Ipv6Addr::from_bits(u128::MAX), 1 << 16);
        while let Some(p) = Step::forward_checked(n, step_size) {
            assert_eq!(Step::forward_overflowing(n, step_size), (p, false));
            n = p;
        }
        assert_eq!(
            Step::forward_overflowing(n, step_size),
            (Ipv6Addr::from_bits(n.to_bits().wrapping_add(step_size as u128)), true)
        );

        let mut n = Step::forward(Ipv6Addr::from_bits(u128::MIN), 1 << 16);
        while let Some(m) = Step::backward_checked(n, step_size) {
            assert_eq!(Step::backward_overflowing(n, step_size), (m, false));
            n = m;
        }
        assert_eq!(
            Step::backward_overflowing(n, step_size),
            (Ipv6Addr::from_bits(n.to_bits().wrapping_sub(step_size as u128)), true)
        );
    }
}
