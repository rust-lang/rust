//@ compile-flags: --check-cfg=cfg(target_has_reliable_f16b)
//@ run-pass

#![feature(f16b, cfg_target_has_reliable_f16b)]

extern crate core;

use core::num::f16b;
#[cfg(target_has_reliable_f16b)]
use std::fmt::{Debug, LowerExp, UpperExp};

#[cfg(target_has_reliable_f16b)]
const ONE: f16b = f16b::from_bits(0x3f80);

#[cfg(target_has_reliable_f16b)]
const ONE_BITS: u16 = ONE.to_bits();

#[cfg(target_has_reliable_f16b)]
fn assert_traits<T>()
where
    T: Default + Copy + Clone + Debug + LowerExp + UpperExp + PartialEq + PartialOrd,
{
}

fn main() {
    assert_eq!(size_of::<f16b>(), 2);
    assert_eq!(align_of::<f16b>(), 2);

    #[cfg(target_has_reliable_f16b)]
    {
        assert_traits::<f16b>();

        assert_eq!(f16b::default().to_bits(), 0);
        assert_eq!(ONE_BITS, 0x3f80);
        assert_eq!(f32::from(ONE).to_bits(), 0x3f80_0000);

        let two = f16b::from_bits(0x4000);
        let negative_zero = f16b::from_bits(0x8000);
        let nan = f16b::from_bits(0x7fc0);
        assert!(ONE < two);
        assert_eq!(f16b::from_bits(0), negative_zero);
        assert!(nan != nan);
        assert_eq!(format!("{ONE:?}"), "1.0");
        assert_eq!(format!("{ONE:e}"), "1e0");
        assert_eq!(format!("{ONE:E}"), "1E0");
    }
}
