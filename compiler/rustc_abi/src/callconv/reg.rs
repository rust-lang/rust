#[cfg(feature = "nightly")]
use rustc_macros::HashStable_Generic;

use crate::{Align, HasDataLayout, Size};

#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum RegKind {
    Integer,
    Float,
    Vector,
}

#[cfg_attr(feature = "nightly", derive(HashStable_Generic))]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct Reg {
    pub kind: RegKind,
    pub size: Size,
}

macro_rules! reg_ctor {
    ($name:ident, $kind:ident, $bits:expr) => {
        pub fn $name() -> Reg {
            Reg { kind: RegKind::$kind, size: Size::from_bits($bits) }
        }
    };
}

impl Reg {
    reg_ctor!(i8, Integer, 8);
    reg_ctor!(i16, Integer, 16);
    reg_ctor!(i32, Integer, 32);
    reg_ctor!(i64, Integer, 64);
    reg_ctor!(i128, Integer, 128);

    reg_ctor!(f32, Float, 32);
    reg_ctor!(f64, Float, 64);
}

impl Reg {
    pub fn align<C: HasDataLayout>(&self, cx: &C) -> Align {
        let dl = cx.data_layout();
        match self.kind {
            RegKind::Integer => match self.size.bits() {
                1 => dl.i1_align,
                2..=8 => dl.i8_align,
                9..=16 => dl.i16_align,
                17..=32 => dl.i32_align,
                33..=64 => dl.i64_align,
                65..=128 => dl.i128_align,
                _ => panic!("unsupported integer: {self:?}"),
            },
            RegKind::Float => match self.size.bits() {
                16 => dl.f16_align,
                32 => dl.f32_align,
                64 => dl.f64_align,
                128 => dl.f128_align,
                _ => panic!("unsupported float: {self:?}"),
            },
            RegKind::Vector => dl.llvmlike_vector_align(self.size),
        }
    }
}
