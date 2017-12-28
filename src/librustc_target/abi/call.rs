// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use abi::{Align, HasDataLayout, Size};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PassMode {
    /// Ignore the argument (useful for empty struct).
    Ignore,
    /// Pass the argument directly.
    Direct(ArgAttributes),
    /// Pass a pair's elements directly in two arguments.
    Pair(ArgAttributes, ArgAttributes),
    /// Pass the argument after casting it, to either
    /// a single uniform or a pair of registers.
    Cast(CastTarget),
    /// Pass the argument indirectly via a hidden pointer.
    Indirect(ArgAttributes),
}

// Hack to disable non_upper_case_globals only for the bitflags! and not for the rest
// of this module
pub use self::attr_impl::ArgAttribute;

#[allow(non_upper_case_globals)]
#[allow(unused)]
mod attr_impl {
    // The subset of llvm::Attribute needed for arguments, packed into a bitfield.
    bitflags! {
        #[derive(Default)]
        pub struct ArgAttribute: u16 {
            const ByVal     = 1 << 0;
            const NoAlias   = 1 << 1;
            const NoCapture = 1 << 2;
            const NonNull   = 1 << 3;
            const ReadOnly  = 1 << 4;
            const SExt      = 1 << 5;
            const StructRet = 1 << 6;
            const ZExt      = 1 << 7;
            const InReg     = 1 << 8;
        }
    }
}

/// A compact representation of LLVM attributes (at least those relevant for this module)
/// that can be manipulated without interacting with LLVM's Attribute machinery.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct ArgAttributes {
    pub regular: ArgAttribute,
    pub pointee_size: Size,
    pub pointee_align: Option<Align>
}

impl ArgAttributes {
    pub fn new() -> Self {
        ArgAttributes {
            regular: ArgAttribute::default(),
            pointee_size: Size::from_bytes(0),
            pointee_align: None,
        }
    }

    pub fn set(&mut self, attr: ArgAttribute) -> &mut Self {
        self.regular = self.regular | attr;
        self
    }

    pub fn contains(&self, attr: ArgAttribute) -> bool {
        self.regular.contains(attr)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum RegKind {
    Integer,
    Float,
    Vector
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Reg {
    pub kind: RegKind,
    pub size: Size,
}

macro_rules! reg_ctor {
    ($name:ident, $kind:ident, $bits:expr) => {
        pub fn $name() -> Reg {
            Reg {
                kind: RegKind::$kind,
                size: Size::from_bits($bits)
            }
        }
    }
}

impl Reg {
    reg_ctor!(i8, Integer, 8);
    reg_ctor!(i16, Integer, 16);
    reg_ctor!(i32, Integer, 32);
    reg_ctor!(i64, Integer, 64);

    reg_ctor!(f32, Float, 32);
    reg_ctor!(f64, Float, 64);
}

impl Reg {
    pub fn align<C: HasDataLayout>(&self, cx: C) -> Align {
        let dl = cx.data_layout();
        match self.kind {
            RegKind::Integer => {
                match self.size.bits() {
                    1 => dl.i1_align,
                    2...8 => dl.i8_align,
                    9...16 => dl.i16_align,
                    17...32 => dl.i32_align,
                    33...64 => dl.i64_align,
                    65...128 => dl.i128_align,
                    _ => panic!("unsupported integer: {:?}", self)
                }
            }
            RegKind::Float => {
                match self.size.bits() {
                    32 => dl.f32_align,
                    64 => dl.f64_align,
                    _ => panic!("unsupported float: {:?}", self)
                }
            }
            RegKind::Vector => dl.vector_align(self.size)
        }
    }
}

/// An argument passed entirely registers with the
/// same kind (e.g. HFA / HVA on PPC64 and AArch64).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Uniform {
    pub unit: Reg,

    /// The total size of the argument, which can be:
    /// * equal to `unit.size` (one scalar/vector)
    /// * a multiple of `unit.size` (an array of scalar/vectors)
    /// * if `unit.kind` is `Integer`, the last element
    ///   can be shorter, i.e. `{ i64, i64, i32 }` for
    ///   64-bit integers with a total size of 20 bytes
    pub total: Size,
}

impl From<Reg> for Uniform {
    fn from(unit: Reg) -> Uniform {
        Uniform {
            unit,
            total: unit.size
        }
    }
}

impl Uniform {
    pub fn align<C: HasDataLayout>(&self, cx: C) -> Align {
        self.unit.align(cx)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CastTarget {
    pub prefix: [Option<RegKind>; 8],
    pub prefix_chunk: Size,
    pub rest: Uniform,
}

impl From<Reg> for CastTarget {
    fn from(unit: Reg) -> CastTarget {
        CastTarget::from(Uniform::from(unit))
    }
}

impl From<Uniform> for CastTarget {
    fn from(uniform: Uniform) -> CastTarget {
        CastTarget {
            prefix: [None; 8],
            prefix_chunk: Size::from_bytes(0),
            rest: uniform
        }
    }
}

impl CastTarget {
    pub fn pair(a: Reg, b: Reg) -> CastTarget {
        CastTarget {
            prefix: [Some(a.kind), None, None, None, None, None, None, None],
            prefix_chunk: a.size,
            rest: Uniform::from(b)
        }
    }

    pub fn size<C: HasDataLayout>(&self, cx: C) -> Size {
        (self.prefix_chunk * self.prefix.iter().filter(|x| x.is_some()).count() as u64)
             .abi_align(self.rest.align(cx)) + self.rest.total
    }

    pub fn align<C: HasDataLayout>(&self, cx: C) -> Align {
        self.prefix.iter()
            .filter_map(|x| x.map(|kind| Reg { kind: kind, size: self.prefix_chunk }.align(cx)))
            .fold(cx.data_layout().aggregate_align.max(self.rest.align(cx)),
                |acc, align| acc.max(align))
    }
}