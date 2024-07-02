//! Provide information about the machine that this is being compiled into.

use crate::compiler_interface::with;
use serde::Serialize;

/// The properties of the target machine being compiled into.
#[derive(Clone, PartialEq, Eq, Serialize)]
pub struct MachineInfo {
    pub endian: Endian,
    pub pointer_width: MachineSize,
}

impl MachineInfo {
    pub fn target() -> MachineInfo {
        with(|cx| cx.target_info())
    }

    pub fn target_endianness() -> Endian {
        with(|cx| cx.target_info().endian)
    }

    pub fn target_pointer_width() -> MachineSize {
        with(|cx| cx.target_info().pointer_width)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Serialize)]
pub enum Endian {
    Little,
    Big,
}

/// Represent the size of a component.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Serialize)]
pub struct MachineSize {
    num_bits: usize,
}

impl MachineSize {
    #[inline(always)]
    pub fn bytes(self) -> usize {
        self.num_bits / 8
    }

    #[inline(always)]
    pub fn bits(self) -> usize {
        self.num_bits
    }

    #[inline(always)]
    pub fn from_bits(num_bits: usize) -> MachineSize {
        MachineSize { num_bits }
    }

    #[inline]
    pub fn unsigned_int_max(self) -> Option<u128> {
        (self.num_bits <= 128).then(|| u128::MAX >> (128 - self.bits()))
    }
}
