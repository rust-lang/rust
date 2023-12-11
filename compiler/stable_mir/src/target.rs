//! Provide information about the machine that this is being compiled into.

use crate::compiler_interface::with;

/// The properties of the target machine being compiled into.
#[derive(Clone, PartialEq, Eq)]
pub struct MachineInfo {
    pub endian: Endian,
    pub pointer_width: MachineSize,
}

impl MachineInfo {
    pub fn target() -> MachineInfo {
        with(|cx| cx.target_info())
    }

    pub fn target_endianess() -> Endian {
        with(|cx| cx.target_info().endian)
    }

    pub fn target_pointer_width() -> MachineSize {
        with(|cx| cx.target_info().pointer_width)
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Endian {
    Little,
    Big,
}

/// Represent the size of a component.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct MachineSize {
    num_bits: usize,
}

impl MachineSize {
    pub fn bytes(self) -> usize {
        self.num_bits / 8
    }

    pub fn bits(self) -> usize {
        self.num_bits
    }

    pub fn from_bits(num_bits: usize) -> MachineSize {
        MachineSize { num_bits }
    }
}
