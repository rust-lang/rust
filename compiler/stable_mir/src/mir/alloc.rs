use crate::mir::mono::{Instance, StaticDef};
use crate::ty::{Allocation, Binder, ExistentialTraitRef, IndexedVal, Ty};
use crate::with;

/// An allocation in the SMIR global memory can be either a function pointer,
/// a static, or a "real" allocation with some data in it.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum GlobalAlloc {
    /// The alloc ID is used as a function pointer.
    Function(Instance),
    /// This alloc ID points to a symbolic (not-reified) vtable.
    /// The `None` trait ref is used to represent auto traits.
    VTable(Ty, Option<Binder<ExistentialTraitRef>>),
    /// The alloc ID points to a "lazy" static variable that did not get computed (yet).
    /// This is also used to break the cycle in recursive statics.
    Static(StaticDef),
    /// The alloc ID points to memory.
    Memory(Allocation),
}

impl From<AllocId> for GlobalAlloc {
    fn from(value: AllocId) -> Self {
        with(|cx| cx.global_alloc(value))
    }
}

impl GlobalAlloc {
    pub fn vtable_allocation(&self) -> Option<AllocId> {
        with(|cx| cx.vtable_allocation(self))
    }
}

/// A unique identification number for each provenance
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AllocId(usize);

impl IndexedVal for AllocId {
    fn to_val(index: usize) -> Self {
        AllocId(index)
    }
    fn to_index(&self) -> usize {
        self.0
    }
}
