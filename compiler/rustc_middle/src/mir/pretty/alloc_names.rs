use std::cell::RefCell;
use std::fmt::{self, Debug, Display, Formatter};

use rustc_data_structures::fx::FxIndexSet;

use super::{RenderAllocation, pretty_print_const_value_tcx};
use crate::mir::interpret::{AllocBytes, AllocId, CtfeProvenance, Pointer, Scalar};
use crate::mir::{Const, ConstOperand, ConstValue, VarDebugInfoContents};
use crate::ty;

/// Owned by one emitted file. IDs are keys, never an ordering criterion.
#[derive(Default)]
pub(super) struct AllocationNames(RefCell<FxIndexSet<AllocId>>);

/// Static dispatch keeps the ordinary printers free of a naming table or a mode check.
/// Only the explicit MIR entry point supplies `&AllocationNames`.
pub(super) trait AllocIdPrinter: Copy {
    const USE_LOCAL_ALLOC_NAMES: bool;
    type Id: Debug;
    type Ptr: Debug;

    fn id(self, id: AllocId) -> Self::Id;
    fn pointer(self, ptr: Pointer) -> Self::Ptr;

    fn wrap<T>(self, value: T) -> WithAllocNames<T, Self> {
        WithAllocNames { value, alloc_id_printer: self }
    }
}

#[derive(Clone, Copy)]
pub(super) struct RawAllocIds;

impl AllocIdPrinter for RawAllocIds {
    const USE_LOCAL_ALLOC_NAMES: bool = false;
    type Id = AllocId;
    type Ptr = Pointer;

    fn id(self, id: AllocId) -> AllocId {
        id
    }

    fn pointer(self, ptr: Pointer) -> Pointer {
        ptr
    }
}

impl AllocIdPrinter for &AllocationNames {
    const USE_LOCAL_ALLOC_NAMES: bool = true;
    type Id = AllocationName;
    type Ptr = NamedPointer;

    fn id(self, id: AllocId) -> AllocationName {
        AllocationName(self.0.borrow_mut().insert_full(id).0 + 1)
    }

    fn pointer(self, ptr: Pointer) -> NamedPointer {
        NamedPointer { ptr, name: self.id(ptr.provenance.alloc_id()) }
    }
}

pub(super) struct AllocationName(usize);

impl Debug for AllocationName {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if f.alternate() { write!(f, "a{}", self.0) } else { write!(f, "alloc{}", self.0) }
    }
}

pub(super) struct NamedPointer {
    ptr: Pointer,
    name: AllocationName,
}

impl Debug for NamedPointer {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        CtfeProvenance::fmt_pointer_with_alloc_id(&self.ptr, f, &self.name)
    }
}

pub(super) struct WithAllocNames<T, N> {
    pub(super) value: T,
    pub(super) alloc_id_printer: N,
}

impl<N: AllocIdPrinter> Debug for WithAllocNames<&VarDebugInfoContents<'_>, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.value {
            VarDebugInfoContents::Const(c) => write!(f, "{}", self.alloc_id_printer.wrap(c)),
            VarDebugInfoContents::Place(p) => write!(f, "{p:?}"),
        }
    }
}

impl<N: AllocIdPrinter> Debug for WithAllocNames<&ConstOperand<'_>, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<N: AllocIdPrinter> Display for WithAllocNames<&Const<'_>, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if !N::USE_LOCAL_ALLOC_NAMES {
            return Display::fmt(self.value, f);
        }
        match *self.value {
            Const::Val(value, ty) => ty::tls::with(|tcx| {
                pretty_print_const_value_tcx(tcx, value, tcx.lift(ty), f, self.alloc_id_printer)
            }),
            _ => Display::fmt(self.value, f),
        }
    }
}

impl<N: AllocIdPrinter> Debug for WithAllocNames<&Scalar, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.value {
            Scalar::Ptr(ptr, _) => write!(f, "{:?}", self.alloc_id_printer.pointer(*ptr)),
            Scalar::Int(value) => write!(f, "{value:?}"),
        }
    }
}

impl<N: AllocIdPrinter> Debug for WithAllocNames<&ConstValue, N> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.value {
            ConstValue::Scalar(value) => {
                f.debug_tuple("Scalar").field(&self.alloc_id_printer.wrap(value)).finish()
            }
            ConstValue::ZeroSized => f.write_str("ZeroSized"),
            ConstValue::Slice { alloc_id, meta } => f
                .debug_struct("Slice")
                .field("alloc_id", &self.alloc_id_printer.id(*alloc_id))
                .field("meta", meta)
                .finish(),
            ConstValue::Indirect { alloc_id, offset } => f
                .debug_struct("Indirect")
                .field("alloc_id", &self.alloc_id_printer.id(*alloc_id))
                .field("offset", offset)
                .finish(),
        }
    }
}

impl<Extra, Bytes: AllocBytes, N: AllocIdPrinter> Display
    for WithAllocNames<RenderAllocation<'_, '_, CtfeProvenance, Extra, Bytes>, N>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.value.fmt_with_pointer_formatter(f, |ptr| self.alloc_id_printer.pointer(ptr))
    }
}

#[cfg(test)]
mod tests;
