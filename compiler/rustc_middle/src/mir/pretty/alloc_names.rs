use std::cell::RefCell;
use std::fmt::{self, Debug, Display, Formatter};

use rustc_data_structures::fx::FxIndexSet;

use super::{RenderAllocation, pretty_print_const_value_tcx};
use crate::mir::interpret::{AllocBytes, AllocId, CtfeProvenance, Pointer, Scalar};
use crate::mir::{Const, ConstValue};
use crate::ty;

#[derive(Copy, Clone)]
pub(super) struct AllocationNames<'a>(pub(super) Option<&'a RefCell<FxIndexSet<AllocId>>>);

impl<'a> AllocationNames<'a> {
    pub(super) fn wrap<T>(self, value: T) -> MirFormat<'a, T> {
        MirFormat { value, names: self }
    }
}

// Formatting a nested operand directly with `Debug` can print rustc's internal `AllocId`.
// Its number can differ from the `allocN` used for the same allocation in the MIR output.
pub(super) struct MirFormat<'a, T> {
    pub(super) value: T,
    pub(super) names: AllocationNames<'a>,
}

impl Debug for MirFormat<'_, AllocId> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let Some(names) = self.names.0 else { return Debug::fmt(&self.value, f) };
        let number = names.borrow_mut().insert_full(self.value).0 + 1;
        if f.alternate() { write!(f, "a{number}") } else { write!(f, "alloc{number}") }
    }
}

impl Debug for MirFormat<'_, Pointer> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        CtfeProvenance::fmt_pointer(
            &self.value,
            f,
            self.names.wrap(self.value.provenance.alloc_id()),
        )
    }
}

impl Display for MirFormat<'_, &Const<'_>> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.names.0.is_none() {
            return Display::fmt(self.value, f);
        }
        match *self.value {
            Const::Val(value, ty) => ty::tls::with(|tcx| {
                pretty_print_const_value_tcx(tcx, value, tcx.lift(ty), f, self.names)
            }),
            Const::Ty(..) | Const::Unevaluated(..) => Display::fmt(self.value, f),
        }
    }
}

impl Debug for MirFormat<'_, &ConstValue> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self.value {
            ConstValue::Scalar(Scalar::Ptr(ptr, _)) => {
                f.debug_tuple("Scalar").field(&format_args!("{:?}", self.names.wrap(*ptr))).finish()
            }
            ConstValue::Scalar(Scalar::Int(_)) | ConstValue::ZeroSized => Debug::fmt(self.value, f),
            ConstValue::Slice { alloc_id, meta } => f
                .debug_struct("Slice")
                .field("alloc_id", &self.names.wrap(*alloc_id))
                .field("meta", meta)
                .finish(),
            ConstValue::Indirect { alloc_id, offset } => f
                .debug_struct("Indirect")
                .field("alloc_id", &self.names.wrap(*alloc_id))
                .field("offset", offset)
                .finish(),
        }
    }
}

impl<Extra, Bytes: AllocBytes> Display
    for MirFormat<'_, RenderAllocation<'_, '_, CtfeProvenance, Extra, Bytes>>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.value.fmt_contents(f, |ptr| self.names.wrap(ptr))
    }
}

#[cfg(test)]
mod tests;
