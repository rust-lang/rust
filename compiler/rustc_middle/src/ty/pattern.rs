use std::fmt;

use rustc_data_structures::intern::Interned;
use rustc_macros::HashStable;
use rustc_type_ir::ir_print::IrPrint;
use rustc_type_ir::{
    FlagComputation, Flags, {self as ir},
};

use super::TyCtxt;
use crate::ty;

pub type PatternKind<'tcx> = ir::PatternKind<TyCtxt<'tcx>>;

#[derive(Copy, Clone, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Pattern<'tcx>(pub Interned<'tcx, PatternKind<'tcx>>);

impl<'tcx> Flags for Pattern<'tcx> {
    fn flags(&self) -> rustc_type_ir::TypeFlags {
        match &**self {
            ty::PatternKind::Range { start, end } => {
                FlagComputation::for_const_kind(&start.kind()).flags
                    | FlagComputation::for_const_kind(&end.kind()).flags
            }
            ty::PatternKind::Or(pats) => {
                let mut flags = pats[0].flags();
                for pat in pats[1..].iter() {
                    flags |= pat.flags();
                }
                flags
            }
        }
    }

    fn outer_exclusive_binder(&self) -> rustc_type_ir::DebruijnIndex {
        match &**self {
            ty::PatternKind::Range { start, end } => {
                start.outer_exclusive_binder().max(end.outer_exclusive_binder())
            }
            ty::PatternKind::Or(pats) => {
                let mut idx = pats[0].outer_exclusive_binder();
                for pat in pats[1..].iter() {
                    idx = idx.max(pat.outer_exclusive_binder());
                }
                idx
            }
        }
    }
}

impl<'tcx> std::ops::Deref for Pattern<'tcx> {
    type Target = PatternKind<'tcx>;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl<'tcx> fmt::Debug for Pattern<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", **self)
    }
}

impl<'tcx> IrPrint<PatternKind<'tcx>> for TyCtxt<'tcx> {
    fn print(t: &PatternKind<'tcx>, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *t {
            PatternKind::Range { start, end } => {
                write!(f, "{start}")?;

                if let Some(c) = end.try_to_value() {
                    let end = c.valtree.unwrap_leaf();
                    let size = end.size();
                    let max = match c.ty.kind() {
                        ty::Int(_) => {
                            Some(ty::ScalarInt::truncate_from_int(size.signed_int_max(), size))
                        }
                        ty::Uint(_) => {
                            Some(ty::ScalarInt::truncate_from_uint(size.unsigned_int_max(), size))
                        }
                        ty::Char => Some(ty::ScalarInt::truncate_from_uint(char::MAX, size)),
                        _ => None,
                    };
                    if let Some((max, _)) = max
                        && end == max
                    {
                        return write!(f, "..");
                    }
                }

                write!(f, "..={end}")
            }
            PatternKind::Or(patterns) => {
                write!(f, "(")?;
                let mut first = true;
                for pat in patterns {
                    if first {
                        first = false
                    } else {
                        write!(f, " | ")?;
                    }
                    write!(f, "{pat:?}")?;
                }
                write!(f, ")")
            }
        }
    }

    fn print_debug(t: &PatternKind<'tcx>, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        Self::print(t, fmt)
    }
}

impl<'tcx> rustc_type_ir::inherent::IntoKind for Pattern<'tcx> {
    type Kind = PatternKind<'tcx>;
    fn kind(self) -> Self::Kind {
        *self
    }
}
