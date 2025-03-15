use std::fmt;

use rustc_data_structures::intern::Interned;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};

use crate::ty;

#[derive(Copy, Clone, PartialEq, Eq, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Pattern<'tcx>(pub Interned<'tcx, PatternKind<'tcx>>);

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

impl<'tcx> fmt::Debug for PatternKind<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
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
}

#[derive(Clone, PartialEq, Eq, Hash)]
#[derive(HashStable, TyEncodable, TyDecodable, TypeVisitable, TypeFoldable)]
pub enum PatternKind<'tcx> {
    Range { start: ty::Const<'tcx>, end: ty::Const<'tcx> },
    Or(&'tcx ty::List<Pattern<'tcx>>),
}
