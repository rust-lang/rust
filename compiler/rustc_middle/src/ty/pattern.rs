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
            PatternKind::Range { start, end, include_end } => {
                if let Some(start) = start {
                    write!(f, "{start}")?;
                }
                write!(f, "..")?;
                if include_end {
                    write!(f, "=")?;
                }
                if let Some(end) = end {
                    write!(f, "{end}")?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
#[derive(HashStable, TyEncodable, TyDecodable, TypeVisitable, TypeFoldable)]
pub enum PatternKind<'tcx> {
    Range { start: Option<ty::Const<'tcx>>, end: Option<ty::Const<'tcx>>, include_end: bool },
}
