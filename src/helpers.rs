use std::mem;

use rustc::ty;
use rustc::hir::def_id::{DefId, CRATE_DEF_INDEX};

use super::*;

pub trait ScalarExt {
    /// HACK: this function just extracts all bits if `defined != 0`
    /// Mainly used for args of C-functions and we should totally correctly fetch the size
    /// of their arguments
    fn to_bytes(self) -> EvalResult<'static, u128>;
}

impl ScalarExt for Scalar {
    fn to_bytes(self) -> EvalResult<'static, u128> {
        match self {
            Scalar::Bits { bits, size } => {
                assert_ne!(size, 0);
                Ok(bits)
            },
            Scalar::Ptr(_) => err!(ReadPointerAsBytes),
        }
    }
}

impl ScalarExt for ScalarMaybeUndef {
    fn to_bytes(self) -> EvalResult<'static, u128> {
        self.not_undef()?.to_bytes()
    }
}

pub trait EvalContextExt<'tcx> {
    fn resolve_path(&self, path: &[&str]) -> EvalResult<'tcx, ty::Instance<'tcx>>;
}


impl<'a, 'mir, 'tcx> EvalContextExt<'tcx> for EvalContext<'a, 'mir, 'tcx, super::Evaluator<'tcx>> {
    /// Get an instance for a path.
    fn resolve_path(&self, path: &[&str]) -> EvalResult<'tcx, ty::Instance<'tcx>> {
        self.tcx
            .crates()
            .iter()
            .find(|&&krate| self.tcx.original_crate_name(krate) == path[0])
            .and_then(|krate| {
                let krate = DefId {
                    krate: *krate,
                    index: CRATE_DEF_INDEX,
                };
                let mut items = self.tcx.item_children(krate);
                let mut path_it = path.iter().skip(1).peekable();

                while let Some(segment) = path_it.next() {
                    for item in mem::replace(&mut items, Default::default()).iter() {
                        if item.ident.name == *segment {
                            if path_it.peek().is_none() {
                                return Some(ty::Instance::mono(self.tcx.tcx, item.def.def_id()));
                            }

                            items = self.tcx.item_children(item.def.def_id());
                            break;
                        }
                    }
                }
                None
            })
            .ok_or_else(|| {
                let path = path.iter().map(|&s| s.to_owned()).collect();
                EvalErrorKind::PathNotFound(path).into()
            })
    }
}
