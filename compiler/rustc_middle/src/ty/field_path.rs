use std::ops::ControlFlow;

use rustc_abi::{FIRST_VARIANT, FieldIdx, VariantIdx};
use rustc_macros::{HashStable, TyEncodable};
use rustc_span::{Symbol, sym};

use crate::ty::{self, List, Ty, TyCtxt};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FieldPathKind {
    OffsetOf,
    FieldOf,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, HashStable, Hash, TyEncodable)]
pub struct FieldPath<'tcx>(pub &'tcx List<(VariantIdx, FieldIdx)>);

impl<'tcx> IntoIterator for FieldPath<'tcx> {
    type Item = <&'tcx List<(VariantIdx, FieldIdx)> as IntoIterator>::Item;

    type IntoIter = <&'tcx List<(VariantIdx, FieldIdx)> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'tcx> IntoIterator for &FieldPath<'tcx> {
    type Item = <&'tcx List<(VariantIdx, FieldIdx)> as IntoIterator>::Item;

    type IntoIter = <&'tcx List<(VariantIdx, FieldIdx)> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'tcx> FieldPath<'tcx> {
    pub fn iter(self) -> <Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    pub fn walk<T>(
        self,
        tcx: TyCtxt<'tcx>,
        container: Ty<'tcx>,
        mut walker: impl FnMut(Ty<'tcx>, Symbol, Ty<'tcx>, bool) -> ControlFlow<T>,
    ) -> Option<T> {
        let mut cur = container;
        for (i, (variant, field)) in self.iter().enumerate() {
            let last = i == self.0.len() - 1;
            let (name, field_ty) = match cur.kind() {
                ty::Adt(def, args) => {
                    let variant = def.variant(variant);
                    let field = &variant.fields[field];
                    let field_ty = field.ty(tcx, args);
                    (field.name, field_ty)
                }
                ty::Tuple(tys) => {
                    assert_eq!(FIRST_VARIANT, variant);
                    (sym::integer(field.index()), tys[field.index()])
                }
                _ => bug!("only ADTs and tuples are supported by `field_of!`, found {cur}"),
            };
            match walker(cur, name, field_ty, last) {
                ControlFlow::Break(val) => return Some(val),
                ControlFlow::Continue(()) => cur = field_ty,
            }
        }
        None
    }

    pub fn field_ty(self, tcx: TyCtxt<'tcx>, container: Ty<'tcx>) -> Ty<'tcx> {
        self.walk(tcx, container, |_, _, ty, last| {
            if last { ControlFlow::Break(ty) } else { ControlFlow::Continue(()) }
        })
        .expect("field path to have a last segment")
    }
}

impl<'tcx> rustc_type_ir::inherent::FieldPath<TyCtxt<'tcx>> for FieldPath<'tcx> {
    fn walk<T>(
        self,
        interner: TyCtxt<'tcx>,
        container: Ty<'tcx>,
        walker: impl FnMut(Ty<'tcx>, Symbol, Ty<'tcx>, bool) -> ControlFlow<T>,
    ) -> Option<T> {
        self.walk(interner, container, walker)
    }

    fn field_ty(self, interner: TyCtxt<'tcx>, container: Ty<'tcx>) -> Ty<'tcx> {
        self.field_ty(interner, container)
    }
}
