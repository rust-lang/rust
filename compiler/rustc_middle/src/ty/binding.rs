use crate::ty::Mutability;
use rustc_hir::{self as hir, BindingAnnotation, ByRef};

#[derive(Clone, PartialEq, TyEncodable, TyDecodable, Debug, Copy, HashStable)]
pub enum BindingMode {
    BindByReference(Mutability),
    BindByValue(Mutability),
}

TrivialTypeTraversalImpls! { BindingMode }

impl BindingMode {
    pub fn convert(BindingAnnotation(by_ref, mutbl): BindingAnnotation) -> BindingMode {
        let mutbl = match mutbl {
            hir::Mutability::Not => Mutability::Not,
            hir::Mutability::Mut => Mutability::Mut,
        };

        match by_ref {
            ByRef::No => BindingMode::BindByValue(mutbl),
            ByRef::Yes => BindingMode::BindByReference(mutbl),
        }
    }
}
