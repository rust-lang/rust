use rustc_hir::BindingAnnotation;
use rustc_hir::BindingAnnotation::*;
use rustc_hir::Mutability;

#[derive(Clone, PartialEq, TyEncodable, TyDecodable, Debug, Copy, HashStable)]
pub enum BindingMode {
    BindByReference(Mutability),
    BindByValue(Mutability),
}

TrivialTypeFoldableAndLiftImpls! { BindingMode, }

impl BindingMode {
    pub fn convert(ba: BindingAnnotation) -> BindingMode {
        match ba {
            Unannotated => BindingMode::BindByValue(Mutability::Not),
            Mutable => BindingMode::BindByValue(Mutability::Mut),
            Ref => BindingMode::BindByReference(Mutability::Not),
            RefMut => BindingMode::BindByReference(Mutability::Mut),
        }
    }
}
