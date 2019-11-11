use crate::hir::BindingAnnotation::*;
use crate::hir::BindingAnnotation;
use crate::hir::Mutability;

#[derive(Clone, PartialEq, RustcEncodable, RustcDecodable, Debug, Copy)]
pub enum BindingMode {
    BindByReference(Mutability),
    BindByValue(Mutability),
}

CloneTypeFoldableAndLiftImpls! { BindingMode, }

impl BindingMode {
    pub fn convert(ba: BindingAnnotation) -> BindingMode {
        match ba {
            Unannotated => BindingMode::BindByValue(Mutability::Immutable),
            Mutable => BindingMode::BindByValue(Mutability::Mutable),
            Ref => BindingMode::BindByReference(Mutability::Immutable),
            RefMut => BindingMode::BindByReference(Mutability::Mutable),
        }
    }
}

impl_stable_hash_for!(enum self::BindingMode {
    BindByReference(mutability),
    BindByValue(mutability)
});
