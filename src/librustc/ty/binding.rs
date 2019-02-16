use crate::hir::BindingAnnotation::*;
use crate::hir::BindingAnnotation;
use crate::hir::Mutability;

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug, Copy)]
pub enum BindingMode {
    BindByReference(Mutability),
    BindByValue(Mutability),
}

CloneTypeFoldableAndLiftImpls! { BindingMode, }

impl BindingMode {
    pub fn convert(ba: BindingAnnotation) -> BindingMode {
        match ba {
            Unannotated => BindingMode::BindByValue(Mutability::MutImmutable),
            Mutable => BindingMode::BindByValue(Mutability::MutMutable),
            Ref => BindingMode::BindByReference(Mutability::MutImmutable),
            RefMut => BindingMode::BindByReference(Mutability::MutMutable),
        }
    }
}

impl_stable_hash_for!(enum self::BindingMode {
    BindByReference(mutability),
    BindByValue(mutability)
});
