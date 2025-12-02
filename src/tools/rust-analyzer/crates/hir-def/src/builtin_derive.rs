//! Definition of builtin derive impls.
//!
//! To save time and memory, builtin derives are not really expanded. Instead, we record them
//! and create their impls based on lowered data, see crates/hir-ty/src/builtin_derive.rs.

use hir_expand::builtin::BuiltinDeriveExpander;

macro_rules! declare_enum {
    ( $( $trait:ident ),* $(,)? ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BuiltinDeriveImplTrait {
            $( $trait, )*
        }

        impl BuiltinDeriveImplTrait {
            #[inline]
            pub fn get_id(self, lang_items: &crate::lang_item::LangItems) -> Option<crate::TraitId> {
                match self {
                    $( Self::$trait => lang_items.$trait, )*
                }
            }
        }
    };
}

declare_enum!(
    Copy,
    Clone,
    Default,
    Debug,
    Hash,
    Ord,
    PartialOrd,
    Eq,
    PartialEq,
    CoerceUnsized,
    DispatchFromDyn,
);

pub(crate) fn with_derive_traits(
    derive: BuiltinDeriveExpander,
    mut f: impl FnMut(BuiltinDeriveImplTrait),
) {
    let trait_ = match derive {
        BuiltinDeriveExpander::Copy => BuiltinDeriveImplTrait::Copy,
        BuiltinDeriveExpander::Clone => BuiltinDeriveImplTrait::Clone,
        BuiltinDeriveExpander::Default => BuiltinDeriveImplTrait::Default,
        BuiltinDeriveExpander::Debug => BuiltinDeriveImplTrait::Debug,
        BuiltinDeriveExpander::Hash => BuiltinDeriveImplTrait::Hash,
        BuiltinDeriveExpander::Ord => BuiltinDeriveImplTrait::Ord,
        BuiltinDeriveExpander::PartialOrd => BuiltinDeriveImplTrait::PartialOrd,
        BuiltinDeriveExpander::Eq => BuiltinDeriveImplTrait::Eq,
        BuiltinDeriveExpander::PartialEq => BuiltinDeriveImplTrait::PartialEq,
        BuiltinDeriveExpander::CoercePointee => {
            f(BuiltinDeriveImplTrait::CoerceUnsized);
            f(BuiltinDeriveImplTrait::DispatchFromDyn);
            return;
        }
    };
    f(trait_);
}
