//! Definition of builtin derive impls.
//!
//! To save time and memory, builtin derives are not really expanded. Instead, we record them
//! and create their impls based on lowered data, see crates/hir-ty/src/builtin_derive.rs.

use hir_expand::{InFile, builtin::BuiltinDeriveExpander, name::Name};
use intern::{Symbol, sym};
use tt::TextRange;

use crate::{
    AdtId, BuiltinDeriveImplId, BuiltinDeriveImplLoc, FunctionId, HasModule, db::DefDatabase,
};

macro_rules! declare_enum {
    ( $( $trait:ident => [ $( $method:ident ),* ] ),* $(,)? ) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum BuiltinDeriveImplTrait {
            $( $trait, )*
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[allow(non_camel_case_types)]
        pub enum BuiltinDeriveImplMethod {
            $( $( $method, )* )*
        }

        impl BuiltinDeriveImplTrait {
            #[inline]
            pub fn name(self) -> Symbol {
                match self {
                    $( Self::$trait => sym::$trait, )*
                }
            }

            #[inline]
            pub fn get_id(self, lang_items: &crate::lang_item::LangItems) -> Option<crate::TraitId> {
                match self {
                    $( Self::$trait => lang_items.$trait, )*
                }
            }

            #[inline]
            pub fn get_method(self, method_name: &Symbol) -> Option<BuiltinDeriveImplMethod> {
                match self {
                    $(
                        Self::$trait => {
                            match method_name {
                                $( _ if *method_name == sym::$method => Some(BuiltinDeriveImplMethod::$method), )*
                                _ => None,
                            }
                        }
                    )*
                }
            }

            #[inline]
            pub fn all_methods(self) -> &'static [BuiltinDeriveImplMethod] {
                match self {
                    $( Self::$trait => &[ $(BuiltinDeriveImplMethod::$method),* ], )*
                }
            }
        }

        impl BuiltinDeriveImplMethod {
            #[inline]
            pub fn name(self) -> Symbol {
                match self {
                    $( $( BuiltinDeriveImplMethod::$method => sym::$method, )* )*
                }
            }
        }
    };
}

declare_enum!(
    Copy => [],
    Clone => [clone],
    Default => [default],
    Debug => [fmt],
    Hash => [hash],
    Ord => [cmp],
    PartialOrd => [partial_cmp],
    Eq => [],
    PartialEq => [eq],
    CoerceUnsized => [],
    DispatchFromDyn => [],
);

impl BuiltinDeriveImplMethod {
    pub fn trait_method(
        self,
        db: &dyn DefDatabase,
        impl_: BuiltinDeriveImplId,
    ) -> Option<FunctionId> {
        let loc = impl_.loc(db);
        let lang_items = crate::lang_item::lang_items(db, loc.krate(db));
        let trait_ = impl_.loc(db).trait_.get_id(lang_items)?;
        trait_.trait_items(db).method_by_name(&Name::new_symbol_root(self.name()))
    }
}

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

impl BuiltinDeriveImplLoc {
    pub fn source(&self, db: &dyn DefDatabase) -> InFile<TextRange> {
        let (adt_ast_id, module) = match self.adt {
            AdtId::StructId(adt) => {
                let adt_loc = adt.loc(db);
                (adt_loc.id.upcast(), adt_loc.container)
            }
            AdtId::UnionId(adt) => {
                let adt_loc = adt.loc(db);
                (adt_loc.id.upcast(), adt_loc.container)
            }
            AdtId::EnumId(adt) => {
                let adt_loc = adt.loc(db);
                (adt_loc.id.upcast(), adt_loc.container)
            }
        };
        let derive_range = self.derive_attr_id.find_derive_range(
            db,
            module.krate(db),
            adt_ast_id,
            self.derive_index,
        );
        adt_ast_id.with_value(derive_range)
    }
}
