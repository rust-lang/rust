//! Things related to generics in the next-trait-solver.

use hir_def::{GenericDefId, GenericParamId};

use crate::db::HirDatabase;

use super::SolverDefId;

use super::DbInterner;

pub(crate) fn generics(interner: DbInterner<'_>, def: SolverDefId) -> Generics<'_> {
    let db = interner.db;
    let def = match (def.try_into(), def) {
        (Ok(def), _) => def,
        (_, SolverDefId::InternedOpaqueTyId(id)) => match id.loc(db) {
            crate::ImplTraitId::ReturnTypeImplTrait(function_id, _) => function_id.into(),
            crate::ImplTraitId::TypeAliasImplTrait(type_alias_id, _) => type_alias_id.into(),
        },
        (_, SolverDefId::BuiltinDeriveImplId(id)) => {
            return crate::builtin_derive::generics_of(interner, id);
        }
        (_, SolverDefId::AnonConstId(id)) => {
            let loc = id.loc(db);
            let generic_def = loc.owner.generic_def(db);
            return if loc.allow_using_generic_params {
                Generics::from_generic_def(db, generic_def)
            } else {
                #[expect(
                    deprecated,
                    reason = "`Generics` only exposes an iterator over `GenericParamId`, \
                        so you cannot exploit the erroneous `crate::generics::Generics`"
                )]
                Generics {
                    generics: crate::generics::Generics::empty(generic_def),
                    additional_param: None,
                }
            };
        }
        _ => panic!("No generics for {def:?}"),
    };

    Generics::from_generic_def(db, def)
}

#[derive(Debug)]
pub struct Generics<'db> {
    generics: crate::generics::Generics<'db>,
    /// This is used for builtin derives, specifically `CoercePointee`.
    additional_param: Option<GenericParamId>,
}

impl<'db> Generics<'db> {
    pub(crate) fn from_generic_def(db: &'db dyn HirDatabase, def: GenericDefId) -> Generics<'db> {
        Generics { generics: crate::generics::generics(db, def), additional_param: None }
    }

    pub(crate) fn from_generic_def_plus_one(
        db: &'db dyn HirDatabase,
        def: GenericDefId,
        additional_param: GenericParamId,
    ) -> Generics<'db> {
        Generics {
            generics: crate::generics::generics(db, def),
            additional_param: Some(additional_param),
        }
    }

    pub(super) fn iter_id(&self) -> impl Iterator<Item = GenericParamId> {
        self.generics.iter_id().chain(self.additional_param)
    }
}

impl<'db> rustc_type_ir::inherent::GenericsOf<DbInterner<'db>> for Generics<'db> {
    fn count(&self) -> usize {
        self.generics.len() + usize::from(self.additional_param.is_some())
    }
}
