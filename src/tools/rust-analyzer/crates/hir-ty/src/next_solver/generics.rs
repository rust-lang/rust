//! Things related to generics in the next-trait-solver.

use hir_def::{
    ConstParamId, GenericDefId, GenericParamId, ItemContainerId, LifetimeParamId, Lookup,
    TypeOrConstParamId, TypeParamId,
    db::DefDatabase,
    expr_store::ExpressionStore,
    hir::generics::{
        GenericParamDataRef, GenericParams, LifetimeParamData, LocalLifetimeParamId,
        LocalTypeOrConstParamId, TypeOrConstParamData, TypeParamData, TypeParamProvenance,
        WherePredicate,
    },
};
use hir_expand::name::Name;
use intern::Symbol;
use la_arena::Arena;
use rustc_type_ir::inherent::Ty as _;
use triomphe::Arc;

use crate::{db::HirDatabase, generics::parent_generic_def, next_solver::Ty};

use super::{Const, EarlyParamRegion, ErrorGuaranteed, ParamConst, Region, SolverDefId};

use super::{DbInterner, GenericArg};

pub(crate) fn generics(db: &dyn HirDatabase, def: SolverDefId) -> Generics {
    let mk_lt = |(index, (_, lt)): (usize, (_, &LifetimeParamData))| {
        let name = lt.name.symbol().clone();
        let index = index as u32;
        let kind = GenericParamDefKind::Lifetime;
        GenericParamDef { name, index, kind }
    };
    let mk_ty = |len_lt, (index, p): (usize, &TypeOrConstParamData)| {
        let name = p
            .name()
            .map(|n| n.symbol().clone())
            .unwrap_or_else(|| Name::missing().symbol().clone());
        let index = (len_lt + index) as u32;
        let kind = match p {
            TypeOrConstParamData::TypeParamData(_) => GenericParamDefKind::Type,
            TypeOrConstParamData::ConstParamData(_) => GenericParamDefKind::Const,
        };
        GenericParamDef { name, index, kind }
    };
    let own_params_for_generic_params = |params: &GenericParams| {
        if params.trait_self_param().is_some() {
            let len_lt = params.len_lifetimes() + 1;
            params
                .iter_type_or_consts()
                .take(1)
                .enumerate()
                .map(|t| mk_ty(0, (t.0, t.1.1)))
                .chain(params.iter_lt().enumerate().map(mk_lt))
                .chain(
                    params
                        .iter_type_or_consts()
                        .skip(1)
                        .enumerate()
                        .map(|t| mk_ty(len_lt, (t.0, t.1.1))),
                )
                .collect()
        } else {
            let len_lt = params.len_lifetimes();
            params
                .iter_lt()
                .enumerate()
                .map(mk_lt)
                .chain(
                    params.iter_type_or_consts().enumerate().map(|t| mk_ty(len_lt, (t.0, t.1.1))),
                )
                .collect()
        }
    };

    let (parent, own_params) = match (def.try_into(), def) {
        (Ok(def), _) => {
            (parent_generic_def(db, def), own_params_for_generic_params(&db.generic_params(def)))
        }
        (_, SolverDefId::InternedOpaqueTyId(id)) => {
            match db.lookup_intern_impl_trait_id(id) {
                crate::ImplTraitId::ReturnTypeImplTrait(function_id, _) => {
                    // The opaque type itself does not have generics - only the parent function
                    (Some(GenericDefId::FunctionId(function_id)), vec![])
                }
                crate::ImplTraitId::TypeAliasImplTrait(type_alias_id, _) => (
                    None,
                    own_params_for_generic_params(
                        &db.generic_params(GenericDefId::TypeAliasId(type_alias_id)),
                    ),
                ),
                crate::ImplTraitId::AsyncBlockTypeImplTrait(def, _) => {
                    let param = TypeOrConstParamData::TypeParamData(TypeParamData {
                        name: None,
                        default: None,
                        provenance: TypeParamProvenance::TypeParamList,
                    });
                    // Yes, there is a parent but we don't include it in the generics
                    (None, vec![mk_ty(0, (0, &param))])
                }
            }
        }
        _ => panic!("No generics for {def:?}"),
    };
    let parent_generics = parent.map(|def| Box::new(generics(db, def.into())));

    Generics {
        parent,
        parent_count: parent_generics.map_or(0, |g| g.parent_count + g.own_params.len()),
        own_params,
    }
}

#[derive(Debug)]
pub struct Generics {
    pub parent: Option<GenericDefId>,
    pub parent_count: usize,
    pub own_params: Vec<GenericParamDef>,
}

#[derive(Debug)]
pub struct GenericParamDef {
    pub(crate) name: Symbol,
    //def_id: GenericDefId,
    index: u32,
    pub(crate) kind: GenericParamDefKind,
}

impl GenericParamDef {
    /// Returns the index of the param on the self generics only
    /// (i.e. not including parent generics)
    pub fn index(&self) -> u32 {
        self.index
    }
}

#[derive(Copy, Clone, Debug)]
pub enum GenericParamDefKind {
    Lifetime,
    Type,
    Const,
}

impl<'db> rustc_type_ir::inherent::GenericsOf<DbInterner<'db>> for Generics {
    fn count(&self) -> usize {
        self.parent_count + self.own_params.len()
    }
}
