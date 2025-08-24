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
use intern::{Symbol, sym};
use la_arena::Arena;
use rustc_type_ir::inherent::Ty as _;
use triomphe::Arc;

use crate::{db::HirDatabase, generics::parent_generic_def, next_solver::Ty};

use super::{Const, EarlyParamRegion, ErrorGuaranteed, ParamConst, Region, SolverDefId};

use super::{DbInterner, GenericArg};

pub(crate) fn generics(db: &dyn HirDatabase, def: SolverDefId) -> Generics {
    let mk_lt = |index, lt: &LifetimeParamData| {
        let name = lt.name.symbol().clone();
        let kind = GenericParamDefKind::Lifetime;
        GenericParamDef { name, index, kind }
    };
    let mk_ty = |index, p: &TypeOrConstParamData| {
        let name = p.name().map(|n| n.symbol().clone()).unwrap_or_else(|| sym::MISSING_NAME);
        let kind = match p {
            TypeOrConstParamData::TypeParamData(_) => GenericParamDefKind::Type,
            TypeOrConstParamData::ConstParamData(_) => GenericParamDefKind::Const,
        };
        GenericParamDef { name, index, kind }
    };
    let own_params_for_generic_params = |params: &GenericParams| {
        let mut result = Vec::with_capacity(params.len());
        let mut type_and_consts = params.iter_type_or_consts();
        let mut index = 0;
        if let Some(self_param) = params.trait_self_param() {
            result.push(mk_ty(0, &params[self_param]));
            type_and_consts.next();
            index += 1;
        }
        result.extend(params.iter_lt().map(|(_, data)| {
            let lt = mk_lt(index, data);
            index += 1;
            lt
        }));
        result.extend(type_and_consts.map(|(_, data)| {
            let ty = mk_ty(index, data);
            index += 1;
            ty
        }));
        result
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
                crate::ImplTraitId::TypeAliasImplTrait(type_alias_id, _) => {
                    (Some(type_alias_id.into()), Vec::new())
                }
                crate::ImplTraitId::AsyncBlockTypeImplTrait(def, _) => {
                    let param = TypeOrConstParamData::TypeParamData(TypeParamData {
                        name: None,
                        default: None,
                        provenance: TypeParamProvenance::TypeParamList,
                    });
                    // Yes, there is a parent but we don't include it in the generics
                    (None, vec![mk_ty(0, &param)])
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
