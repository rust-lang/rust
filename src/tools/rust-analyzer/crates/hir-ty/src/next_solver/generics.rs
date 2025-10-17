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
    let mk_lt = |parent, index, local_id, lt: &LifetimeParamData| {
        let name = lt.name.symbol().clone();
        let id = GenericParamId::LifetimeParamId(LifetimeParamId { parent, local_id });
        GenericParamDef { name, index, id }
    };
    let mk_ty = |parent, index, local_id, p: &TypeOrConstParamData| {
        let name = p.name().map(|n| n.symbol().clone()).unwrap_or_else(|| sym::MISSING_NAME);
        let id = TypeOrConstParamId { parent, local_id };
        let id = match p {
            TypeOrConstParamData::TypeParamData(_) => {
                GenericParamId::TypeParamId(TypeParamId::from_unchecked(id))
            }
            TypeOrConstParamData::ConstParamData(_) => {
                GenericParamId::ConstParamId(ConstParamId::from_unchecked(id))
            }
        };
        GenericParamDef { name, index, id }
    };
    let own_params_for_generic_params = |parent, params: &GenericParams| {
        let mut result = Vec::with_capacity(params.len());
        let mut type_and_consts = params.iter_type_or_consts();
        let mut index = 0;
        if let Some(self_param) = params.trait_self_param() {
            result.push(mk_ty(parent, 0, self_param, &params[self_param]));
            type_and_consts.next();
            index += 1;
        }
        result.extend(params.iter_lt().map(|(local_id, data)| {
            let lt = mk_lt(parent, index, local_id, data);
            index += 1;
            lt
        }));
        result.extend(type_and_consts.map(|(local_id, data)| {
            let ty = mk_ty(parent, index, local_id, data);
            index += 1;
            ty
        }));
        result
    };

    let (parent, own_params) = match (def.try_into(), def) {
        (Ok(def), _) => (
            parent_generic_def(db, def),
            own_params_for_generic_params(def, &db.generic_params(def)),
        ),
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
                    // FIXME: It seems utterly sensitive to fake a generic param here.
                    // Also, what a horrible mess!
                    (
                        None,
                        vec![mk_ty(
                            GenericDefId::FunctionId(salsa::plumbing::FromId::from_id(unsafe {
                                salsa::Id::from_index(salsa::Id::MAX_U32 - 1)
                            })),
                            0,
                            LocalTypeOrConstParamId::from_raw(la_arena::RawIdx::from_u32(0)),
                            &param,
                        )],
                    )
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
    pub(crate) id: GenericParamId,
}

impl GenericParamDef {
    /// Returns the index of the param on the self generics only
    /// (i.e. not including parent generics)
    pub fn index(&self) -> u32 {
        self.index
    }
}

impl<'db> rustc_type_ir::inherent::GenericsOf<DbInterner<'db>> for Generics {
    fn count(&self) -> usize {
        self.parent_count + self.own_params.len()
    }
}
