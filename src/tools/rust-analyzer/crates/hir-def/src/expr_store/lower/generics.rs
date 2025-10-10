//! Many kinds of items or constructs can have generic parameters: functions,
//! structs, impls, traits, etc. This module provides a common HIR for these
//! generic parameters. See also the `Generics` type and the `generics_of` query
//! in rustc.

use std::sync::LazyLock;

use either::Either;
use hir_expand::name::{AsName, Name};
use intern::sym;
use la_arena::Arena;
use syntax::ast::{self, HasName, HasTypeBounds};
use thin_vec::ThinVec;
use triomphe::Arc;

use crate::{
    GenericDefId, TypeOrConstParamId, TypeParamId,
    expr_store::{TypePtr, lower::ExprCollector},
    hir::generics::{
        ConstParamData, GenericParams, LifetimeParamData, TypeOrConstParamData, TypeParamData,
        TypeParamProvenance, WherePredicate,
    },
    type_ref::{LifetimeRef, LifetimeRefId, TypeBound, TypeRef, TypeRefId},
};

pub(crate) type ImplTraitLowerFn<'l> = &'l mut dyn for<'ec, 'db> FnMut(
    &'ec mut ExprCollector<'db>,
    TypePtr,
    ThinVec<TypeBound>,
) -> TypeRefId;

pub(crate) struct GenericParamsCollector {
    type_or_consts: Arena<TypeOrConstParamData>,
    lifetimes: Arena<LifetimeParamData>,
    where_predicates: Vec<WherePredicate>,
    parent: GenericDefId,
}

impl GenericParamsCollector {
    pub(crate) fn new(parent: GenericDefId) -> Self {
        Self {
            type_or_consts: Default::default(),
            lifetimes: Default::default(),
            where_predicates: Default::default(),
            parent,
        }
    }
    pub(crate) fn with_self_param(
        ec: &mut ExprCollector<'_>,
        parent: GenericDefId,
        bounds: Option<ast::TypeBoundList>,
    ) -> Self {
        let mut this = Self::new(parent);
        this.fill_self_param(ec, bounds);
        this
    }

    pub(crate) fn lower(
        &mut self,
        ec: &mut ExprCollector<'_>,
        generic_param_list: Option<ast::GenericParamList>,
        where_clause: Option<ast::WhereClause>,
    ) {
        if let Some(params) = generic_param_list {
            self.lower_param_list(ec, params)
        }
        if let Some(where_clause) = where_clause {
            self.lower_where_predicates(ec, where_clause);
        }
    }

    pub(crate) fn collect_impl_trait<R>(
        &mut self,
        ec: &mut ExprCollector<'_>,
        cb: impl FnOnce(&mut ExprCollector<'_>, ImplTraitLowerFn<'_>) -> R,
    ) -> R {
        cb(
            ec,
            &mut Self::lower_argument_impl_trait(
                &mut self.type_or_consts,
                &mut self.where_predicates,
                self.parent,
            ),
        )
    }

    pub(crate) fn finish(self) -> Arc<GenericParams> {
        let Self { mut lifetimes, mut type_or_consts, mut where_predicates, parent: _ } = self;

        if lifetimes.is_empty() && type_or_consts.is_empty() && where_predicates.is_empty() {
            static EMPTY: LazyLock<Arc<GenericParams>> = LazyLock::new(|| {
                Arc::new(GenericParams {
                    lifetimes: Arena::new(),
                    type_or_consts: Arena::new(),
                    where_predicates: Box::default(),
                })
            });
            return Arc::clone(&EMPTY);
        }

        lifetimes.shrink_to_fit();
        type_or_consts.shrink_to_fit();
        where_predicates.shrink_to_fit();
        Arc::new(GenericParams {
            type_or_consts,
            lifetimes,
            where_predicates: where_predicates.into_boxed_slice(),
        })
    }

    fn lower_param_list(&mut self, ec: &mut ExprCollector<'_>, params: ast::GenericParamList) {
        for generic_param in params.generic_params() {
            let enabled = ec.check_cfg(&generic_param);
            if !enabled {
                continue;
            }

            match generic_param {
                ast::GenericParam::TypeParam(type_param) => {
                    let name = type_param.name().map_or_else(Name::missing, |it| it.as_name());
                    let default = type_param.default_type().map(|it| {
                        ec.lower_type_ref(it, &mut ExprCollector::impl_trait_error_allocator)
                    });
                    let param = TypeParamData {
                        name: Some(name.clone()),
                        default,
                        provenance: TypeParamProvenance::TypeParamList,
                    };
                    let idx = self.type_or_consts.alloc(param.into());
                    let type_ref =
                        TypeRef::TypeParam(TypeParamId::from_unchecked(TypeOrConstParamId {
                            parent: self.parent,
                            local_id: idx,
                        }));
                    let type_ref = ec.alloc_type_ref_desugared(type_ref);
                    self.lower_bounds(ec, type_param.type_bound_list(), Either::Left(type_ref));
                }
                ast::GenericParam::ConstParam(const_param) => {
                    let name = const_param.name().map_or_else(Name::missing, |it| it.as_name());
                    let ty = ec.lower_type_ref_opt(
                        const_param.ty(),
                        &mut ExprCollector::impl_trait_error_allocator,
                    );
                    let param = ConstParamData {
                        name,
                        ty,
                        default: const_param.default_val().map(|it| ec.lower_const_arg(it)),
                    };
                    let _idx = self.type_or_consts.alloc(param.into());
                }
                ast::GenericParam::LifetimeParam(lifetime_param) => {
                    let lifetime = ec.lower_lifetime_ref_opt(lifetime_param.lifetime());
                    if let LifetimeRef::Named(name) = &ec.store.lifetimes[lifetime] {
                        let param = LifetimeParamData { name: name.clone() };
                        let _idx = self.lifetimes.alloc(param);
                        self.lower_bounds(
                            ec,
                            lifetime_param.type_bound_list(),
                            Either::Right(lifetime),
                        );
                    }
                }
            }
        }
    }

    fn lower_where_predicates(
        &mut self,
        ec: &mut ExprCollector<'_>,
        where_clause: ast::WhereClause,
    ) {
        for pred in where_clause.predicates() {
            let target = if let Some(type_ref) = pred.ty() {
                Either::Left(
                    ec.lower_type_ref(type_ref, &mut ExprCollector::impl_trait_error_allocator),
                )
            } else if let Some(lifetime) = pred.lifetime() {
                Either::Right(ec.lower_lifetime_ref(lifetime))
            } else {
                continue;
            };

            let lifetimes: Option<Box<_>> =
                pred.for_binder().and_then(|it| it.generic_param_list()).map(|param_list| {
                    // Higher-Ranked Trait Bounds
                    param_list
                        .lifetime_params()
                        .map(|lifetime_param| {
                            lifetime_param
                                .lifetime()
                                .map_or_else(Name::missing, |lt| Name::new_lifetime(&lt.text()))
                        })
                        .collect()
                });
            for bound in pred.type_bound_list().iter().flat_map(|l| l.bounds()) {
                self.lower_type_bound_as_predicate(ec, bound, lifetimes.as_deref(), target);
            }
        }
    }

    fn lower_bounds(
        &mut self,
        ec: &mut ExprCollector<'_>,
        type_bounds: Option<ast::TypeBoundList>,
        target: Either<TypeRefId, LifetimeRefId>,
    ) {
        for bound in type_bounds.iter().flat_map(|type_bound_list| type_bound_list.bounds()) {
            self.lower_type_bound_as_predicate(ec, bound, None, target);
        }
    }

    fn lower_type_bound_as_predicate(
        &mut self,
        ec: &mut ExprCollector<'_>,
        bound: ast::TypeBound,
        hrtb_lifetimes: Option<&[Name]>,
        target: Either<TypeRefId, LifetimeRefId>,
    ) {
        let bound = ec.lower_type_bound(
            bound,
            &mut Self::lower_argument_impl_trait(
                &mut self.type_or_consts,
                &mut self.where_predicates,
                self.parent,
            ),
        );
        let predicate = match (target, bound) {
            (_, TypeBound::Error | TypeBound::Use(_)) => return,
            (Either::Left(type_ref), bound) => match hrtb_lifetimes {
                Some(hrtb_lifetimes) => WherePredicate::ForLifetime {
                    lifetimes: ThinVec::from_iter(hrtb_lifetimes.iter().cloned()),
                    target: type_ref,
                    bound,
                },
                None => WherePredicate::TypeBound { target: type_ref, bound },
            },
            (Either::Right(lifetime), TypeBound::Lifetime(bound)) => {
                WherePredicate::Lifetime { target: lifetime, bound }
            }
            (Either::Right(_), TypeBound::ForLifetime(..) | TypeBound::Path(..)) => return,
        };
        self.where_predicates.push(predicate);
    }

    fn lower_argument_impl_trait(
        type_or_consts: &mut Arena<TypeOrConstParamData>,
        where_predicates: &mut Vec<WherePredicate>,
        parent: GenericDefId,
    ) -> impl for<'ec, 'db> FnMut(&'ec mut ExprCollector<'db>, TypePtr, ThinVec<TypeBound>) -> TypeRefId
    {
        move |ec, ptr, impl_trait_bounds| {
            let param = TypeParamData {
                name: None,
                default: None,
                provenance: TypeParamProvenance::ArgumentImplTrait,
            };
            let param_id = TypeRef::TypeParam(TypeParamId::from_unchecked(TypeOrConstParamId {
                parent,
                local_id: type_or_consts.alloc(param.into()),
            }));
            let type_ref = ec.alloc_type_ref(param_id, ptr);
            for bound in impl_trait_bounds {
                where_predicates
                    .push(WherePredicate::TypeBound { target: type_ref, bound: bound.clone() });
            }
            type_ref
        }
    }

    fn fill_self_param(&mut self, ec: &mut ExprCollector<'_>, bounds: Option<ast::TypeBoundList>) {
        let self_ = Name::new_symbol_root(sym::Self_);
        let idx = self.type_or_consts.alloc(
            TypeParamData {
                name: Some(self_),
                default: None,
                provenance: TypeParamProvenance::TraitSelf,
            }
            .into(),
        );
        debug_assert_eq!(idx, GenericParams::SELF_PARAM_ID_IN_SELF);
        let type_ref = TypeRef::TypeParam(TypeParamId::from_unchecked(TypeOrConstParamId {
            parent: self.parent,
            local_id: idx,
        }));
        let self_ = ec.alloc_type_ref_desugared(type_ref);
        if let Some(bounds) = bounds {
            self.lower_bounds(ec, Some(bounds), Either::Left(self_));
        }
    }
}
