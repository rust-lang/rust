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
    expr_store::lower::ExprCollector,
    hir::generics::{
        ConstParamData, GenericParams, LifetimeParamData, TypeOrConstParamData, TypeParamData,
        TypeParamProvenance, WherePredicate, WherePredicateTypeTarget,
    },
    type_ref::{LifetimeRef, TypeBound, TypeRef, TypeRefId},
};

pub(crate) struct GenericParamsCollector<'db, 'c> {
    expr_collector: &'c mut ExprCollector<'db>,
    type_or_consts: Arena<TypeOrConstParamData>,
    lifetimes: Arena<LifetimeParamData>,
    where_predicates: Vec<WherePredicate>,
    parent: GenericDefId,
}

impl<'db, 'c> GenericParamsCollector<'db, 'c> {
    pub(crate) fn new(expr_collector: &'c mut ExprCollector<'db>, parent: GenericDefId) -> Self {
        Self {
            expr_collector,
            type_or_consts: Default::default(),
            lifetimes: Default::default(),
            where_predicates: Default::default(),
            parent,
        }
    }

    pub(crate) fn fill_self_param(&mut self, bounds: Option<ast::TypeBoundList>) {
        let self_ = Name::new_symbol_root(sym::Self_.clone());
        let idx = self.type_or_consts.alloc(
            TypeParamData {
                name: Some(self_.clone()),
                default: None,
                provenance: TypeParamProvenance::TraitSelf,
            }
            .into(),
        );
        let type_ref = TypeRef::TypeParam(TypeParamId::from_unchecked(TypeOrConstParamId {
            parent: self.parent,
            local_id: idx,
        }));
        let self_ = self.expr_collector.alloc_type_ref_desugared(type_ref);
        if let Some(bounds) = bounds {
            self.lower_bounds(Some(bounds), Either::Left(self_));
        }
    }

    pub(crate) fn lower(
        &mut self,
        generic_param_list: Option<ast::GenericParamList>,
        where_clause: Option<ast::WhereClause>,
    ) {
        if let Some(params) = generic_param_list {
            self.lower_param_list(params)
        }
        if let Some(where_clause) = where_clause {
            self.lower_where_predicates(where_clause);
        }
    }

    fn lower_param_list(&mut self, params: ast::GenericParamList) {
        for generic_param in params.generic_params() {
            let enabled = self.expr_collector.expander.is_cfg_enabled(
                self.expr_collector.db,
                self.expr_collector.module.krate(),
                &generic_param,
            );
            if !enabled {
                continue;
            }

            match generic_param {
                ast::GenericParam::TypeParam(type_param) => {
                    let name = type_param.name().map_or_else(Name::missing, |it| it.as_name());
                    let default = type_param
                        .default_type()
                        .map(|it| self.expr_collector.lower_type_ref(it, &mut |_| TypeRef::Error));
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
                    let type_ref = self.expr_collector.alloc_type_ref_desugared(type_ref);
                    self.lower_bounds(type_param.type_bound_list(), Either::Left(type_ref));
                }
                ast::GenericParam::ConstParam(const_param) => {
                    let name = const_param.name().map_or_else(Name::missing, |it| it.as_name());
                    let ty = self
                        .expr_collector
                        .lower_type_ref_opt(const_param.ty(), &mut |_| TypeRef::Error);
                    let param = ConstParamData {
                        name,
                        ty,
                        default: const_param
                            .default_val()
                            .map(|it| self.expr_collector.lower_const_arg(it)),
                    };
                    let _idx = self.type_or_consts.alloc(param.into());
                }
                ast::GenericParam::LifetimeParam(lifetime_param) => {
                    let lifetime_ref =
                        self.expr_collector.lower_lifetime_ref_opt(lifetime_param.lifetime());
                    if let LifetimeRef::Named(name) = &lifetime_ref {
                        let param = LifetimeParamData { name: name.clone() };
                        let _idx = self.lifetimes.alloc(param);
                        self.lower_bounds(
                            lifetime_param.type_bound_list(),
                            Either::Right(lifetime_ref),
                        );
                    }
                }
            }
        }
    }

    fn lower_where_predicates(&mut self, where_clause: ast::WhereClause) {
        for pred in where_clause.predicates() {
            let target = if let Some(type_ref) = pred.ty() {
                Either::Left(self.expr_collector.lower_type_ref(type_ref, &mut |_| TypeRef::Error))
            } else if let Some(lifetime) = pred.lifetime() {
                Either::Right(self.expr_collector.lower_lifetime_ref(lifetime))
            } else {
                continue;
            };

            let lifetimes: Option<Box<_>> = pred.generic_param_list().map(|param_list| {
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
                self.lower_type_bound_as_predicate(bound, lifetimes.as_deref(), target.clone());
            }
        }
    }

    fn lower_bounds(
        &mut self,
        type_bounds: Option<ast::TypeBoundList>,
        target: Either<TypeRefId, LifetimeRef>,
    ) {
        for bound in type_bounds.iter().flat_map(|type_bound_list| type_bound_list.bounds()) {
            self.lower_type_bound_as_predicate(bound, None, target.clone());
        }
    }

    fn lower_type_bound_as_predicate(
        &mut self,
        bound: ast::TypeBound,
        hrtb_lifetimes: Option<&[Name]>,
        target: Either<TypeRefId, LifetimeRef>,
    ) {
        let bound = self.expr_collector.lower_type_bound(
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
                    lifetimes: hrtb_lifetimes.to_vec().into_boxed_slice(),
                    target: WherePredicateTypeTarget::TypeRef(type_ref),
                    bound,
                },
                None => WherePredicate::TypeBound {
                    target: WherePredicateTypeTarget::TypeRef(type_ref),
                    bound,
                },
            },
            (Either::Right(lifetime), TypeBound::Lifetime(bound)) => {
                WherePredicate::Lifetime { target: lifetime, bound }
            }
            (Either::Right(_), TypeBound::ForLifetime(..) | TypeBound::Path(..)) => return,
        };
        self.where_predicates.push(predicate);
    }

    pub(crate) fn collect_impl_trait<R>(
        &mut self,
        cb: impl FnOnce(&mut ExprCollector<'_>, &mut dyn FnMut(ThinVec<TypeBound>) -> TypeRef) -> R,
    ) -> R {
        cb(
            self.expr_collector,
            &mut Self::lower_argument_impl_trait(
                &mut self.type_or_consts,
                &mut self.where_predicates,
                self.parent,
            ),
        )
    }

    fn lower_argument_impl_trait(
        type_or_consts: &mut Arena<TypeOrConstParamData>,
        where_predicates: &mut Vec<WherePredicate>,
        parent: GenericDefId,
    ) -> impl FnMut(ThinVec<TypeBound>) -> TypeRef {
        move |impl_trait_bounds| {
            let param = TypeParamData {
                name: None,
                default: None,
                provenance: TypeParamProvenance::ArgumentImplTrait,
            };
            let param_id = type_or_consts.alloc(param.into());
            for bound in impl_trait_bounds {
                where_predicates.push(WherePredicate::TypeBound {
                    target: WherePredicateTypeTarget::TypeOrConstParam(param_id),
                    bound: bound.clone(),
                });
            }
            TypeRef::TypeParam(TypeParamId::from_unchecked(TypeOrConstParamId {
                parent,
                local_id: param_id,
            }))
        }
    }

    pub(crate) fn finish(self) -> Arc<GenericParams> {
        let Self {
            mut lifetimes,
            mut type_or_consts,
            mut where_predicates,
            expr_collector: _,
            parent: _,
        } = self;

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
}
