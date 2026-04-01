use hir::HirId;
use hir::def::{DefKind, Res};
use rustc_ast::*;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::GenericParamDefKind;
use rustc_middle::{bug, ty};
use rustc_span::symbol::kw;
use rustc_span::{Ident, Span};

use crate::{LoweringContext, ResolverAstLoweringExt};

pub(super) enum DelegationGenerics<T> {
    /// User-specified args are present: `reuse foo::<String>;`.
    UserSpecified,
    /// The default case when no user-specified args are present: `reuse Trait::foo;`.
    Default(T),
    /// In free-to-trait reuse, when user specified args for trait `reuse Trait::<i32>::foo;`
    /// in this case we need to both generate `Self` and process user args.
    SelfAndUserSpecified(T),
    /// In delegations from trait impl to other entities like free functions or trait functions,
    /// we want to generate a function whose generics matches generics of signature function
    /// in trait.
    TraitImpl(T, bool /* Has user-specified args */),
}

/// Used for storing either ty generics or their uplifted HIR version. First we obtain
/// ty generics. Next, at some point of generics processing we need to uplift those
/// generics to HIR, for this purpose we use `into_hir_generics` that uplifts ty generics
/// and replaces Ty variant with Hir. Such approach is useful as we can call this method
/// at any time knowing that uplifting will occur at most only once. Then, in order to obtain generic
/// params or args we use `hir_generics_or_empty` or `into_generic_args` functions.
/// There also may be situations when we obtained ty generics but never uplifted them to HIR,
/// meaning we did not propagate them and thus we do not need to generate generic params
/// (i.e., method call scenarios), in such a case this approach helps
/// a lot as if `into_hir_generics` will not be called then uplifting will not happen.
pub(super) enum HirOrTyGenerics<'hir> {
    Ty(DelegationGenerics<&'hir [ty::GenericParamDef]>),
    Hir(DelegationGenerics<&'hir hir::Generics<'hir>>),
}

pub(super) struct GenericsGenerationResult<'hir> {
    pub(super) generics: HirOrTyGenerics<'hir>,
    pub(super) args_segment_id: Option<HirId>,
}

pub(super) struct GenericsGenerationResults<'hir> {
    pub(super) parent: GenericsGenerationResult<'hir>,
    pub(super) child: GenericsGenerationResult<'hir>,
}

pub(super) struct GenericArgsPropagationDetails {
    pub(super) should_propagate: bool,
    pub(super) use_args_in_sig_inheritance: bool,
}

impl<T> DelegationGenerics<T> {
    fn args_propagation_details(&self) -> GenericArgsPropagationDetails {
        match self {
            DelegationGenerics::UserSpecified | DelegationGenerics::SelfAndUserSpecified { .. } => {
                GenericArgsPropagationDetails {
                    should_propagate: false,
                    use_args_in_sig_inheritance: true,
                }
            }
            DelegationGenerics::TraitImpl(_, user_specified) => GenericArgsPropagationDetails {
                should_propagate: !*user_specified,
                use_args_in_sig_inheritance: false,
            },
            DelegationGenerics::Default(_) => GenericArgsPropagationDetails {
                should_propagate: true,
                use_args_in_sig_inheritance: false,
            },
        }
    }
}

impl<'hir> HirOrTyGenerics<'hir> {
    pub(super) fn into_hir_generics(
        &mut self,
        ctx: &mut LoweringContext<'_, 'hir, impl ResolverAstLoweringExt<'hir>>,
        span: Span,
    ) -> &mut HirOrTyGenerics<'hir> {
        if let HirOrTyGenerics::Ty(params) = self {
            let mut uplift_params = |generics: &'hir [ty::GenericParamDef]| {
                ctx.uplift_delegation_generic_params(span, generics)
            };

            let hir_generics = match params {
                DelegationGenerics::UserSpecified => DelegationGenerics::UserSpecified,
                DelegationGenerics::Default(params) => {
                    DelegationGenerics::Default(uplift_params(params))
                }
                DelegationGenerics::SelfAndUserSpecified(params) => {
                    DelegationGenerics::SelfAndUserSpecified(uplift_params(params))
                }
                DelegationGenerics::TraitImpl(params, user_specified) => {
                    DelegationGenerics::TraitImpl(uplift_params(params), *user_specified)
                }
            };

            *self = HirOrTyGenerics::Hir(hir_generics);
        }

        self
    }

    fn hir_generics_or_empty(&self) -> &'hir hir::Generics<'hir> {
        match self {
            HirOrTyGenerics::Ty(_) => hir::Generics::empty(),
            HirOrTyGenerics::Hir(hir_generics) => match hir_generics {
                DelegationGenerics::UserSpecified => hir::Generics::empty(),
                DelegationGenerics::Default(generics)
                | DelegationGenerics::SelfAndUserSpecified(generics)
                | DelegationGenerics::TraitImpl(generics, _) => generics,
            },
        }
    }

    pub(super) fn into_generic_args(
        &self,
        ctx: &mut LoweringContext<'_, 'hir, impl ResolverAstLoweringExt<'hir>>,
        add_lifetimes: bool,
        span: Span,
    ) -> &'hir hir::GenericArgs<'hir> {
        match self {
            HirOrTyGenerics::Ty(_) => {
                bug!("Attempting to get generic args before uplifting to HIR")
            }
            HirOrTyGenerics::Hir(hir_generics) => match hir_generics {
                DelegationGenerics::UserSpecified => hir::GenericArgs::NONE,
                DelegationGenerics::Default(generics)
                | DelegationGenerics::SelfAndUserSpecified(generics)
                | DelegationGenerics::TraitImpl(generics, _) => {
                    ctx.create_generics_args_from_params(generics.params, add_lifetimes, span)
                }
            },
        }
    }

    pub(super) fn args_propagation_details(&self) -> GenericArgsPropagationDetails {
        match self {
            HirOrTyGenerics::Ty(ty_generics) => ty_generics.args_propagation_details(),
            HirOrTyGenerics::Hir(hir_generics) => hir_generics.args_propagation_details(),
        }
    }
}

impl<'hir> GenericsGenerationResult<'hir> {
    fn new(
        generics: DelegationGenerics<&'hir [ty::GenericParamDef]>,
    ) -> GenericsGenerationResult<'hir> {
        GenericsGenerationResult { generics: HirOrTyGenerics::Ty(generics), args_segment_id: None }
    }
}

impl<'hir> GenericsGenerationResults<'hir> {
    pub(super) fn all_params(
        &mut self,
        span: Span,
        ctx: &mut LoweringContext<'_, 'hir, impl ResolverAstLoweringExt<'hir>>,
    ) -> impl Iterator<Item = hir::GenericParam<'hir>> {
        // Now we always call `into_hir_generics` both on child and parent,
        // however in future we would not do that, when scenarios like
        // method call will be supported (if HIR generics were not obtained
        // then it means that we did not propagated them, thus we do not need
        // to generate params).
        let mut create_params = |result: &mut GenericsGenerationResult<'hir>| {
            result.generics.into_hir_generics(ctx, span).hir_generics_or_empty().params
        };

        let parent = create_params(&mut self.parent);
        let child = create_params(&mut self.child);

        // Order generics, first we have parent and child lifetimes,
        // then parent and child types and consts.
        // `generics_of` in `rustc_hir_analysis` will order them anyway,
        // however we want the order to be consistent in HIR too.
        parent
            .iter()
            .filter(|p| p.is_lifetime())
            .chain(child.iter().filter(|p| p.is_lifetime()))
            .chain(parent.iter().filter(|p| !p.is_lifetime()))
            .chain(child.iter().filter(|p| !p.is_lifetime()))
            .copied()
    }

    /// As we add hack predicates(`'a: 'a`) for all lifetimes (see `uplift_delegation_generic_params`
    /// and `generate_lifetime_predicate` functions) we need to add them to delegation generics.
    /// Those predicates will not affect resulting predicate inheritance and folding
    /// in `rustc_hir_analysis`, as we inherit all predicates from delegation signature.
    pub(super) fn all_predicates(
        &mut self,
        span: Span,
        ctx: &mut LoweringContext<'_, 'hir, impl ResolverAstLoweringExt<'hir>>,
    ) -> impl Iterator<Item = hir::WherePredicate<'hir>> {
        // Now we always call `into_hir_generics` both on child and parent,
        // however in future we would not do that, when scenarios like
        // method call will be supported (if HIR generics were not obtained
        // then it means that we did not propagated them, thus we do not need
        // to generate predicates).
        let mut create_predicates = |result: &mut GenericsGenerationResult<'hir>| {
            result.generics.into_hir_generics(ctx, span).hir_generics_or_empty().predicates
        };

        let parent = create_predicates(&mut self.parent);
        let child = create_predicates(&mut self.child);

        parent.into_iter().chain(child).copied()
    }
}

impl<'hir, R: ResolverAstLoweringExt<'hir>> LoweringContext<'_, 'hir, R> {
    pub(super) fn uplift_delegation_generics(
        &mut self,
        delegation: &Delegation,
        sig_id: DefId,
        item_id: NodeId,
    ) -> GenericsGenerationResults<'hir> {
        let delegation_parent_kind =
            self.tcx.def_kind(self.tcx.local_parent(self.local_def_id(item_id)));

        let segments = &delegation.path.segments;
        let len = segments.len();
        let child_user_specified = segments[len - 1].args.is_some();

        let sig_params = &self.tcx.generics_of(sig_id).own_params[..];

        // If we are in trait impl always generate function whose generics matches
        // those that are defined in trait.
        if matches!(delegation_parent_kind, DefKind::Impl { of_trait: true }) {
            // Considering parent generics, during signature inheritance
            // we will take those args that are in trait impl header trait ref.
            let parent = GenericsGenerationResult::new(DelegationGenerics::TraitImpl(&[], true));

            let child = DelegationGenerics::TraitImpl(sig_params, child_user_specified);
            let child = GenericsGenerationResult::new(child);

            return GenericsGenerationResults { parent, child };
        }

        let delegation_in_free_ctx =
            !matches!(delegation_parent_kind, DefKind::Trait | DefKind::Impl { .. });

        let sig_parent = self.tcx.parent(sig_id);
        let sig_in_trait = matches!(self.tcx.def_kind(sig_parent), DefKind::Trait);

        let can_add_generics_to_parent = len >= 2
            && self.get_resolution_id(segments[len - 2].id).is_some_and(|def_id| {
                matches!(self.tcx.def_kind(def_id), DefKind::Trait | DefKind::TraitAlias)
            });

        let generate_self = delegation_in_free_ctx && sig_in_trait;
        let parent_generics = if can_add_generics_to_parent {
            let sig_parent_params = &self.tcx.generics_of(sig_parent).own_params[..];

            if segments[len - 2].args.is_some() {
                if generate_self {
                    // Take only first Self parameter, it is trait so Self must be present.
                    DelegationGenerics::SelfAndUserSpecified(&sig_parent_params[..1])
                } else {
                    DelegationGenerics::UserSpecified
                }
            } else {
                let skip_self = usize::from(!generate_self);
                DelegationGenerics::Default(&sig_parent_params[skip_self..])
            }
        } else {
            DelegationGenerics::<&'hir [ty::GenericParamDef]>::Default(&[])
        };

        let child_generics = if child_user_specified {
            DelegationGenerics::UserSpecified
        } else {
            DelegationGenerics::Default(sig_params)
        };

        GenericsGenerationResults {
            parent: GenericsGenerationResult::new(parent_generics),
            child: GenericsGenerationResult::new(child_generics),
        }
    }

    fn uplift_delegation_generic_params(
        &mut self,
        span: Span,
        params: &'hir [ty::GenericParamDef],
    ) -> &'hir hir::Generics<'hir> {
        let params = self.arena.alloc_from_iter(params.iter().map(|p| {
            let def_kind = match p.kind {
                GenericParamDefKind::Lifetime => DefKind::LifetimeParam,
                GenericParamDefKind::Type { .. } => DefKind::TyParam,
                GenericParamDefKind::Const { .. } => DefKind::ConstParam,
            };

            let param_ident = Ident::new(p.name, span);
            let def_name = Some(param_ident.name);
            let path_data = def_kind.def_path_data(def_name);
            let node_id = self.next_node_id();

            let def_id = self.create_def(node_id, def_name, def_kind, path_data, span);

            let kind = match p.kind {
                GenericParamDefKind::Lifetime => {
                    hir::GenericParamKind::Lifetime { kind: hir::LifetimeParamKind::Explicit }
                }
                GenericParamDefKind::Type { synthetic, .. } => {
                    hir::GenericParamKind::Type { default: None, synthetic }
                }
                GenericParamDefKind::Const { .. } => {
                    let hir_id = self.next_id();
                    let kind = hir::TyKind::InferDelegation(hir::InferDelegation::DefId(p.def_id));

                    hir::GenericParamKind::Const {
                        ty: self.arena.alloc(hir::Ty { kind, hir_id, span }),
                        default: None,
                    }
                }
            };

            // Important: we don't use `self.next_id()` as we want to execute
            // `lower_node_id` routine so param's id is added to `self.children`.
            let hir_id = self.lower_node_id(node_id);

            hir::GenericParam {
                hir_id,
                colon_span: Some(span),
                def_id,
                kind,
                name: hir::ParamName::Plain(param_ident),
                pure_wrt_drop: p.pure_wrt_drop,
                source: hir::GenericParamSource::Generics,
                span,
            }
        }));

        // HACK: for now we generate predicates such that all lifetimes are early bound,
        // we can not not generate early-bound lifetimes, but we can't know which of them
        // are late-bound at this level of compilation.
        let predicates =
            self.arena.alloc_from_iter(params.iter().filter_map(|p| {
                p.is_lifetime().then(|| self.generate_lifetime_predicate(p, span))
            }));

        self.arena.alloc(hir::Generics {
            params,
            predicates,
            has_where_clause_predicates: false,
            where_clause_span: span,
            span,
        })
    }

    fn generate_lifetime_predicate(
        &mut self,
        p: &hir::GenericParam<'hir>,
        span: Span,
    ) -> hir::WherePredicate<'hir> {
        let create_lifetime = |this: &mut Self| -> &'hir hir::Lifetime {
            this.arena.alloc(hir::Lifetime {
                hir_id: this.next_id(),
                ident: p.name.ident(),
                kind: hir::LifetimeKind::Param(p.def_id),
                source: hir::LifetimeSource::Path { angle_brackets: hir::AngleBrackets::Full },
                syntax: hir::LifetimeSyntax::ExplicitBound,
            })
        };

        hir::WherePredicate {
            hir_id: self.next_id(),
            span,
            kind: self.arena.alloc(hir::WherePredicateKind::RegionPredicate(
                hir::WhereRegionPredicate {
                    in_where_clause: true,
                    lifetime: create_lifetime(self),
                    bounds: self
                        .arena
                        .alloc_slice(&[hir::GenericBound::Outlives(create_lifetime(self))]),
                },
            )),
        }
    }

    fn create_generics_args_from_params(
        &mut self,
        params: &[hir::GenericParam<'hir>],
        add_lifetimes: bool,
        span: Span,
    ) -> &'hir hir::GenericArgs<'hir> {
        self.arena.alloc(hir::GenericArgs {
            args: self.arena.alloc_from_iter(params.iter().filter_map(|p| {
                // Skip self generic arg, we do not need to propagate it.
                if p.name.ident().name == kw::SelfUpper || p.is_impl_trait() {
                    return None;
                }

                let create_path = |this: &mut Self| {
                    let res = Res::Def(
                        match p.kind {
                            hir::GenericParamKind::Lifetime { .. } => DefKind::LifetimeParam,
                            hir::GenericParamKind::Type { .. } => DefKind::TyParam,
                            hir::GenericParamKind::Const { .. } => DefKind::ConstParam,
                        },
                        p.def_id.to_def_id(),
                    );

                    hir::QPath::Resolved(
                        None,
                        self.arena.alloc(hir::Path {
                            segments: this.arena.alloc_slice(&[hir::PathSegment {
                                args: None,
                                hir_id: this.next_id(),
                                ident: p.name.ident(),
                                infer_args: false,
                                res,
                            }]),
                            res,
                            span: p.span,
                        }),
                    )
                };

                match p.kind {
                    hir::GenericParamKind::Lifetime { .. } => match add_lifetimes {
                        true => Some(hir::GenericArg::Lifetime(self.arena.alloc(hir::Lifetime {
                            hir_id: self.next_id(),
                            ident: p.name.ident(),
                            kind: hir::LifetimeKind::Param(p.def_id),
                            source: hir::LifetimeSource::Path {
                                angle_brackets: hir::AngleBrackets::Full,
                            },
                            syntax: hir::LifetimeSyntax::ExplicitBound,
                        }))),
                        false => None,
                    },
                    hir::GenericParamKind::Type { .. } => {
                        Some(hir::GenericArg::Type(self.arena.alloc(hir::Ty {
                            hir_id: self.next_id(),
                            span: p.span,
                            kind: hir::TyKind::Path(create_path(self)),
                        })))
                    }
                    hir::GenericParamKind::Const { .. } => {
                        Some(hir::GenericArg::Const(self.arena.alloc(hir::ConstArg {
                            hir_id: self.next_id(),
                            kind: hir::ConstArgKind::Path(create_path(self)),
                            span: p.span,
                        })))
                    }
                }
            })),
            constraints: &[],
            parenthesized: hir::GenericArgsParentheses::No,
            span_ext: span,
        })
    }
}
