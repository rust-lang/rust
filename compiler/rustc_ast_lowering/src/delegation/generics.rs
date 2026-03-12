use hir::HirId;
use hir::def::{DefKind, Res};
use rustc_ast::*;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::GenericParamDefKind;
use rustc_middle::{bug, ty};
use rustc_span::sym::{self};
use rustc_span::symbol::kw;
use rustc_span::{DUMMY_SP, Ident, Span};
use thin_vec::{ThinVec, thin_vec};

use crate::{AstOwner, LoweringContext, ResolverAstLoweringExt};

pub(super) enum DelegationGenerics<T> {
    /// User-specified args are present: `reuse foo::<String>;`.
    UserSpecified,
    /// The default case when no user-specified args are present: `reuse Trait::foo;`.
    Default(Option<T>),
    /// In free-to-trait reuse, when user specified args for trait `reuse Trait::<i32>::foo;`
    /// in this case we need to both generate `Self` and process user args.
    SelfAndUserSpecified(Option<T>),
}

/// Used for storing either AST generics or their lowered HIR version. Firstly we obtain
/// AST generics either from local function from AST index or from external function
/// through `tcx`. Next, at some point of generics processing we need to lower those
/// generics to HIR, for this purpose we use `into_hir_generics` that lowers AST generics
/// and replaces Ast variant with Hir. Such approach is useful as we can call this method
/// at any time knowing that lowering will occur at most only once. Then, in order to obtain generic
/// params or args we use `hir_generics_or_empty` or `into_generic_args` functions.
/// There also may be situations when we obtained AST generics but never lowered them to HIR,
/// meaning we did not propagate them and thus we do not need to generate generic params
/// (i.e., method call scenarios), in such a case this approach helps
/// a lot as if `into_hir_generics` will not be called then lowering will not happen.
pub(super) enum HirOrAstGenerics<'hir> {
    Ast(DelegationGenerics<Generics>),
    Hir(DelegationGenerics<&'hir hir::Generics<'hir>>),
}

pub(super) struct GenericsGenerationResult<'hir> {
    pub(super) generics: HirOrAstGenerics<'hir>,
    pub(super) args_segment_id: Option<HirId>,
}

pub(super) struct GenericsGenerationResults<'hir> {
    pub(super) parent: GenericsGenerationResult<'hir>,
    pub(super) child: GenericsGenerationResult<'hir>,
}

impl<T> DelegationGenerics<T> {
    fn is_user_specified(&self) -> bool {
        matches!(
            self,
            DelegationGenerics::UserSpecified | DelegationGenerics::SelfAndUserSpecified { .. }
        )
    }
}

impl<'hir> HirOrAstGenerics<'hir> {
    pub(super) fn into_hir_generics<R: ResolverAstLoweringExt<'hir>>(
        &mut self,
        ctx: &mut LoweringContext<'_, 'hir, R>,
        item_id: NodeId,
        span: Span,
    ) -> &mut HirOrAstGenerics<'hir> {
        if let HirOrAstGenerics::Ast(generics) = self {
            let process_params = |generics: &mut Generics| {
                ctx.lower_delegation_generic_params(item_id, span, &mut generics.params)
            };

            let hir_generics = match generics {
                DelegationGenerics::UserSpecified => DelegationGenerics::UserSpecified,
                DelegationGenerics::Default(generics) => {
                    DelegationGenerics::Default(generics.as_mut().map(process_params))
                }
                DelegationGenerics::SelfAndUserSpecified(generics) => {
                    DelegationGenerics::SelfAndUserSpecified(generics.as_mut().map(process_params))
                }
            };

            *self = HirOrAstGenerics::Hir(hir_generics);
        }

        self
    }

    fn hir_generics_or_empty(&self) -> &'hir hir::Generics<'hir> {
        match self {
            HirOrAstGenerics::Ast(_) => hir::Generics::empty(),
            HirOrAstGenerics::Hir(hir_generics) => match hir_generics {
                DelegationGenerics::UserSpecified => hir::Generics::empty(),
                DelegationGenerics::Default(generics)
                | DelegationGenerics::SelfAndUserSpecified(generics) => {
                    generics.unwrap_or(hir::Generics::empty())
                }
            },
        }
    }

    pub(super) fn into_generic_args<R: ResolverAstLoweringExt<'hir>>(
        &self,
        ctx: &mut LoweringContext<'_, 'hir, R>,
        add_lifetimes: bool,
        span: Span,
    ) -> Option<&'hir hir::GenericArgs<'hir>> {
        match self {
            HirOrAstGenerics::Ast(_) => {
                bug!("Attempting to get generic args before lowering to HIR")
            }
            HirOrAstGenerics::Hir(hir_generics) => match hir_generics {
                DelegationGenerics::UserSpecified => None,
                DelegationGenerics::Default(generics)
                | DelegationGenerics::SelfAndUserSpecified(generics) => generics.map(|generics| {
                    ctx.create_generics_args_from_params(generics.params, add_lifetimes, span)
                }),
            },
        }
    }

    pub(super) fn is_user_specified(&self) -> bool {
        match self {
            HirOrAstGenerics::Ast(ast_generics) => ast_generics.is_user_specified(),
            HirOrAstGenerics::Hir(hir_generics) => hir_generics.is_user_specified(),
        }
    }
}

impl<'a> GenericsGenerationResult<'a> {
    fn new(generics: DelegationGenerics<Generics>) -> GenericsGenerationResult<'a> {
        GenericsGenerationResult {
            generics: HirOrAstGenerics::Ast(generics),
            args_segment_id: None,
        }
    }
}

impl<'hir> GenericsGenerationResults<'hir> {
    pub(super) fn all_params<R: ResolverAstLoweringExt<'hir>>(
        &mut self,
        item_id: NodeId,
        span: Span,
        ctx: &mut LoweringContext<'_, 'hir, R>,
    ) -> impl Iterator<Item = hir::GenericParam<'hir>> {
        // Now we always call `into_hir_generics` both on child and parent,
        // however in future we would not do that, when scenarios like
        // method call will be supported (if HIR generics were not obtained
        // then it means that we did not propagated them, thus we do not need
        // to generate params).
        let parent = self
            .parent
            .generics
            .into_hir_generics(ctx, item_id, span)
            .hir_generics_or_empty()
            .params;

        let child = self
            .child
            .generics
            .into_hir_generics(ctx, item_id, span)
            .hir_generics_or_empty()
            .params;

        // Order generics, firstly we have parent and child lifetimes,
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

    /// As we add hack predicates(`'a: 'a`) for all lifetimes (see `lower_delegation_generic_params`
    /// and `generate_lifetime_predicate` functions) we need to add them to delegation generics.
    /// Those predicates will not affect resulting predicate inheritance and folding
    /// in `rustc_hir_analysis`, as we inherit all predicates from delegation signature.
    pub(super) fn all_predicates<R: ResolverAstLoweringExt<'hir>>(
        &mut self,
        item_id: NodeId,
        span: Span,
        ctx: &mut LoweringContext<'_, 'hir, R>,
    ) -> impl Iterator<Item = hir::WherePredicate<'hir>> {
        // Now we always call `into_hir_generics` both on child and parent,
        // however in future we would not do that, when scenarios like
        // method call will be supported (if HIR generics were not obtained
        // then it means that we did not propagated them, thus we do not need
        // to generate predicates).
        self.parent
            .generics
            .into_hir_generics(ctx, item_id, span)
            .hir_generics_or_empty()
            .predicates
            .into_iter()
            .chain(
                self.child
                    .generics
                    .into_hir_generics(ctx, item_id, span)
                    .hir_generics_or_empty()
                    .predicates
                    .into_iter(),
            )
            .copied()
    }
}

impl<'hir, R: ResolverAstLoweringExt<'hir>> LoweringContext<'_, 'hir, R> {
    pub(super) fn lower_delegation_generics(
        &mut self,
        delegation: &Delegation,
        root_fn_id: DefId,
        item_id: NodeId,
        span: Span,
    ) -> GenericsGenerationResults<'hir> {
        let delegation_in_free_ctx = !matches!(
            self.tcx.def_kind(self.tcx.local_parent(self.local_def_id(item_id))),
            DefKind::Trait | DefKind::Impl { .. }
        );

        let root_function_in_trait =
            matches!(self.tcx.def_kind(self.tcx.parent(root_fn_id)), DefKind::Trait);

        let generate_self = delegation_in_free_ctx && root_function_in_trait;

        let parent_generics_factory = |this: &mut Self, user_specified: bool| {
            this.get_parent_generics(
                this.tcx.parent(root_fn_id),
                generate_self,
                user_specified,
                span,
            )
        };

        let segments = &delegation.path.segments;
        let len = segments.len();

        let can_add_generics_to_parent = len >= 2
            && self.get_resolution_id(segments[len - 2].id).is_some_and(|def_id| {
                matches!(self.tcx.def_kind(def_id), DefKind::Trait | DefKind::TraitAlias)
            });

        let parent_generics = if can_add_generics_to_parent {
            if segments[len - 2].args.is_some() {
                if generate_self {
                    DelegationGenerics::SelfAndUserSpecified(parent_generics_factory(self, true))
                } else {
                    DelegationGenerics::UserSpecified
                }
            } else {
                DelegationGenerics::Default(parent_generics_factory(self, false))
            }
        } else {
            DelegationGenerics::Default(None)
        };

        let child_generics = if segments[len - 1].args.is_some() {
            DelegationGenerics::UserSpecified
        } else {
            DelegationGenerics::Default(self.get_fn_like_generics(root_fn_id, span))
        };

        GenericsGenerationResults {
            parent: GenericsGenerationResult::new(parent_generics),
            child: GenericsGenerationResult::new(child_generics),
        }
    }

    fn lower_delegation_generic_params(
        &mut self,
        item_id: NodeId,
        span: Span,
        params: &mut ThinVec<GenericParam>,
    ) -> &'hir hir::Generics<'hir> {
        for p in params.iter_mut() {
            // We want to create completely new params, so we generate
            // a new id, otherwise assertions will be triggered.
            p.id = self.next_node_id();

            // Remove default params, as they are not supported on functions
            // and there will duplicate DefId  when we try to lower them later.
            match &mut p.kind {
                GenericParamKind::Lifetime => {}
                GenericParamKind::Type { default } => *default = None,
                GenericParamKind::Const { default, .. } => *default = None,
            }

            // Note that we use self.disambiguator here, if we will create new every time
            // we will get ICE if params have the same name.
            let def_id = self
                .tcx
                .create_def(
                    self.resolver.def_id(item_id).unwrap(),
                    Some(p.ident.name),
                    match p.kind {
                        GenericParamKind::Lifetime => DefKind::LifetimeParam,
                        GenericParamKind::Type { .. } => DefKind::TyParam,
                        GenericParamKind::Const { .. } => DefKind::ConstParam,
                    },
                    None,
                    &mut self.disambiguator,
                )
                .def_id();

            self.resolver.insert_new_def_id(p.id, def_id);
        }

        // Fallback to default generic param lowering, we modified them in the loop above.
        let params = self.arena.alloc_from_iter(
            params.iter().map(|p| self.lower_generic_param(p, hir::GenericParamSource::Generics)),
        );

        // HACK: for now we generate predicates such that all lifetimes are early bound,
        // we can not not generate early-bound lifetimes, but we can't know which of them
        // are late-bound at this level of compilation.
        // FIXME(fn_delegation): proper support for late bound lifetimes.
        self.arena.alloc(hir::Generics {
            params,
            predicates: self.arena.alloc_from_iter(
                params
                    .iter()
                    .filter_map(|p| p.is_lifetime().then(|| self.generate_lifetime_predicate(p))),
            ),
            has_where_clause_predicates: false,
            where_clause_span: span,
            span,
        })
    }

    fn generate_lifetime_predicate(
        &mut self,
        p: &hir::GenericParam<'hir>,
    ) -> hir::WherePredicate<'hir> {
        let create_lifetime = |this: &mut Self| -> &'hir hir::Lifetime {
            this.arena.alloc(hir::Lifetime {
                hir_id: this.next_id(),
                ident: p.name.ident(),
                kind: rustc_hir::LifetimeKind::Param(p.def_id),
                source: rustc_hir::LifetimeSource::Path {
                    angle_brackets: rustc_hir::AngleBrackets::Full,
                },
                syntax: rustc_hir::LifetimeSyntax::ExplicitBound,
            })
        };

        hir::WherePredicate {
            hir_id: self.next_id(),
            span: DUMMY_SP,
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
                if p.name.ident().name == kw::SelfUpper {
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

    fn get_fn_like_generics(&mut self, id: DefId, span: Span) -> Option<Generics> {
        if let Some(local_id) = id.as_local() {
            match self.ast_index.get(local_id) {
                Some(AstOwner::Item(item)) if let ItemKind::Fn(f) = &item.kind => {
                    Some(f.generics.clone())
                }
                Some(AstOwner::AssocItem(item, _)) if let AssocItemKind::Fn(f) = &item.kind => {
                    Some(f.generics.clone())
                }
                _ => None,
            }
        } else {
            self.get_external_generics(id, false, span)
        }
    }

    fn get_external_generics(
        &mut self,
        id: DefId,
        processing_parent: bool,
        span: Span,
    ) -> Option<Generics> {
        let generics = self.tcx.generics_of(id);
        if generics.own_params.is_empty() {
            return None;
        }

        // Skip first Self parameter if we are in trait, it will be added later.
        let to_skip = (processing_parent && generics.has_self) as usize;

        Some(Generics {
            params: generics
                .own_params
                .iter()
                .skip(to_skip)
                .map(|p| GenericParam {
                    attrs: Default::default(),
                    bounds: Default::default(),
                    colon_span: None,
                    id: self.next_node_id(),
                    ident: Ident::with_dummy_span(p.name),
                    is_placeholder: false,
                    kind: match p.kind {
                        GenericParamDefKind::Lifetime => GenericParamKind::Lifetime,
                        GenericParamDefKind::Type { .. } => {
                            GenericParamKind::Type { default: None }
                        }
                        GenericParamDefKind::Const { .. } => self.map_const_kind(p, span),
                    },
                })
                .collect(),
            where_clause: Default::default(),
            span: DUMMY_SP,
        })
    }

    fn map_const_kind(&mut self, p: &ty::GenericParamDef, span: Span) -> GenericParamKind {
        let const_type = self.tcx.type_of(p.def_id).instantiate_identity();

        let (type_symbol, res) = match const_type.kind() {
            ty::Bool => (sym::bool, Res::PrimTy(hir::PrimTy::Bool)),
            ty::Uint(uint) => (uint.name(), Res::PrimTy(hir::PrimTy::Uint(*uint))),
            ty::Int(int) => (int.name(), Res::PrimTy(hir::PrimTy::Int(*int))),
            ty::Char => (sym::char, Res::PrimTy(hir::PrimTy::Char)),
            _ => {
                self.tcx
                    .dcx()
                    .span_delayed_bug(span, format!("Unexpected const type: {}", const_type));

                (sym::dummy, Res::Err)
            }
        };

        let node_id = self.next_node_id();

        self.resolver.insert_partial_res(node_id, hir::def::PartialRes::new(res));

        GenericParamKind::Const {
            ty: Box::new(Ty {
                id: node_id,
                kind: TyKind::Path(
                    None,
                    Path {
                        segments: thin_vec![PathSegment {
                            ident: Ident::with_dummy_span(type_symbol),
                            id: self.next_node_id(),
                            args: None
                        }],
                        span: DUMMY_SP,
                        tokens: None,
                    },
                ),
                span: DUMMY_SP,
                tokens: None,
            }),
            span: DUMMY_SP,
            default: None,
        }
    }

    fn get_parent_generics(
        &mut self,
        id: DefId,
        add_self: bool,
        user_specified: bool,
        span: Span,
    ) -> Option<Generics> {
        // If args are user-specified we still maybe need to add self.
        let mut generics = if user_specified {
            None
        } else {
            if let Some(local_id) = id.as_local() {
                if let Some(AstOwner::Item(item)) = self.ast_index.get(local_id)
                    && matches!(item.kind, ItemKind::Trait(..))
                {
                    item.opt_generics().cloned()
                } else {
                    None
                }
            } else {
                self.get_external_generics(id, true, span)
            }
        };

        if add_self {
            generics.get_or_insert_default().params.insert(
                0,
                GenericParam {
                    id: self.next_node_id(),
                    ident: Ident::new(kw::SelfUpper, DUMMY_SP),
                    attrs: Default::default(),
                    bounds: vec![],
                    is_placeholder: false,
                    kind: GenericParamKind::Type { default: None },
                    colon_span: None,
                },
            );
        }

        generics
    }
}
