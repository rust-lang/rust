use hir::HirId;
use hir::def::{DefKind, Res};
use rustc_ast::*;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{GenericParamDefKind, TyCtxt};
use rustc_middle::{bug, ty};
use rustc_span::symbol::kw;
use rustc_span::{ErrorGuaranteed, Ident, Span, sym};

use crate::LoweringContext;
use crate::delegation::resolution::resolver::DelegationResolver;
use crate::diagnostics::DelegationInfersMismatch;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(super) enum GenericsPosition {
    Parent,
    Child,
}

#[derive(Debug)]
pub(super) enum GenericArgSlot<T> {
    UserSpecified,
    Generate(T, Option<usize> /* Infer arg index from AST */),
}

pub(super) struct DelegationGenerics<T> {
    data: T,
    pos: GenericsPosition,
    trait_impl: bool,
}

type TyGenerics<'hir> = Vec<GenericArgSlot<&'hir ty::GenericParamDef>>;

impl<'hir> DelegationGenerics<TyGenerics<'hir>> {
    fn generate_all(
        params: &'hir [ty::GenericParamDef],
        pos: GenericsPosition,
        trait_impl: bool,
    ) -> Self {
        DelegationGenerics {
            data: params.iter().map(|p| GenericArgSlot::Generate(p, None)).collect(),
            pos,
            trait_impl,
        }
    }
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
    Ty(DelegationGenerics<TyGenerics<'hir>>),
    Hir(DelegationGenerics<&'hir hir::Generics<'hir>>),
}

pub(super) struct GenericsGenerationResult<'hir> {
    pub(super) generics: HirOrTyGenerics<'hir>,
    pub(super) args_segment_id: HirId,
    pub(super) use_for_sig_inheritance: bool,
}

impl GenericsGenerationResult<'_> {
    pub(super) fn segment_id_for_sig(&self) -> Option<HirId> {
        self.use_for_sig_inheritance.then(|| self.args_segment_id)
    }
}

pub(super) struct GenericsGenerationResults<'hir> {
    pub(super) parent: GenericsGenerationResult<'hir>,
    pub(super) child: GenericsGenerationResult<'hir>,
    pub(super) self_ty_propagation_kind: Option<hir::DelegationSelfTyPropagationKind>,
}

pub(super) struct DelegationGenericArgsIterator<'hir> {
    index: usize = Default::default(),
    params: &'hir [hir::GenericParam<'hir>],
}

/// During generic args propagation we need to create generic args
/// (and their `HirId`s) on demand, as some of generic args can not be used
/// and in this case an assert of an unseen `HirId` will be triggered. Moreover,
/// when replacing infers with generated generic params we should reuse existing
/// `HirId` of replaced infer, thus this iterator abstracts the way `HirId`s are
/// created for new generic args.
impl<'hir> DelegationGenericArgsIterator<'hir> {
    pub(super) fn next(
        &mut self,
        ctx: &mut LoweringContext<'_, 'hir>,
        hir_id_factory: impl FnOnce(&mut LoweringContext<'_, 'hir>) -> HirId,
    ) -> Option<hir::GenericArg<'hir>> {
        let p = loop {
            if self.index >= self.params.len() {
                return None;
            }

            let p = self.params[self.index];
            self.index += 1;

            // Skip self generic arg, we do not need to propagate it.
            if p.name.ident().name == kw::SelfUpper || p.is_impl_trait() {
                continue;
            }

            break p;
        };

        let hir_id = hir_id_factory(ctx);

        Some(match p.kind {
            hir::GenericParamKind::Lifetime { .. } => {
                hir::GenericArg::Lifetime(ctx.arena.alloc(hir::Lifetime {
                    hir_id,
                    ident: p.name.ident(),
                    kind: hir::LifetimeKind::Param(p.def_id),
                    source: hir::LifetimeSource::Path { angle_brackets: hir::AngleBrackets::Full },
                    syntax: hir::LifetimeSyntax::ExplicitBound,
                }))
            }
            hir::GenericParamKind::Type { .. } => hir::GenericArg::Type(ctx.arena.alloc(hir::Ty {
                hir_id,
                span: p.span,
                kind: hir::TyKind::Path(ctx.create_generic_arg_path(&p)),
            })),
            hir::GenericParamKind::Const { .. } => {
                hir::GenericArg::Const(ctx.arena.alloc(hir::ConstArg {
                    hir_id,
                    kind: hir::ConstArgKind::Path(ctx.create_generic_arg_path(&p)),
                    span: p.span,
                }))
            }
        })
    }

    pub(super) fn consume_all(
        mut self,
        ctx: &mut LoweringContext<'_, 'hir>,
    ) -> Vec<hir::GenericArg<'hir>> {
        let mut args = vec![];
        while let Some(arg) = self.next(ctx, |ctx| ctx.next_id()) {
            args.push(arg);
        }

        args
    }
}

impl<'hir> HirOrTyGenerics<'hir> {
    pub(super) fn into_hir_generics(&mut self, ctx: &mut LoweringContext<'_, 'hir>, span: Span) {
        if let HirOrTyGenerics::Ty(ty) = self {
            let rename_self = ty.pos == GenericsPosition::Child;
            let params = ctx.uplift_delegation_generic_params(span, &ty.data, rename_self);

            *self = HirOrTyGenerics::Hir(DelegationGenerics {
                data: params,
                pos: ty.pos,
                trait_impl: ty.trait_impl,
            });
        }
    }

    fn hir_generics_or_empty(&self) -> &'hir hir::Generics<'hir> {
        match self {
            HirOrTyGenerics::Ty(_) => hir::Generics::empty(),
            HirOrTyGenerics::Hir(hir) => hir.data,
        }
    }

    pub(super) fn create_args_iterator(&self) -> DelegationGenericArgsIterator<'hir> {
        match self {
            HirOrTyGenerics::Ty(_) => {
                bug!("attempting to get generic args before uplifting to HIR")
            }
            HirOrTyGenerics::Hir(hir) => {
                DelegationGenericArgsIterator { params: hir.data.params, .. }
            }
        }
    }

    pub(super) fn infer_indices(&self) -> FxHashSet<usize> {
        match self {
            HirOrTyGenerics::Ty(ty) => ty
                .data
                .iter()
                .flat_map(|slot| match slot {
                    GenericArgSlot::Generate(_, Some(idx)) => Some(*idx),
                    _ => None,
                })
                .collect(),
            HirOrTyGenerics::Hir(_) => bug!("accessed infer indices on uplifted generics"),
        }
    }

    pub(super) fn is_trait_impl(&self) -> bool {
        match self {
            HirOrTyGenerics::Ty(ty) => ty.trait_impl,
            HirOrTyGenerics::Hir(hir) => hir.trait_impl,
        }
    }

    pub(super) fn find_self_param(&self) -> &'hir hir::GenericParam<'hir> {
        match self {
            HirOrTyGenerics::Ty(_) => {
                bug!("accessed ty-level generics while searching for uplifted `Self` param")
            }
            HirOrTyGenerics::Hir(hir) => hir
                .data
                .params
                .iter()
                .find(|p| p.name.ident().name == kw::SelfUpper)
                .expect("`Self` generic param is not found while expected"),
        }
    }

    pub(crate) fn pos(&self) -> GenericsPosition {
        match self {
            HirOrTyGenerics::Ty(ty) => ty.pos,
            HirOrTyGenerics::Hir(hir) => hir.pos,
        }
    }
}

impl<'hir> GenericsGenerationResult<'hir> {
    fn new(generics: DelegationGenerics<TyGenerics<'hir>>) -> GenericsGenerationResult<'hir> {
        GenericsGenerationResult {
            generics: HirOrTyGenerics::Ty(generics),
            args_segment_id: HirId::INVALID,
            use_for_sig_inheritance: false,
        }
    }
}

enum ParentSegmentArgs<'a> {
    /// Parent segment is valid and generic args are specified:
    /// `reuse Trait::<'static, ()>::foo;`.
    Specified(&'a AngleBracketedArgs),
    /// Parent segment is valid and args are not specified:
    /// `reuse Trait::foo;`.
    NotSpecified,
    /// Parent segment does not exist (`reuse foo`) or we can not
    /// add generics to it:
    /// ```rust
    /// mod to_reuse {
    ///     fn foo() {}
    /// }
    ///
    /// // Can't add generic args to module.
    /// reuse to_reuse::foo;
    /// ```
    Invalid,
}

struct GenericsResolution<'a, 'tcx> {
    trait_impl: bool,

    parent_args: ParentSegmentArgs<'a>,
    child_args: Option<&'a AngleBracketedArgs>,

    sig_parent_params: &'tcx [ty::GenericParamDef],
    sig_child_params: &'tcx [ty::GenericParamDef],

    free_to_trait_delegation: bool,
    /// `reuse Trait::foo;`.
    qself_is_none: bool,
    /// `reuse <_ as Trait>::foo;`.
    qself_is_infer: bool,
    /// Whether we should generate `Self` generic param.
    generate_self: bool,
}

impl<'hir> DelegationResolver<'_, 'hir> {
    fn resolve_generics<'a>(
        &self,
        delegation: &'a Delegation,
        sig_id: DefId,
    ) -> Result<GenericsResolution<'a, 'hir>, ErrorGuaranteed> {
        let tcx = self.tcx();
        let delegation_parent_kind = tcx.def_kind(tcx.local_parent(self.owner_id()));

        let delegation_in_free_ctx =
            !matches!(delegation_parent_kind, DefKind::Trait | DefKind::Impl { .. });

        let sig_parent = tcx.parent(sig_id);
        let sig_in_trait = matches!(tcx.def_kind(sig_parent), DefKind::Trait);
        let free_to_trait_delegation = delegation_in_free_ctx && sig_in_trait;

        let mut sig_parent_params: &[ty::GenericParamDef] = &[];

        let qself_is_infer =
            delegation.qself.as_ref().is_some_and(|qself| qself.ty.is_maybe_parenthesised_infer());

        let qself_is_none = delegation.qself.is_none();

        let parent_args = if let [.., parent_segment, _] = &delegation.path.segments[..] {
            let res = self.get_resolution_id(parent_segment.id)?;
            if matches!(tcx.def_kind(res), DefKind::Trait | DefKind::TraitAlias) {
                sig_parent_params = &tcx.generics_of(sig_parent).own_params;
                self.get_user_args(parent_segment)
                    .map(|args| ParentSegmentArgs::Specified(args))
                    .unwrap_or(ParentSegmentArgs::NotSpecified)
            } else {
                ParentSegmentArgs::Invalid
            }
        } else {
            ParentSegmentArgs::Invalid
        };

        Ok(GenericsResolution {
            parent_args,
            sig_parent_params,
            qself_is_none,
            qself_is_infer,
            free_to_trait_delegation,
            generate_self: free_to_trait_delegation && (qself_is_none || qself_is_infer),
            trait_impl: matches!(delegation_parent_kind, DefKind::Impl { of_trait: true }),
            sig_child_params: &tcx.generics_of(sig_id).own_params,
            child_args: self.get_user_args(
                delegation.path.segments.last().expect("must be at least one segment"),
            ),
        })
    }

    fn get_user_args<'a>(&self, segment: &'a PathSegment) -> Option<&'a AngleBracketedArgs> {
        let Some(args) = &segment.args else { return None };
        let GenericArgs::AngleBracketed(args) = args else {
            self.tcx().dcx().span_delayed_bug(
                segment.span(),
                "expected angle-bracketed generic args in delegation segment",
            );

            return None;
        };

        // Treat empty args `reuse foo::<> as bar` as `reuse foo as bar`,
        // the same logic applied when we call function `fn f<T>(t: T)`
        // like that `f::<>(())`, in HIR no `<>` will be generated.
        (!args.args.is_empty()).then(|| args)
    }

    pub(super) fn resolve_and_generate_generics(
        &self,
        delegation: &Delegation,
        sig_id: DefId,
    ) -> Result<GenericsGenerationResults<'hir>, ErrorGuaranteed> {
        let res @ GenericsResolution {
            trait_impl,
            generate_self,
            sig_child_params,
            sig_parent_params,
            ..
        } = self.resolve_generics(delegation, sig_id)?;

        // If we are in trait impl always generate function whose generics matches
        // those that are defined in trait.
        if trait_impl {
            // Considering parent generics, during signature inheritance
            // we will take those args that are in trait impl header trait ref.
            let parent =
                DelegationGenerics { data: vec![], pos: GenericsPosition::Child, trait_impl: true };

            let parent = GenericsGenerationResult::new(parent);

            let child =
                DelegationGenerics::generate_all(sig_child_params, GenericsPosition::Child, true);

            let child = GenericsGenerationResult::new(child);

            return Ok(GenericsGenerationResults { parent, child, self_ty_propagation_kind: None });
        }

        let tcx = self.tcx();
        let parent_generics = match res.parent_args {
            ParentSegmentArgs::Specified(args) => DelegationGenerics {
                data: Self::create_slots_from_args(
                    tcx,
                    args,
                    &sig_parent_params[usize::from(!generate_self)..],
                    generate_self,
                ),
                pos: GenericsPosition::Parent,
                trait_impl,
            },
            ParentSegmentArgs::NotSpecified => DelegationGenerics::generate_all(
                &sig_parent_params[usize::from(!generate_self)..],
                GenericsPosition::Parent,
                trait_impl,
            ),
            ParentSegmentArgs::Invalid => {
                DelegationGenerics { data: vec![], pos: GenericsPosition::Parent, trait_impl }
            }
        };

        let child_generics = if let Some(args) = res.child_args {
            let synth_params_index = sig_child_params
                .iter()
                .position(|p| p.kind.is_synthetic())
                .unwrap_or(sig_child_params.len());

            let mut slots = Self::create_slots_from_args(
                tcx,
                args,
                &sig_child_params[..synth_params_index],
                trait_impl,
            );

            for synth_param in &sig_child_params[synth_params_index..] {
                slots.push(GenericArgSlot::Generate(synth_param, None));
            }

            DelegationGenerics { data: slots, pos: GenericsPosition::Child, trait_impl }
        } else {
            DelegationGenerics::generate_all(sig_child_params, GenericsPosition::Child, trait_impl)
        };

        Ok(GenericsGenerationResults {
            parent: GenericsGenerationResult::new(parent_generics),
            child: GenericsGenerationResult::new(child_generics),
            self_ty_propagation_kind: match res.free_to_trait_delegation {
                true => Some(match res.qself_is_none {
                    true => hir::DelegationSelfTyPropagationKind::SelfParam,
                    false => match res.qself_is_infer {
                        true => hir::DelegationSelfTyPropagationKind::SelfParam,
                        // HirId is filled during generic args propagation.
                        false => hir::DelegationSelfTyPropagationKind::SelfTy(HirId::INVALID),
                    },
                }),
                false => None,
            },
        })
    }

    /// Generates generic argument slots for user-specified `args` and
    /// generic `params` of the signature function. This function checks whether
    /// there are infers (`kw::UnderscoreLifetime` or `kw::Underscore`) in
    /// user-specified args, and if so we add `Generate` slot meaning we have to
    /// generate generic param for delegation and propagate it instead of this infer.
    /// We zip over user-specified args and signature generic params, so if there are more
    /// infers than generic params then we will not process all infers thus not generating
    /// more generic params then needed (anyway it is an error).
    fn create_slots_from_args(
        tcx: TyCtxt<'_>,
        args: &AngleBracketedArgs,
        params: &'hir [ty::GenericParamDef],
        add_first_self: bool,
    ) -> TyGenerics<'hir> {
        let mut slots = vec![];
        if add_first_self {
            slots.push(GenericArgSlot::Generate(&params[0], None));
        }

        let params = &params[usize::from(add_first_self)..];
        for (idx, (arg, param)) in args.args.iter().zip(params).enumerate() {
            let AngleBracketedArg::Arg(arg) = arg else { continue };

            let is_infer = match arg {
                GenericArg::Lifetime(lt) => lt.ident.name == kw::UnderscoreLifetime,
                GenericArg::Type(ty) => ty.is_maybe_parenthesised_infer(),
                GenericArg::Const(_) => false,
            };

            // If `'_` is used instead of `_` (or vice versa) we emit a meaningful
            // error instead of processing this infer or leaving it as is for signature
            // inheritance.
            if is_infer
                && matches!(
                    (arg, &param.kind),
                    (
                        GenericArg::Lifetime(_),
                        GenericParamDefKind::Type { .. } | GenericParamDefKind::Const { .. }
                    ) | (
                        GenericArg::Type(_) | GenericArg::Const(_),
                        GenericParamDefKind::Lifetime { .. }
                    )
                )
            {
                let (actual, expected) = if matches!(arg, GenericArg::Lifetime(..)) {
                    (kw::UnderscoreLifetime, kw::Underscore)
                } else {
                    (kw::Underscore, kw::UnderscoreLifetime)
                };

                tcx.dcx().emit_err(DelegationInfersMismatch { span: arg.span(), actual, expected });
            }

            slots.push(match is_infer {
                true => GenericArgSlot::Generate(param, Some(idx)),
                false => GenericArgSlot::UserSpecified,
            });
        }

        slots
    }
}

impl<'hir> GenericsGenerationResults<'hir> {
    pub(super) fn all_params(&self) -> impl Iterator<Item = hir::GenericParam<'hir>> {
        let parent = self.parent.generics.hir_generics_or_empty().params;
        let child = self.child.generics.hir_generics_or_empty().params;

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
    pub(super) fn all_predicates(&self) -> impl Iterator<Item = hir::WherePredicate<'hir>> {
        self.parent
            .generics
            .hir_generics_or_empty()
            .predicates
            .into_iter()
            .chain(self.child.generics.hir_generics_or_empty().predicates)
            .copied()
    }
}

impl<'hir> LoweringContext<'_, 'hir> {
    fn uplift_delegation_generic_params(
        &mut self,
        span: Span,
        params: &[GenericArgSlot<&ty::GenericParamDef>],
        rename_self: bool,
    ) -> &'hir hir::Generics<'hir> {
        let params = self.arena.alloc_from_iter(params.iter().flat_map(|p| {
            let GenericArgSlot::Generate(p, _) = p else { return None };

            let def_kind = match p.kind {
                GenericParamDefKind::Lifetime => DefKind::LifetimeParam,
                GenericParamDefKind::Type { .. } => DefKind::TyParam,
                GenericParamDefKind::Const { .. } => DefKind::ConstParam,
            };

            // Rename Self generic param to This so it is properly propagated.
            // If the user will create a function `fn foo<Self>() {}` with generic
            // param "Self" then it will not be generated in HIR, the same thing
            // applies to traits, `trait Trait<Self> {}` will be represented as
            // `trait Trait {}` in HIR and "unexpected keyword `Self` in generic parameters"
            // error will be emitted.
            // Note that we do not rename `Self` to `This` after non-recursive reuse
            // from Trait, in this case the `Self` should not be propagated
            // (we rely that implicit `Self` generic param of a trait is named "Self")
            // and it is OK to have Self generic param generated during lowering.
            let param_name =
                if rename_self && p.name == kw::SelfUpper { sym::This } else { p.name };

            let param_ident = Ident::new(param_name, span);
            let def_name = Some(param_ident.name);
            let node_id = self.next_node_id();

            let def_id = self.create_def(node_id, def_name, def_kind, span);

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

            Some(hir::GenericParam {
                hir_id,
                colon_span: Some(span),
                def_id,
                kind,
                name: hir::ParamName::Plain(param_ident),
                pure_wrt_drop: p.pure_wrt_drop,
                source: hir::GenericParamSource::Generics,
                span,
            })
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

    pub(super) fn create_generic_arg_path(
        &mut self,
        p: &hir::GenericParam<'hir>,
    ) -> hir::QPath<'hir> {
        let res = Res::Def(
            match p.kind {
                hir::GenericParamKind::Lifetime { .. } => DefKind::LifetimeParam,
                hir::GenericParamKind::Type { .. } => DefKind::TyParam,
                hir::GenericParamKind::Const { .. } => DefKind::ConstParam,
            },
            p.def_id.to_def_id(),
        );

        self.create_resolved_path(res, p.name.ident(), p.span)
    }

    pub(super) fn create_resolved_path(
        &mut self,
        res: Res,
        ident: Ident,
        span: Span,
    ) -> hir::QPath<'hir> {
        hir::QPath::Resolved(
            None,
            self.arena.alloc(hir::Path {
                via_crate: None,
                segments: self.arena.alloc_slice(&[hir::PathSegment {
                    args: None,
                    hir_id: self.next_id(),
                    ident,
                    infer_args: false,
                    res,
                    delegation_child_segment: false,
                }]),
                res,
                span,
            }),
        )
    }
}
