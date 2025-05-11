use std::assert_matches::assert_matches;
use std::ops::ControlFlow;

use hir::intravisit::{self, Visitor};
use hir::{GenericParamKind, HirId, Node};
use rustc_hir::def::DefKind;
use rustc_hir::def_id::LocalDefId;
use rustc_hir::intravisit::VisitorExt;
use rustc_hir::{self as hir, AmbigArg};
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::lint;
use rustc_span::{Span, Symbol, kw};
use tracing::{debug, instrument};

use crate::delegation::inherit_generics_for_delegation_item;
use crate::middle::resolve_bound_vars as rbv;

#[instrument(level = "debug", skip(tcx), ret)]
pub(super) fn generics_of(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::Generics {
    use rustc_hir::*;

    // For an RPITIT, synthesize generics which are equal to the opaque's generics
    // and parent fn's generics compressed into one list.
    if let Some(ty::ImplTraitInTraitData::Trait { fn_def_id, opaque_def_id }) =
        tcx.opt_rpitit_info(def_id.to_def_id())
    {
        debug!("RPITIT fn_def_id={fn_def_id:?} opaque_def_id={opaque_def_id:?}");
        let trait_def_id = tcx.parent(fn_def_id);
        let opaque_ty_generics = tcx.generics_of(opaque_def_id);
        let opaque_ty_parent_count = opaque_ty_generics.parent_count;
        let mut own_params = opaque_ty_generics.own_params.clone();

        let parent_generics = tcx.generics_of(trait_def_id);
        let parent_count = parent_generics.parent_count + parent_generics.own_params.len();

        let mut trait_fn_params = tcx.generics_of(fn_def_id).own_params.clone();

        for param in &mut own_params {
            param.index = param.index + parent_count as u32 + trait_fn_params.len() as u32
                - opaque_ty_parent_count as u32;
        }

        trait_fn_params.extend(own_params);
        own_params = trait_fn_params;

        let param_def_id_to_index =
            own_params.iter().map(|param| (param.def_id, param.index)).collect();

        return ty::Generics {
            parent: Some(trait_def_id),
            parent_count,
            own_params,
            param_def_id_to_index,
            has_self: opaque_ty_generics.has_self,
            has_late_bound_regions: opaque_ty_generics.has_late_bound_regions,
        };
    }

    let hir_id = tcx.local_def_id_to_hir_id(def_id);

    let node = tcx.hir_node(hir_id);
    if let Some(sig) = node.fn_sig()
        && let Some(sig_id) = sig.decl.opt_delegation_sig_id()
    {
        return inherit_generics_for_delegation_item(tcx, def_id, sig_id);
    }

    let parent_def_id = match node {
        Node::ImplItem(_)
        | Node::TraitItem(_)
        | Node::Variant(_)
        | Node::Ctor(..)
        | Node::Field(_) => {
            let parent_id = tcx.hir_get_parent_item(hir_id);
            Some(parent_id.to_def_id())
        }
        // FIXME(#43408) always enable this once `lazy_normalization` is
        // stable enough and does not need a feature gate anymore.
        Node::AnonConst(_) => {
            let parent_did = tcx.parent(def_id.to_def_id());

            // We don't do this unconditionally because the `DefId` parent of an anon const
            // might be an implicitly created closure during `async fn` desugaring. This would
            // have the wrong generics.
            //
            // i.e. `async fn foo<'a>() { let a = [(); { 1 + 2 }]; bar().await() }`
            // would implicitly have a closure in its body that would be the parent of
            // the `{ 1 + 2 }` anon const. This closure's generics is simply a witness
            // instead of `['a]`.
            let parent_did = if let DefKind::AnonConst = tcx.def_kind(parent_did) {
                parent_did
            } else {
                tcx.hir_get_parent_item(hir_id).to_def_id()
            };
            debug!(?parent_did);

            let mut in_param_ty = false;
            for (_parent, node) in tcx.hir_parent_iter(hir_id) {
                if let Some(generics) = node.generics() {
                    let mut visitor = AnonConstInParamTyDetector { in_param_ty: false, ct: hir_id };

                    in_param_ty = visitor.visit_generics(generics).is_break();
                    break;
                }
            }

            if in_param_ty {
                // We do not allow generic parameters in anon consts if we are inside
                // of a const parameter type, e.g. `struct Foo<const N: usize, const M: [u8; N]>` is not allowed.
                None
            } else if tcx.features().generic_const_exprs() {
                let parent_node = tcx.parent_hir_node(hir_id);
                debug!(?parent_node);
                if let Node::Variant(Variant { disr_expr: Some(constant), .. }) = parent_node
                    && constant.hir_id == hir_id
                {
                    // enum variant discriminants are not allowed to use any kind of generics
                    None
                } else if let Some(param_id) = tcx.hir_opt_const_param_default_param_def_id(hir_id)
                {
                    // If the def_id we are calling generics_of on is an anon ct default i.e:
                    //
                    // struct Foo<const N: usize = { .. }>;
                    //        ^^^       ^          ^^^^^^ def id of this anon const
                    //        ^         ^ param_id
                    //        ^ parent_def_id
                    //
                    // then we only want to return generics for params to the left of `N`. If we don't do that we
                    // end up with that const looking like: `ty::ConstKind::Unevaluated(def_id, args: [N#0])`.
                    //
                    // This causes ICEs (#86580) when building the args for Foo in `fn foo() -> Foo { .. }` as
                    // we instantiate the defaults with the partially built args when we build the args. Instantiating
                    // the `N#0` on the unevaluated const indexes into the empty args we're in the process of building.
                    //
                    // We fix this by having this function return the parent's generics ourselves and truncating the
                    // generics to only include non-forward declared params (with the exception of the `Self` ty)
                    //
                    // For the above code example that means we want `args: []`
                    // For the following struct def we want `args: [N#0]` when generics_of is called on
                    // the def id of the `{ N + 1 }` anon const
                    // struct Foo<const N: usize, const M: usize = { N + 1 }>;
                    //
                    // This has some implications for how we get the predicates available to the anon const
                    // see `explicit_predicates_of` for more information on this
                    let generics = tcx.generics_of(parent_did);
                    let param_def_idx = generics.param_def_id_to_index[&param_id.to_def_id()];
                    // In the above example this would be .params[..N#0]
                    let own_params = generics.params_to(param_def_idx as usize, tcx).to_owned();
                    let param_def_id_to_index =
                        own_params.iter().map(|param| (param.def_id, param.index)).collect();

                    return ty::Generics {
                        // we set the parent of these generics to be our parent's parent so that we
                        // dont end up with args: [N, M, N] for the const default on a struct like this:
                        // struct Foo<const N: usize, const M: usize = { ... }>;
                        parent: generics.parent,
                        parent_count: generics.parent_count,
                        own_params,
                        param_def_id_to_index,
                        has_self: generics.has_self,
                        has_late_bound_regions: generics.has_late_bound_regions,
                    };
                } else {
                    // HACK(eddyb) this provides the correct generics when
                    // `feature(generic_const_expressions)` is enabled, so that const expressions
                    // used with const generics, e.g. `Foo<{N+1}>`, can work at all.
                    //
                    // Note that we do not supply the parent generics when using
                    // `min_const_generics`.
                    Some(parent_did)
                }
            } else {
                let parent_node = tcx.parent_hir_node(hir_id);
                let parent_node = match parent_node {
                    Node::ConstArg(ca) => tcx.parent_hir_node(ca.hir_id),
                    _ => parent_node,
                };
                match parent_node {
                    // HACK(eddyb) this provides the correct generics for repeat
                    // expressions' count (i.e. `N` in `[x; N]`), and explicit
                    // `enum` discriminants (i.e. `D` in `enum Foo { Bar = D }`),
                    // as they shouldn't be able to cause query cycle errors.
                    Node::Expr(Expr { kind: ExprKind::Repeat(_, ct), .. })
                        if ct.anon_const_hir_id() == Some(hir_id) =>
                    {
                        Some(parent_did)
                    }
                    Node::TyPat(_) => Some(parent_did),
                    // Field default values inherit the ADT's generics.
                    Node::Field(_) => Some(parent_did),
                    _ => None,
                }
            }
        }
        Node::ConstBlock(_)
        | Node::Expr(&hir::Expr { kind: hir::ExprKind::Closure { .. }, .. }) => {
            Some(tcx.typeck_root_def_id(def_id.to_def_id()))
        }
        Node::OpaqueTy(&hir::OpaqueTy {
            origin:
                hir::OpaqueTyOrigin::FnReturn { parent: fn_def_id, in_trait_or_impl }
                | hir::OpaqueTyOrigin::AsyncFn { parent: fn_def_id, in_trait_or_impl },
            ..
        }) => {
            if in_trait_or_impl.is_some() {
                assert_matches!(tcx.def_kind(fn_def_id), DefKind::AssocFn);
            } else {
                assert_matches!(tcx.def_kind(fn_def_id), DefKind::AssocFn | DefKind::Fn);
            }
            Some(fn_def_id.to_def_id())
        }
        Node::OpaqueTy(&hir::OpaqueTy {
            origin: hir::OpaqueTyOrigin::TyAlias { parent, in_assoc_ty },
            ..
        }) => {
            if in_assoc_ty {
                assert_matches!(tcx.def_kind(parent), DefKind::AssocTy);
            } else {
                assert_matches!(tcx.def_kind(parent), DefKind::TyAlias);
            }
            debug!("generics_of: parent of opaque ty {:?} is {:?}", def_id, parent);
            // Opaque types are always nested within another item, and
            // inherit the generics of the item.
            Some(parent.to_def_id())
        }
        _ => None,
    };

    enum Defaults {
        Allowed,
        // See #36887
        FutureCompatDisallowed,
        Deny,
    }

    let hir_generics = node.generics().unwrap_or(hir::Generics::empty());
    let (opt_self, allow_defaults) = match node {
        Node::Item(item) => {
            match item.kind {
                ItemKind::Trait(..) | ItemKind::TraitAlias(..) => {
                    // Add in the self type parameter.
                    //
                    // Something of a hack: use the node id for the trait, also as
                    // the node id for the Self type parameter.
                    let opt_self = Some(ty::GenericParamDef {
                        index: 0,
                        name: kw::SelfUpper,
                        def_id: def_id.to_def_id(),
                        pure_wrt_drop: false,
                        kind: ty::GenericParamDefKind::Type {
                            has_default: false,
                            synthetic: false,
                        },
                    });

                    (opt_self, Defaults::Allowed)
                }
                ItemKind::TyAlias(..)
                | ItemKind::Enum(..)
                | ItemKind::Struct(..)
                | ItemKind::Union(..) => (None, Defaults::Allowed),
                ItemKind::Const(..) => (None, Defaults::Deny),
                _ => (None, Defaults::FutureCompatDisallowed),
            }
        }

        Node::OpaqueTy(..) => (None, Defaults::Allowed),

        // GATs
        Node::TraitItem(item) if matches!(item.kind, TraitItemKind::Type(..)) => {
            (None, Defaults::Deny)
        }
        Node::ImplItem(item) if matches!(item.kind, ImplItemKind::Type(..)) => {
            (None, Defaults::Deny)
        }

        _ => (None, Defaults::FutureCompatDisallowed),
    };

    let has_self = opt_self.is_some();
    let mut parent_has_self = false;
    let mut own_start = has_self as u32;
    let parent_count = parent_def_id.map_or(0, |def_id| {
        let generics = tcx.generics_of(def_id);
        assert!(!has_self);
        parent_has_self = generics.has_self;
        own_start = generics.count() as u32;
        generics.parent_count + generics.own_params.len()
    });

    let mut own_params: Vec<_> = Vec::with_capacity(hir_generics.params.len() + has_self as usize);

    if let Some(opt_self) = opt_self {
        own_params.push(opt_self);
    }

    let early_lifetimes = super::early_bound_lifetimes_from_generics(tcx, hir_generics);
    own_params.extend(early_lifetimes.enumerate().map(|(i, param)| ty::GenericParamDef {
        name: param.name.ident().name,
        index: own_start + i as u32,
        def_id: param.def_id.to_def_id(),
        pure_wrt_drop: param.pure_wrt_drop,
        kind: ty::GenericParamDefKind::Lifetime,
    }));

    // Now create the real type and const parameters.
    let type_start = own_start - has_self as u32 + own_params.len() as u32;
    let mut i: u32 = 0;
    let mut next_index = || {
        let prev = i;
        i += 1;
        prev + type_start
    };

    const TYPE_DEFAULT_NOT_ALLOWED: &'static str = "defaults for type parameters are only allowed in \
    `struct`, `enum`, `type`, or `trait` definitions";

    own_params.extend(hir_generics.params.iter().filter_map(|param| match param.kind {
        GenericParamKind::Lifetime { .. } => None,
        GenericParamKind::Type { default, synthetic, .. } => {
            if default.is_some() {
                match allow_defaults {
                    Defaults::Allowed => {}
                    Defaults::FutureCompatDisallowed => {
                        tcx.node_span_lint(
                            lint::builtin::INVALID_TYPE_PARAM_DEFAULT,
                            param.hir_id,
                            param.span,
                            |lint| {
                                lint.primary_message(TYPE_DEFAULT_NOT_ALLOWED);
                            },
                        );
                    }
                    Defaults::Deny => {
                        tcx.dcx().span_err(param.span, TYPE_DEFAULT_NOT_ALLOWED);
                    }
                }
            }

            let kind = ty::GenericParamDefKind::Type { has_default: default.is_some(), synthetic };

            Some(ty::GenericParamDef {
                index: next_index(),
                name: param.name.ident().name,
                def_id: param.def_id.to_def_id(),
                pure_wrt_drop: param.pure_wrt_drop,
                kind,
            })
        }
        GenericParamKind::Const { ty: _, default, synthetic } => {
            if !matches!(allow_defaults, Defaults::Allowed) && default.is_some() {
                tcx.dcx().span_err(
                    param.span,
                    "defaults for const parameters are only allowed in \
                    `struct`, `enum`, `type`, or `trait` definitions",
                );
            }

            let index = next_index();

            Some(ty::GenericParamDef {
                index,
                name: param.name.ident().name,
                def_id: param.def_id.to_def_id(),
                pure_wrt_drop: param.pure_wrt_drop,
                kind: ty::GenericParamDefKind::Const { has_default: default.is_some(), synthetic },
            })
        }
    }));

    // provide junk type parameter defs - the only place that
    // cares about anything but the length is instantiation,
    // and we don't do that for closures.
    if let Node::Expr(&hir::Expr {
        kind: hir::ExprKind::Closure(hir::Closure { kind, .. }), ..
    }) = node
    {
        // See `ClosureArgsParts`, `CoroutineArgsParts`, and `CoroutineClosureArgsParts`
        // for info on the usage of each of these fields.
        let dummy_args = match kind {
            ClosureKind::Closure => &["<closure_kind>", "<closure_signature>", "<upvars>"][..],
            ClosureKind::Coroutine(_) => &[
                "<coroutine_kind>",
                "<resume_ty>",
                "<yield_ty>",
                "<return_ty>",
                "<witness>",
                "<upvars>",
            ][..],
            ClosureKind::CoroutineClosure(_) => &[
                "<closure_kind>",
                "<closure_signature_parts>",
                "<upvars>",
                "<bound_captures_by_ref>",
                "<witness>",
            ][..],
        };

        own_params.extend(dummy_args.iter().map(|&arg| ty::GenericParamDef {
            index: next_index(),
            name: Symbol::intern(arg),
            def_id: def_id.to_def_id(),
            pure_wrt_drop: false,
            kind: ty::GenericParamDefKind::Type { has_default: false, synthetic: false },
        }));
    }

    // provide junk type parameter defs for const blocks.
    if let Node::ConstBlock(_) = node {
        own_params.push(ty::GenericParamDef {
            index: next_index(),
            name: rustc_span::sym::const_ty_placeholder,
            def_id: def_id.to_def_id(),
            pure_wrt_drop: false,
            kind: ty::GenericParamDefKind::Type { has_default: false, synthetic: false },
        });
    }

    if let Node::OpaqueTy(&hir::OpaqueTy { .. }) = node {
        assert!(own_params.is_empty());

        let lifetimes = tcx.opaque_captured_lifetimes(def_id);
        debug!(?lifetimes);

        own_params.extend(lifetimes.iter().map(|&(_, param)| ty::GenericParamDef {
            name: tcx.item_name(param.to_def_id()),
            index: next_index(),
            def_id: param.to_def_id(),
            pure_wrt_drop: false,
            kind: ty::GenericParamDefKind::Lifetime,
        }))
    }

    let param_def_id_to_index =
        own_params.iter().map(|param| (param.def_id, param.index)).collect();

    ty::Generics {
        parent: parent_def_id,
        parent_count,
        own_params,
        param_def_id_to_index,
        has_self: has_self || parent_has_self,
        has_late_bound_regions: has_late_bound_regions(tcx, node),
    }
}

fn has_late_bound_regions<'tcx>(tcx: TyCtxt<'tcx>, node: Node<'tcx>) -> Option<Span> {
    struct LateBoundRegionsDetector<'tcx> {
        tcx: TyCtxt<'tcx>,
        outer_index: ty::DebruijnIndex,
    }

    impl<'tcx> Visitor<'tcx> for LateBoundRegionsDetector<'tcx> {
        type Result = ControlFlow<Span>;
        fn visit_ty(&mut self, ty: &'tcx hir::Ty<'tcx, AmbigArg>) -> ControlFlow<Span> {
            match ty.kind {
                hir::TyKind::BareFn(..) => {
                    self.outer_index.shift_in(1);
                    let res = intravisit::walk_ty(self, ty);
                    self.outer_index.shift_out(1);
                    res
                }
                hir::TyKind::UnsafeBinder(_) => {
                    self.outer_index.shift_in(1);
                    let res = intravisit::walk_ty(self, ty);
                    self.outer_index.shift_out(1);
                    res
                }
                _ => intravisit::walk_ty(self, ty),
            }
        }

        fn visit_poly_trait_ref(&mut self, tr: &'tcx hir::PolyTraitRef<'tcx>) -> ControlFlow<Span> {
            self.outer_index.shift_in(1);
            let res = intravisit::walk_poly_trait_ref(self, tr);
            self.outer_index.shift_out(1);
            res
        }

        fn visit_lifetime(&mut self, lt: &'tcx hir::Lifetime) -> ControlFlow<Span> {
            match self.tcx.named_bound_var(lt.hir_id) {
                Some(rbv::ResolvedArg::StaticLifetime | rbv::ResolvedArg::EarlyBound(..)) => {
                    ControlFlow::Continue(())
                }
                Some(rbv::ResolvedArg::LateBound(debruijn, _, _))
                    if debruijn < self.outer_index =>
                {
                    ControlFlow::Continue(())
                }
                Some(
                    rbv::ResolvedArg::LateBound(..)
                    | rbv::ResolvedArg::Free(..)
                    | rbv::ResolvedArg::Error(_),
                )
                | None => ControlFlow::Break(lt.ident.span),
            }
        }
    }

    fn has_late_bound_regions<'tcx>(
        tcx: TyCtxt<'tcx>,
        generics: &'tcx hir::Generics<'tcx>,
        decl: &'tcx hir::FnDecl<'tcx>,
    ) -> Option<Span> {
        let mut visitor = LateBoundRegionsDetector { tcx, outer_index: ty::INNERMOST };
        for param in generics.params {
            if let GenericParamKind::Lifetime { .. } = param.kind {
                if tcx.is_late_bound(param.hir_id) {
                    return Some(param.span);
                }
            }
        }
        visitor.visit_fn_decl(decl).break_value()
    }

    let decl = node.fn_decl()?;
    let generics = node.generics()?;
    has_late_bound_regions(tcx, generics, decl)
}

struct AnonConstInParamTyDetector {
    in_param_ty: bool,
    ct: HirId,
}

impl<'v> Visitor<'v> for AnonConstInParamTyDetector {
    type Result = ControlFlow<()>;

    fn visit_generic_param(&mut self, p: &'v hir::GenericParam<'v>) -> Self::Result {
        if let GenericParamKind::Const { ty, default: _, synthetic: _ } = p.kind {
            let prev = self.in_param_ty;
            self.in_param_ty = true;
            let res = self.visit_ty_unambig(ty);
            self.in_param_ty = prev;
            res
        } else {
            ControlFlow::Continue(())
        }
    }

    fn visit_anon_const(&mut self, c: &'v hir::AnonConst) -> Self::Result {
        if self.in_param_ty && self.ct == c.hir_id {
            return ControlFlow::Break(());
        }
        intravisit::walk_anon_const(self, c)
    }
}
