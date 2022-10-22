use hir::intravisit::{self, Visitor};
use hir::{GenericParamKind, HirId};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::lint;
use rustc_span::symbol::{kw, Symbol};

pub(super) fn generics_of(tcx: TyCtxt<'_>, def_id: DefId) -> ty::Generics {
    use rustc_hir::*;

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id.expect_local());

    let node = tcx.hir().get(hir_id);
    let parent_def_id = match node {
        Node::ImplItem(_)
        | Node::TraitItem(_)
        | Node::Variant(_)
        | Node::Ctor(..)
        | Node::Field(_) => {
            let parent_id = tcx.hir().get_parent_item(hir_id);
            Some(parent_id.to_def_id())
        }
        // FIXME(#43408) always enable this once `lazy_normalization` is
        // stable enough and does not need a feature gate anymore.
        Node::AnonConst(_) => {
            let parent_def_id = tcx.hir().get_parent_item(hir_id);

            let mut in_param_ty = false;
            for (_parent, node) in tcx.hir().parent_iter(hir_id) {
                if let Some(generics) = node.generics() {
                    let mut visitor = AnonConstInParamTyDetector {
                        in_param_ty: false,
                        found_anon_const_in_param_ty: false,
                        ct: hir_id,
                    };

                    visitor.visit_generics(generics);
                    in_param_ty = visitor.found_anon_const_in_param_ty;
                    break;
                }
            }

            if in_param_ty {
                // We do not allow generic parameters in anon consts if we are inside
                // of a const parameter type, e.g. `struct Foo<const N: usize, const M: [u8; N]>` is not allowed.
                None
            } else if tcx.lazy_normalization() {
                if let Some(param_id) = tcx.hir().opt_const_param_default_param_hir_id(hir_id) {
                    // If the def_id we are calling generics_of on is an anon ct default i.e:
                    //
                    // struct Foo<const N: usize = { .. }>;
                    //        ^^^       ^          ^^^^^^ def id of this anon const
                    //        ^         ^ param_id
                    //        ^ parent_def_id
                    //
                    // then we only want to return generics for params to the left of `N`. If we don't do that we
                    // end up with that const looking like: `ty::ConstKind::Unevaluated(def_id, substs: [N#0])`.
                    //
                    // This causes ICEs (#86580) when building the substs for Foo in `fn foo() -> Foo { .. }` as
                    // we substitute the defaults with the partially built substs when we build the substs. Subst'ing
                    // the `N#0` on the unevaluated const indexes into the empty substs we're in the process of building.
                    //
                    // We fix this by having this function return the parent's generics ourselves and truncating the
                    // generics to only include non-forward declared params (with the exception of the `Self` ty)
                    //
                    // For the above code example that means we want `substs: []`
                    // For the following struct def we want `substs: [N#0]` when generics_of is called on
                    // the def id of the `{ N + 1 }` anon const
                    // struct Foo<const N: usize, const M: usize = { N + 1 }>;
                    //
                    // This has some implications for how we get the predicates available to the anon const
                    // see `explicit_predicates_of` for more information on this
                    let generics = tcx.generics_of(parent_def_id.to_def_id());
                    let param_def = tcx.hir().local_def_id(param_id).to_def_id();
                    let param_def_idx = generics.param_def_id_to_index[&param_def];
                    // In the above example this would be .params[..N#0]
                    let params = generics.params[..param_def_idx as usize].to_owned();
                    let param_def_id_to_index =
                        params.iter().map(|param| (param.def_id, param.index)).collect();

                    return ty::Generics {
                        // we set the parent of these generics to be our parent's parent so that we
                        // dont end up with substs: [N, M, N] for the const default on a struct like this:
                        // struct Foo<const N: usize, const M: usize = { ... }>;
                        parent: generics.parent,
                        parent_count: generics.parent_count,
                        params,
                        param_def_id_to_index,
                        has_self: generics.has_self,
                    };
                }

                // HACK(eddyb) this provides the correct generics when
                // `feature(generic_const_expressions)` is enabled, so that const expressions
                // used with const generics, e.g. `Foo<{N+1}>`, can work at all.
                //
                // Note that we do not supply the parent generics when using
                // `min_const_generics`.
                Some(parent_def_id.to_def_id())
            } else {
                let parent_node = tcx.hir().get(tcx.hir().get_parent_node(hir_id));
                match parent_node {
                    // HACK(eddyb) this provides the correct generics for repeat
                    // expressions' count (i.e. `N` in `[x; N]`), and explicit
                    // `enum` discriminants (i.e. `D` in `enum Foo { Bar = D }`),
                    // as they shouldn't be able to cause query cycle errors.
                    Node::Expr(&Expr { kind: ExprKind::Repeat(_, ref constant), .. })
                        if constant.hir_id() == hir_id =>
                    {
                        Some(parent_def_id.to_def_id())
                    }
                    Node::Variant(Variant { disr_expr: Some(ref constant), .. })
                        if constant.hir_id == hir_id =>
                    {
                        Some(parent_def_id.to_def_id())
                    }
                    Node::Expr(&Expr { kind: ExprKind::ConstBlock(_), .. }) => {
                        Some(tcx.typeck_root_def_id(def_id))
                    }
                    // Exclude `GlobalAsm` here which cannot have generics.
                    Node::Expr(&Expr { kind: ExprKind::InlineAsm(asm), .. })
                        if asm.operands.iter().any(|(op, _op_sp)| match op {
                            hir::InlineAsmOperand::Const { anon_const }
                            | hir::InlineAsmOperand::SymFn { anon_const } => {
                                anon_const.hir_id == hir_id
                            }
                            _ => false,
                        }) =>
                    {
                        Some(parent_def_id.to_def_id())
                    }
                    _ => None,
                }
            }
        }
        Node::Expr(&hir::Expr { kind: hir::ExprKind::Closure { .. }, .. }) => {
            Some(tcx.typeck_root_def_id(def_id))
        }
        Node::Item(item) => match item.kind {
            ItemKind::OpaqueTy(hir::OpaqueTy { origin, in_trait, .. }) => {
                if let hir::OpaqueTyOrigin::FnReturn(fn_def_id)
                | hir::OpaqueTyOrigin::AsyncFn(fn_def_id) = origin
                {
                    if in_trait {
                        assert!(matches!(tcx.def_kind(fn_def_id), DefKind::AssocFn))
                    } else {
                        assert!(matches!(tcx.def_kind(fn_def_id), DefKind::AssocFn | DefKind::Fn))
                    }
                }
                let parent_id = tcx.hir().get_parent_item(hir_id);
                assert_ne!(parent_id, hir::CRATE_OWNER_ID);
                debug!("generics_of: parent of opaque ty {:?} is {:?}", def_id, parent_id);
                // Opaque types are always nested within another item, and
                // inherit the generics of the item.
                Some(parent_id.to_def_id())
            }
            _ => None,
        },
        _ => None,
    };

    enum Defaults {
        Allowed,
        // See #36887
        FutureCompatDisallowed,
        Deny,
    }

    let no_generics = hir::Generics::empty();
    let ast_generics = node.generics().unwrap_or(&no_generics);
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
                        def_id,
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
                | ItemKind::OpaqueTy(..)
                | ItemKind::Union(..) => (None, Defaults::Allowed),
                _ => (None, Defaults::FutureCompatDisallowed),
            }
        }

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
        generics.parent_count + generics.params.len()
    });

    let mut params: Vec<_> = Vec::with_capacity(ast_generics.params.len() + has_self as usize);

    if let Some(opt_self) = opt_self {
        params.push(opt_self);
    }

    // Now create the real lifetime, type and const parameters.
    const TYPE_DEFAULT_NOT_ALLOWED: &'static str = "defaults for type parameters are only allowed in \
    `struct`, `enum`, `type`, or `trait` definitions";

    // When late-bound lifetimes did not appear in generics, the user was allowed to introduce as
    // many late-bound lifetimes as they wanted, and they were not counted in the substs.
    // In order to keep this possibility while having substs behaving reasonably, we push
    // late-bound lifetimes to the end of the generics.  That way, we can keep the behaviour of
    // trait to impl substitution rebase with hopefully minimal breakage.
    params.extend(
        ast_generics
            .params
            .iter()
            .filter(|param| match param.kind {
                GenericParamKind::Lifetime { .. } => !tcx.is_late_bound(param.hir_id),
                GenericParamKind::Type { .. } | GenericParamKind::Const { .. } => true,
            })
            .enumerate()
            .map(|(i, param)| {
                let kind = match param.kind {
                    GenericParamKind::Lifetime { .. } => {
                        ty::GenericParamDefKind::Lifetime { late_bound: false }
                    }
                    GenericParamKind::Type { ref default, synthetic, .. } => {
                        if default.is_some() {
                            match allow_defaults {
                                Defaults::Allowed => {}
                                Defaults::FutureCompatDisallowed
                                    if tcx.features().default_type_parameter_fallback => {}
                                Defaults::FutureCompatDisallowed => {
                                    tcx.struct_span_lint_hir(
                                        lint::builtin::INVALID_TYPE_PARAM_DEFAULT,
                                        param.hir_id,
                                        param.span,
                                        TYPE_DEFAULT_NOT_ALLOWED,
                                        |lint| lint,
                                    );
                                }
                                Defaults::Deny => {
                                    tcx.sess.span_err(param.span, TYPE_DEFAULT_NOT_ALLOWED);
                                }
                            }
                        }

                        ty::GenericParamDefKind::Type { has_default: default.is_some(), synthetic }
                    }
                    GenericParamKind::Const { default, .. } => {
                        if !matches!(allow_defaults, Defaults::Allowed) && default.is_some() {
                            tcx.sess.span_err(
                                param.span,
                                "defaults for const parameters are only allowed in \
                    `struct`, `enum`, `type`, or `trait` definitions",
                            );
                        }

                        ty::GenericParamDefKind::Const { has_default: default.is_some() }
                    }
                };

                ty::GenericParamDef {
                    name: param.name.ident().name,
                    index: own_start + i as u32,
                    def_id: tcx.hir().local_def_id(param.hir_id).to_def_id(),
                    pure_wrt_drop: param.pure_wrt_drop,
                    kind,
                }
            }),
    );

    let own_start = (parent_count + params.len()) as u32;

    params.extend(
        ast_generics
            .params
            .iter()
            .filter(|param| match param.kind {
                GenericParamKind::Lifetime { .. } => tcx.is_late_bound(param.hir_id),
                GenericParamKind::Type { .. } | GenericParamKind::Const { .. } => false,
            })
            .enumerate()
            .map(|(i, param)| {
                let kind = match param.kind {
                    GenericParamKind::Lifetime { .. } => {
                        ty::GenericParamDefKind::Lifetime { late_bound: true }
                    }
                    GenericParamKind::Type { .. } | GenericParamKind::Const { .. } => bug!(),
                };

                ty::GenericParamDef {
                    name: param.name.ident().name,
                    index: own_start + i as u32,
                    def_id: tcx.hir().local_def_id(param.hir_id).to_def_id(),
                    pure_wrt_drop: param.pure_wrt_drop,
                    kind,
                }
            }),
    );

    let type_start = (parent_count + params.len()) as u32;

    // provide junk type parameter defs - the only place that
    // cares about anything but the length is instantiation,
    // and we don't do that for closures.
    if let Node::Expr(&hir::Expr {
        kind: hir::ExprKind::Closure(hir::Closure { movability: gen, .. }),
        ..
    }) = node
    {
        let dummy_args = if gen.is_some() {
            &["<resume_ty>", "<yield_ty>", "<return_ty>", "<witness>", "<upvars>"][..]
        } else {
            &["<closure_kind>", "<closure_signature>", "<upvars>"][..]
        };

        params.extend(dummy_args.iter().enumerate().map(|(i, &arg)| ty::GenericParamDef {
            index: type_start + i as u32,
            name: Symbol::intern(arg),
            def_id,
            pure_wrt_drop: false,
            kind: ty::GenericParamDefKind::Type { has_default: false, synthetic: false },
        }));
    }

    // provide junk type parameter defs for const blocks.
    if let Node::AnonConst(_) = node {
        let parent_node = tcx.hir().get(tcx.hir().get_parent_node(hir_id));
        if let Node::Expr(&Expr { kind: ExprKind::ConstBlock(_), .. }) = parent_node {
            params.push(ty::GenericParamDef {
                index: type_start,
                name: Symbol::intern("<const_ty>"),
                def_id,
                pure_wrt_drop: false,
                kind: ty::GenericParamDefKind::Type { has_default: false, synthetic: false },
            });
        }
    }

    let param_def_id_to_index = params.iter().map(|param| (param.def_id, param.index)).collect();

    ty::Generics {
        parent: parent_def_id,
        parent_count,
        params,
        param_def_id_to_index,
        has_self: has_self || parent_has_self,
    }
}

struct AnonConstInParamTyDetector {
    in_param_ty: bool,
    found_anon_const_in_param_ty: bool,
    ct: HirId,
}

impl<'v> Visitor<'v> for AnonConstInParamTyDetector {
    fn visit_generic_param(&mut self, p: &'v hir::GenericParam<'v>) {
        if let GenericParamKind::Const { ty, default: _ } = p.kind {
            let prev = self.in_param_ty;
            self.in_param_ty = true;
            self.visit_ty(ty);
            self.in_param_ty = prev;
        }
    }

    fn visit_anon_const(&mut self, c: &'v hir::AnonConst) {
        if self.in_param_ty && self.ct == c.hir_id {
            self.found_anon_const_in_param_ty = true;
        } else {
            intravisit::walk_anon_const(self, c)
        }
    }
}
