use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{struct_span_err, Applicability, StashKey};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit;
use rustc_hir::intravisit::Visitor;
use rustc_hir::Node;
use rustc_middle::hir::map::Map;
use rustc_middle::ty::subst::{GenericArgKind, InternalSubsts};
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::{self, DefIdTree, Ty, TyCtxt, TypeFoldable};
use rustc_session::parse::feature_err;
use rustc_span::symbol::{sym, Ident};
use rustc_span::{Span, DUMMY_SP};
use rustc_trait_selection::traits;

use super::ItemCtxt;
use super::{bad_placeholder_type, is_suggestable_infer_ty};

pub(super) fn type_of(tcx: TyCtxt<'_>, def_id: DefId) -> Ty<'_> {
    use rustc_hir::*;

    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();

    let icx = ItemCtxt::new(tcx, def_id);

    match tcx.hir().get(hir_id) {
        Node::TraitItem(item) => match item.kind {
            TraitItemKind::Fn(..) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id);
                tcx.mk_fn_def(def_id, substs)
            }
            TraitItemKind::Const(ref ty, body_id) => body_id
                .and_then(|body_id| {
                    if is_suggestable_infer_ty(ty) {
                        Some(infer_placeholder_type(tcx, def_id, body_id, ty.span, item.ident))
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| icx.to_ty(ty)),
            TraitItemKind::Type(_, Some(ref ty)) => icx.to_ty(ty),
            TraitItemKind::Type(_, None) => {
                span_bug!(item.span, "associated type missing default");
            }
        },

        Node::ImplItem(item) => match item.kind {
            ImplItemKind::Fn(..) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id);
                tcx.mk_fn_def(def_id, substs)
            }
            ImplItemKind::Const(ref ty, body_id) => {
                if is_suggestable_infer_ty(ty) {
                    infer_placeholder_type(tcx, def_id, body_id, ty.span, item.ident)
                } else {
                    icx.to_ty(ty)
                }
            }
            ImplItemKind::OpaqueTy(_) => {
                if tcx.impl_trait_ref(tcx.hir().get_parent_did(hir_id)).is_none() {
                    report_assoc_ty_on_inherent_impl(tcx, item.span);
                }

                find_opaque_ty_constraints(tcx, def_id)
            }
            ImplItemKind::TyAlias(ref ty) => {
                if tcx.impl_trait_ref(tcx.hir().get_parent_did(hir_id)).is_none() {
                    report_assoc_ty_on_inherent_impl(tcx, item.span);
                }

                icx.to_ty(ty)
            }
        },

        Node::Item(item) => {
            match item.kind {
                ItemKind::Static(ref ty, .., body_id) | ItemKind::Const(ref ty, body_id) => {
                    if is_suggestable_infer_ty(ty) {
                        infer_placeholder_type(tcx, def_id, body_id, ty.span, item.ident)
                    } else {
                        icx.to_ty(ty)
                    }
                }
                ItemKind::TyAlias(ref self_ty, _) | ItemKind::Impl { ref self_ty, .. } => {
                    icx.to_ty(self_ty)
                }
                ItemKind::Fn(..) => {
                    let substs = InternalSubsts::identity_for_item(tcx, def_id);
                    tcx.mk_fn_def(def_id, substs)
                }
                ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Union(..) => {
                    let def = tcx.adt_def(def_id);
                    let substs = InternalSubsts::identity_for_item(tcx, def_id);
                    tcx.mk_adt(def, substs)
                }
                ItemKind::OpaqueTy(OpaqueTy { impl_trait_fn: None, .. }) => {
                    find_opaque_ty_constraints(tcx, def_id)
                }
                // Opaque types desugared from `impl Trait`.
                ItemKind::OpaqueTy(OpaqueTy { impl_trait_fn: Some(owner), origin, .. }) => {
                    let concrete_types = match origin {
                        OpaqueTyOrigin::FnReturn | OpaqueTyOrigin::AsyncFn => {
                            &tcx.mir_borrowck(owner).concrete_opaque_types
                        }
                        OpaqueTyOrigin::Misc => {
                            // We shouldn't leak borrowck results through impl trait in bindings.
                            // For example, we shouldn't be able to tell if `x` in
                            // `let x: impl Sized + 'a = &()` has type `&'static ()` or `&'a ()`.
                            &tcx.typeck_tables_of(owner).concrete_opaque_types
                        }
                        OpaqueTyOrigin::TypeAlias => {
                            span_bug!(item.span, "Type alias impl trait shouldn't have an owner")
                        }
                    };
                    let concrete_ty = concrete_types
                        .get(&def_id)
                        .map(|opaque| opaque.concrete_type)
                        .unwrap_or_else(|| {
                            tcx.sess.delay_span_bug(
                                DUMMY_SP,
                                &format!(
                                    "owner {:?} has no opaque type for {:?} in its tables",
                                    owner, def_id,
                                ),
                            );
                            if tcx.typeck_tables_of(owner).tainted_by_errors {
                                // Some error in the
                                // owner fn prevented us from populating
                                // the `concrete_opaque_types` table.
                                tcx.types.err
                            } else {
                                // We failed to resolve the opaque type or it
                                // resolves to itself. Return the non-revealed
                                // type, which should result in E0720.
                                tcx.mk_opaque(
                                    def_id,
                                    InternalSubsts::identity_for_item(tcx, def_id),
                                )
                            }
                        });
                    debug!("concrete_ty = {:?}", concrete_ty);
                    if concrete_ty.has_erased_regions() {
                        // FIXME(impl_trait_in_bindings) Handle this case.
                        tcx.sess.span_fatal(
                            item.span,
                            "lifetimes in impl Trait types in bindings are not currently supported",
                        );
                    }
                    concrete_ty
                }
                ItemKind::Trait(..)
                | ItemKind::TraitAlias(..)
                | ItemKind::Mod(..)
                | ItemKind::ForeignMod(..)
                | ItemKind::GlobalAsm(..)
                | ItemKind::ExternCrate(..)
                | ItemKind::Use(..) => {
                    span_bug!(
                        item.span,
                        "compute_type_of_item: unexpected item type: {:?}",
                        item.kind
                    );
                }
            }
        }

        Node::ForeignItem(foreign_item) => match foreign_item.kind {
            ForeignItemKind::Fn(..) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id);
                tcx.mk_fn_def(def_id, substs)
            }
            ForeignItemKind::Static(ref t, _) => icx.to_ty(t),
            ForeignItemKind::Type => tcx.mk_foreign(def_id),
        },

        Node::Ctor(&ref def) | Node::Variant(Variant { data: ref def, .. }) => match *def {
            VariantData::Unit(..) | VariantData::Struct(..) => {
                tcx.type_of(tcx.hir().get_parent_did(hir_id))
            }
            VariantData::Tuple(..) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id);
                tcx.mk_fn_def(def_id, substs)
            }
        },

        Node::Field(field) => icx.to_ty(&field.ty),

        Node::Expr(&Expr { kind: ExprKind::Closure(.., gen), .. }) => {
            let substs = InternalSubsts::identity_for_item(tcx, def_id);
            if let Some(movability) = gen {
                tcx.mk_generator(def_id, substs, movability)
            } else {
                tcx.mk_closure(def_id, substs)
            }
        }

        Node::AnonConst(_) => {
            let parent_node = tcx.hir().get(tcx.hir().get_parent_node(hir_id));
            match parent_node {
                Node::Ty(&Ty { kind: TyKind::Array(_, ref constant), .. })
                | Node::Ty(&Ty { kind: TyKind::Typeof(ref constant), .. })
                | Node::Expr(&Expr { kind: ExprKind::Repeat(_, ref constant), .. })
                    if constant.hir_id == hir_id =>
                {
                    tcx.types.usize
                }

                Node::Variant(Variant { disr_expr: Some(ref e), .. }) if e.hir_id == hir_id => {
                    tcx.adt_def(tcx.hir().get_parent_did(hir_id)).repr.discr_type().to_ty(tcx)
                }

                Node::Ty(&Ty { kind: TyKind::Path(_), .. })
                | Node::Expr(&Expr { kind: ExprKind::Struct(..), .. })
                | Node::Expr(&Expr { kind: ExprKind::Path(_), .. })
                | Node::TraitRef(..) => {
                    let path = match parent_node {
                        Node::Ty(&Ty { kind: TyKind::Path(QPath::Resolved(_, path)), .. })
                        | Node::Expr(&Expr {
                            kind:
                                ExprKind::Path(QPath::Resolved(_, path))
                                | ExprKind::Struct(&QPath::Resolved(_, path), ..),
                            ..
                        })
                        | Node::TraitRef(&TraitRef { path, .. }) => &*path,
                        _ => {
                            tcx.sess.delay_span_bug(
                                DUMMY_SP,
                                &format!("unexpected const parent path {:?}", parent_node),
                            );
                            return tcx.types.err;
                        }
                    };

                    // We've encountered an `AnonConst` in some path, so we need to
                    // figure out which generic parameter it corresponds to and return
                    // the relevant type.

                    let (arg_index, segment) = path
                        .segments
                        .iter()
                        .filter_map(|seg| seg.args.as_ref().map(|args| (args.args, seg)))
                        .find_map(|(args, seg)| {
                            args.iter()
                                .filter(|arg| arg.is_const())
                                .enumerate()
                                .filter(|(_, arg)| arg.id() == hir_id)
                                .map(|(index, _)| (index, seg))
                                .next()
                        })
                        .unwrap_or_else(|| {
                            bug!("no arg matching AnonConst in path");
                        });

                    // Try to use the segment resolution if it is valid, otherwise we
                    // default to the path resolution.
                    let res = segment.res.filter(|&r| r != Res::Err).unwrap_or(path.res);
                    let generics = match res {
                        Res::Def(DefKind::Ctor(..), def_id) => {
                            tcx.generics_of(tcx.parent(def_id).unwrap())
                        }
                        Res::Def(_, def_id) => tcx.generics_of(def_id),
                        res => {
                            tcx.sess.delay_span_bug(
                                DUMMY_SP,
                                &format!(
                                    "unexpected anon const res {:?} in path: {:?}",
                                    res, path,
                                ),
                            );
                            return tcx.types.err;
                        }
                    };

                    let ty = generics
                        .params
                        .iter()
                        .filter(|param| {
                            if let ty::GenericParamDefKind::Const = param.kind {
                                true
                            } else {
                                false
                            }
                        })
                        .nth(arg_index)
                        .map(|param| tcx.type_of(param.def_id));

                    if let Some(ty) = ty {
                        ty
                    } else {
                        // This is no generic parameter associated with the arg. This is
                        // probably from an extra arg where one is not needed.
                        tcx.sess.delay_span_bug(
                            DUMMY_SP,
                            &format!(
                                "missing generic parameter for `AnonConst`, parent: {:?}, res: {:?}",
                                parent_node, res
                            ),
                        );
                        tcx.types.err
                    }
                }

                x => {
                    tcx.sess.delay_span_bug(
                        DUMMY_SP,
                        &format!("unexpected const parent in type_of_def_id(): {:?}", x),
                    );
                    tcx.types.err
                }
            }
        }

        Node::GenericParam(param) => match &param.kind {
            GenericParamKind::Type { default: Some(ref ty), .. } => icx.to_ty(ty),
            GenericParamKind::Const { ty: ref hir_ty, .. } => {
                let ty = icx.to_ty(hir_ty);
                if !tcx.features().const_compare_raw_pointers {
                    let err = match ty.peel_refs().kind {
                        ty::FnPtr(_) => Some("function pointers"),
                        ty::RawPtr(_) => Some("raw pointers"),
                        _ => None,
                    };
                    if let Some(unsupported_type) = err {
                        feature_err(
                            &tcx.sess.parse_sess,
                            sym::const_compare_raw_pointers,
                            hir_ty.span,
                            &format!(
                                "using {} as const generic parameters is unstable",
                                unsupported_type
                            ),
                        )
                        .emit();
                    };
                }
                if traits::search_for_structural_match_violation(param.hir_id, param.span, tcx, ty)
                    .is_some()
                {
                    struct_span_err!(
                        tcx.sess,
                        hir_ty.span,
                        E0741,
                        "the types of const generic parameters must derive `PartialEq` and `Eq`",
                    )
                    .span_label(
                        hir_ty.span,
                        format!("`{}` doesn't derive both `PartialEq` and `Eq`", ty),
                    )
                    .emit();
                }
                ty
            }
            x => bug!("unexpected non-type Node::GenericParam: {:?}", x),
        },

        x => {
            bug!("unexpected sort of node in type_of_def_id(): {:?}", x);
        }
    }
}

fn find_opaque_ty_constraints(tcx: TyCtxt<'_>, def_id: DefId) -> Ty<'_> {
    use rustc_hir::{Expr, ImplItem, Item, TraitItem};

    debug!("find_opaque_ty_constraints({:?})", def_id);

    struct ConstraintLocator<'tcx> {
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        // (first found type span, actual type)
        found: Option<(Span, Ty<'tcx>)>,
    }

    impl ConstraintLocator<'_> {
        fn check(&mut self, def_id: DefId) {
            // Don't try to check items that cannot possibly constrain the type.
            if !self.tcx.has_typeck_tables(def_id) {
                debug!(
                    "find_opaque_ty_constraints: no constraint for `{:?}` at `{:?}`: no tables",
                    self.def_id, def_id,
                );
                return;
            }
            // Calling `mir_borrowck` can lead to cycle errors through
            // const-checking, avoid calling it if we don't have to.
            if !self.tcx.typeck_tables_of(def_id).concrete_opaque_types.contains_key(&self.def_id) {
                debug!(
                    "find_opaque_ty_constraints: no constraint for `{:?}` at `{:?}`",
                    self.def_id, def_id,
                );
                return;
            }
            // Use borrowck to get the type with unerased regions.
            let ty = self.tcx.mir_borrowck(def_id).concrete_opaque_types.get(&self.def_id);
            if let Some(ty::ResolvedOpaqueTy { concrete_type, substs }) = ty {
                debug!(
                    "find_opaque_ty_constraints: found constraint for `{:?}` at `{:?}`: {:?}",
                    self.def_id, def_id, ty,
                );

                // FIXME(oli-obk): trace the actual span from inference to improve errors.
                let span = self.tcx.def_span(def_id);

                // HACK(eddyb) this check shouldn't be needed, as `wfcheck`
                // performs the same checks, in theory, but I've kept it here
                // using `delay_span_bug`, just in case `wfcheck` slips up.
                let opaque_generics = self.tcx.generics_of(self.def_id);
                let mut used_params: FxHashSet<_> = FxHashSet::default();
                for (i, arg) in substs.iter().enumerate() {
                    let arg_is_param = match arg.unpack() {
                        GenericArgKind::Type(ty) => matches!(ty.kind, ty::Param(_)),
                        GenericArgKind::Lifetime(lt) => {
                            matches!(lt, ty::ReEarlyBound(_) | ty::ReFree(_))
                        }
                        GenericArgKind::Const(ct) => matches!(ct.val, ty::ConstKind::Param(_)),
                    };

                    if arg_is_param {
                        if !used_params.insert(arg) {
                            // There was already an entry for `arg`, meaning a generic parameter
                            // was used twice.
                            self.tcx.sess.delay_span_bug(
                                span,
                                &format!(
                                    "defining opaque type use restricts opaque \
                                     type by using the generic parameter `{}` twice",
                                    arg,
                                ),
                            );
                        }
                    } else {
                        let param = opaque_generics.param_at(i, self.tcx);
                        self.tcx.sess.delay_span_bug(
                            span,
                            &format!(
                                "defining opaque type use does not fully define opaque type: \
                                 generic parameter `{}` is specified as concrete {} `{}`",
                                param.name,
                                param.kind.descr(),
                                arg,
                            ),
                        );
                    }
                }

                if let Some((prev_span, prev_ty)) = self.found {
                    if *concrete_type != prev_ty {
                        debug!("find_opaque_ty_constraints: span={:?}", span);
                        // Found different concrete types for the opaque type.
                        let mut err = self.tcx.sess.struct_span_err(
                            span,
                            "concrete type differs from previous defining opaque type use",
                        );
                        err.span_label(
                            span,
                            format!("expected `{}`, got `{}`", prev_ty, concrete_type),
                        );
                        err.span_note(prev_span, "previous use here");
                        err.emit();
                    }
                } else {
                    self.found = Some((span, concrete_type));
                }
            } else {
                debug!(
                    "find_opaque_ty_constraints: no constraint for `{:?}` at `{:?}`",
                    self.def_id, def_id,
                );
            }
        }
    }

    impl<'tcx> intravisit::Visitor<'tcx> for ConstraintLocator<'tcx> {
        type Map = Map<'tcx>;

        fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
            intravisit::NestedVisitorMap::All(self.tcx.hir())
        }
        fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
            if let hir::ExprKind::Closure(..) = ex.kind {
                let def_id = self.tcx.hir().local_def_id(ex.hir_id);
                self.check(def_id);
            }
            intravisit::walk_expr(self, ex);
        }
        fn visit_item(&mut self, it: &'tcx Item<'tcx>) {
            debug!("find_existential_constraints: visiting {:?}", it);
            let def_id = self.tcx.hir().local_def_id(it.hir_id);
            // The opaque type itself or its children are not within its reveal scope.
            if def_id != self.def_id {
                self.check(def_id);
                intravisit::walk_item(self, it);
            }
        }
        fn visit_impl_item(&mut self, it: &'tcx ImplItem<'tcx>) {
            debug!("find_existential_constraints: visiting {:?}", it);
            let def_id = self.tcx.hir().local_def_id(it.hir_id);
            // The opaque type itself or its children are not within its reveal scope.
            if def_id != self.def_id {
                self.check(def_id);
                intravisit::walk_impl_item(self, it);
            }
        }
        fn visit_trait_item(&mut self, it: &'tcx TraitItem<'tcx>) {
            debug!("find_existential_constraints: visiting {:?}", it);
            let def_id = self.tcx.hir().local_def_id(it.hir_id);
            self.check(def_id);
            intravisit::walk_trait_item(self, it);
        }
    }

    let hir_id = tcx.hir().as_local_hir_id(def_id).unwrap();
    let scope = tcx.hir().get_defining_scope(hir_id);
    let mut locator = ConstraintLocator { def_id, tcx, found: None };

    debug!("find_opaque_ty_constraints: scope={:?}", scope);

    if scope == hir::CRATE_HIR_ID {
        intravisit::walk_crate(&mut locator, tcx.hir().krate());
    } else {
        debug!("find_opaque_ty_constraints: scope={:?}", tcx.hir().get(scope));
        match tcx.hir().get(scope) {
            // We explicitly call `visit_*` methods, instead of using `intravisit::walk_*` methods
            // This allows our visitor to process the defining item itself, causing
            // it to pick up any 'sibling' defining uses.
            //
            // For example, this code:
            // ```
            // fn foo() {
            //     type Blah = impl Debug;
            //     let my_closure = || -> Blah { true };
            // }
            // ```
            //
            // requires us to explicitly process `foo()` in order
            // to notice the defining usage of `Blah`.
            Node::Item(ref it) => locator.visit_item(it),
            Node::ImplItem(ref it) => locator.visit_impl_item(it),
            Node::TraitItem(ref it) => locator.visit_trait_item(it),
            other => bug!("{:?} is not a valid scope for an opaque type item", other),
        }
    }

    match locator.found {
        Some((_, ty)) => ty,
        None => {
            let span = tcx.def_span(def_id);
            tcx.sess.span_err(span, "could not find defining uses");
            tcx.types.err
        }
    }
}

fn infer_placeholder_type(
    tcx: TyCtxt<'_>,
    def_id: DefId,
    body_id: hir::BodyId,
    span: Span,
    item_ident: Ident,
) -> Ty<'_> {
    let ty = tcx.diagnostic_only_typeck_tables_of(def_id).node_type(body_id.hir_id);

    // If this came from a free `const` or `static mut?` item,
    // then the user may have written e.g. `const A = 42;`.
    // In this case, the parser has stashed a diagnostic for
    // us to improve in typeck so we do that now.
    match tcx.sess.diagnostic().steal_diagnostic(span, StashKey::ItemNoType) {
        Some(mut err) => {
            // The parser provided a sub-optimal `HasPlaceholders` suggestion for the type.
            // We are typeck and have the real type, so remove that and suggest the actual type.
            err.suggestions.clear();
            err.span_suggestion(
                span,
                "provide a type for the item",
                format!("{}: {}", item_ident, ty),
                Applicability::MachineApplicable,
            )
            .emit();
        }
        None => {
            let mut diag = bad_placeholder_type(tcx, vec![span]);
            if ty != tcx.types.err {
                diag.span_suggestion(
                    span,
                    "replace `_` with the correct type",
                    ty.to_string(),
                    Applicability::MaybeIncorrect,
                );
            }
            diag.emit();
        }
    }

    // Typeck doesn't expect erased regions to be returned from `type_of`.
    tcx.fold_regions(&ty, &mut false, |r, _| match r {
        ty::ReErased => tcx.lifetimes.re_static,
        _ => r,
    })
}

fn report_assoc_ty_on_inherent_impl(tcx: TyCtxt<'_>, span: Span) {
    struct_span_err!(
        tcx.sess,
        span,
        E0202,
        "associated types are not yet supported in inherent impls (see #8995)"
    )
    .emit();
}
