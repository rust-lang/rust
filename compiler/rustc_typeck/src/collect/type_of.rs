use crate::errors::AssocTypeOnInherentImpl;
use rustc_data_structures::fx::FxHashSet;
use rustc_errors::{Applicability, ErrorReported, StashKey};
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{HirId, Node};
use rustc_middle::hir::map::Map;
use rustc_middle::ty::subst::{GenericArgKind, InternalSubsts};
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::{self, DefIdTree, Ty, TyCtxt, TypeFoldable};
use rustc_span::symbol::Ident;
use rustc_span::{Span, DUMMY_SP};

use super::ItemCtxt;
use super::{bad_placeholder_type, is_suggestable_infer_ty};

/// Computes the relevant generic parameter for a potential generic const argument.
///
/// This should be called using the query `tcx.opt_const_param_of`.
pub(super) fn opt_const_param_of(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<DefId> {
    use hir::*;
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

    if let Node::AnonConst(_) = tcx.hir().get(hir_id) {
        let parent_node_id = tcx.hir().get_parent_node(hir_id);
        let parent_node = tcx.hir().get(parent_node_id);

        match parent_node {
            Node::Expr(&Expr {
                kind:
                    ExprKind::MethodCall(segment, ..) | ExprKind::Path(QPath::TypeRelative(_, segment)),
                ..
            }) => {
                let body_owner = tcx.hir().local_def_id(tcx.hir().enclosing_body_owner(hir_id));
                let tables = tcx.typeck(body_owner);
                // This may fail in case the method/path does not actually exist.
                // As there is no relevant param for `def_id`, we simply return
                // `None` here.
                let type_dependent_def = tables.type_dependent_def_id(parent_node_id)?;
                let idx = segment
                    .args
                    .and_then(|args| {
                        args.args
                            .iter()
                            .filter(|arg| arg.is_const())
                            .position(|arg| arg.id() == hir_id)
                    })
                    .unwrap_or_else(|| {
                        bug!("no arg matching AnonConst in segment");
                    });

                tcx.generics_of(type_dependent_def)
                    .params
                    .iter()
                    .filter(|param| matches!(param.kind, ty::GenericParamDefKind::Const))
                    .nth(idx)
                    .map(|param| param.def_id)
            }

            Node::Ty(&Ty { kind: TyKind::Path(_), .. })
            | Node::Expr(&Expr { kind: ExprKind::Path(_) | ExprKind::Struct(..), .. })
            | Node::TraitRef(..)
            | Node::Pat(_) => {
                let path = match parent_node {
                    Node::Ty(&Ty { kind: TyKind::Path(QPath::Resolved(_, path)), .. })
                    | Node::TraitRef(&TraitRef { path, .. }) => &*path,
                    Node::Expr(&Expr {
                        kind:
                            ExprKind::Path(QPath::Resolved(_, path))
                            | ExprKind::Struct(&QPath::Resolved(_, path), ..),
                        ..
                    }) => {
                        let body_owner =
                            tcx.hir().local_def_id(tcx.hir().enclosing_body_owner(hir_id));
                        let _tables = tcx.typeck(body_owner);
                        &*path
                    }
                    Node::Pat(pat) => {
                        if let Some(path) = get_path_containing_arg_in_pat(pat, hir_id) {
                            path
                        } else {
                            tcx.sess.delay_span_bug(
                                tcx.def_span(def_id),
                                &format!(
                                    "unable to find const parent for {} in pat {:?}",
                                    hir_id, pat
                                ),
                            );
                            return None;
                        }
                    }
                    _ => {
                        tcx.sess.delay_span_bug(
                            tcx.def_span(def_id),
                            &format!("unexpected const parent path {:?}", parent_node),
                        );
                        return None;
                    }
                };

                // We've encountered an `AnonConst` in some path, so we need to
                // figure out which generic parameter it corresponds to and return
                // the relevant type.
                let (arg_index, segment) = path
                    .segments
                    .iter()
                    .filter_map(|seg| seg.args.map(|args| (args.args, seg)))
                    .find_map(|(args, seg)| {
                        args.iter()
                            .filter(|arg| arg.is_const())
                            .position(|arg| arg.id() == hir_id)
                            .map(|index| (index, seg))
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
                    Res::Err => {
                        tcx.sess.delay_span_bug(tcx.def_span(def_id), "anon const with Res::Err");
                        return None;
                    }
                    _ => {
                        // If the user tries to specify generics on a type that does not take them,
                        // e.g. `usize<T>`, we may hit this branch, in which case we treat it as if
                        // no arguments have been passed. An error should already have been emitted.
                        tcx.sess.delay_span_bug(
                            tcx.def_span(def_id),
                            &format!("unexpected anon const res {:?} in path: {:?}", res, path),
                        );
                        return None;
                    }
                };

                generics
                    .params
                    .iter()
                    .filter(|param| matches!(param.kind, ty::GenericParamDefKind::Const))
                    .nth(arg_index)
                    .map(|param| param.def_id)
            }
            _ => None,
        }
    } else {
        None
    }
}

fn get_path_containing_arg_in_pat<'hir>(
    pat: &'hir hir::Pat<'hir>,
    arg_id: HirId,
) -> Option<&'hir hir::Path<'hir>> {
    use hir::*;

    let is_arg_in_path = |p: &hir::Path<'_>| {
        p.segments
            .iter()
            .filter_map(|seg| seg.args)
            .flat_map(|args| args.args)
            .any(|arg| arg.id() == arg_id)
    };
    let mut arg_path = None;
    pat.walk(|pat| match pat.kind {
        PatKind::Struct(QPath::Resolved(_, path), _, _)
        | PatKind::TupleStruct(QPath::Resolved(_, path), _, _)
        | PatKind::Path(QPath::Resolved(_, path))
            if is_arg_in_path(path) =>
        {
            arg_path = Some(path);
            false
        }
        _ => true,
    });
    arg_path
}

pub(super) fn type_of(tcx: TyCtxt<'_>, def_id: DefId) -> Ty<'_> {
    let def_id = def_id.expect_local();
    use rustc_hir::*;

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

    let icx = ItemCtxt::new(tcx, def_id.to_def_id());

    match tcx.hir().get(hir_id) {
        Node::TraitItem(item) => match item.kind {
            TraitItemKind::Fn(..) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                tcx.mk_fn_def(def_id.to_def_id(), substs)
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
                let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                tcx.mk_fn_def(def_id.to_def_id(), substs)
            }
            ImplItemKind::Const(ref ty, body_id) => {
                if is_suggestable_infer_ty(ty) {
                    infer_placeholder_type(tcx, def_id, body_id, ty.span, item.ident)
                } else {
                    icx.to_ty(ty)
                }
            }
            ImplItemKind::TyAlias(ref ty) => {
                if tcx.impl_trait_ref(tcx.hir().get_parent_did(hir_id).to_def_id()).is_none() {
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
                    let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                    tcx.mk_fn_def(def_id.to_def_id(), substs)
                }
                ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Union(..) => {
                    let def = tcx.adt_def(def_id);
                    let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                    tcx.mk_adt(def, substs)
                }
                ItemKind::OpaqueTy(OpaqueTy { origin: hir::OpaqueTyOrigin::Binding, .. }) => {
                    let_position_impl_trait_type(tcx, def_id)
                }
                ItemKind::OpaqueTy(OpaqueTy { impl_trait_fn: None, .. }) => {
                    find_opaque_ty_constraints(tcx, def_id)
                }
                // Opaque types desugared from `impl Trait`.
                ItemKind::OpaqueTy(OpaqueTy { impl_trait_fn: Some(owner), .. }) => {
                    let concrete_ty = tcx
                        .mir_borrowck(owner.expect_local())
                        .concrete_opaque_types
                        .get(&def_id.to_def_id())
                        .map(|opaque| opaque.concrete_type)
                        .unwrap_or_else(|| {
                            tcx.sess.delay_span_bug(
                                DUMMY_SP,
                                &format!(
                                    "owner {:?} has no opaque type for {:?} in its typeck results",
                                    owner, def_id,
                                ),
                            );
                            if let Some(ErrorReported) =
                                tcx.typeck(owner.expect_local()).tainted_by_errors
                            {
                                // Some error in the
                                // owner fn prevented us from populating
                                // the `concrete_opaque_types` table.
                                tcx.ty_error()
                            } else {
                                // We failed to resolve the opaque type or it
                                // resolves to itself. Return the non-revealed
                                // type, which should result in E0720.
                                tcx.mk_opaque(
                                    def_id.to_def_id(),
                                    InternalSubsts::identity_for_item(tcx, def_id.to_def_id()),
                                )
                            }
                        });
                    debug!("concrete_ty = {:?}", concrete_ty);
                    concrete_ty
                }
                ItemKind::Trait(..)
                | ItemKind::TraitAlias(..)
                | ItemKind::Mod(..)
                | ItemKind::ForeignMod { .. }
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
                let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                tcx.mk_fn_def(def_id.to_def_id(), substs)
            }
            ForeignItemKind::Static(ref t, _) => icx.to_ty(t),
            ForeignItemKind::Type => tcx.mk_foreign(def_id.to_def_id()),
        },

        Node::Ctor(&ref def) | Node::Variant(Variant { data: ref def, .. }) => match *def {
            VariantData::Unit(..) | VariantData::Struct(..) => {
                tcx.type_of(tcx.hir().get_parent_did(hir_id).to_def_id())
            }
            VariantData::Tuple(..) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                tcx.mk_fn_def(def_id.to_def_id(), substs)
            }
        },

        Node::Field(field) => icx.to_ty(&field.ty),

        Node::Expr(&Expr { kind: ExprKind::Closure(.., gen), .. }) => {
            let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
            if let Some(movability) = gen {
                tcx.mk_generator(def_id.to_def_id(), substs, movability)
            } else {
                tcx.mk_closure(def_id.to_def_id(), substs)
            }
        }

        Node::AnonConst(_) => {
            if let Some(param) = tcx.opt_const_param_of(def_id) {
                // We defer to `type_of` of the corresponding parameter
                // for generic arguments.
                return tcx.type_of(param);
            }

            let parent_node = tcx.hir().get(tcx.hir().get_parent_node(hir_id));
            match parent_node {
                Node::Ty(&Ty { kind: TyKind::Array(_, ref constant), .. })
                | Node::Ty(&Ty { kind: TyKind::Typeof(ref constant), .. })
                | Node::Expr(&Expr { kind: ExprKind::Repeat(_, ref constant), .. })
                    if constant.hir_id == hir_id =>
                {
                    tcx.types.usize
                }

                Node::Expr(&Expr { kind: ExprKind::ConstBlock(ref anon_const), .. })
                    if anon_const.hir_id == hir_id =>
                {
                    tcx.typeck(def_id).node_type(anon_const.hir_id)
                }

                Node::Variant(Variant { disr_expr: Some(ref e), .. }) if e.hir_id == hir_id => tcx
                    .adt_def(tcx.hir().get_parent_did(hir_id).to_def_id())
                    .repr
                    .discr_type()
                    .to_ty(tcx),

                x => tcx.ty_error_with_message(
                    DUMMY_SP,
                    &format!("unexpected const parent in type_of_def_id(): {:?}", x),
                ),
            }
        }

        Node::GenericParam(param) => match &param.kind {
            GenericParamKind::Type { default: Some(ty), .. }
            | GenericParamKind::Const { ty, .. } => icx.to_ty(ty),
            x => bug!("unexpected non-type Node::GenericParam: {:?}", x),
        },

        x => {
            bug!("unexpected sort of node in type_of_def_id(): {:?}", x);
        }
    }
}

fn find_opaque_ty_constraints(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Ty<'_> {
    use rustc_hir::{Expr, ImplItem, Item, TraitItem};

    debug!("find_opaque_ty_constraints({:?})", def_id);

    struct ConstraintLocator<'tcx> {
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        // (first found type span, actual type)
        found: Option<(Span, Ty<'tcx>)>,
    }

    impl ConstraintLocator<'_> {
        fn check(&mut self, def_id: LocalDefId) {
            // Don't try to check items that cannot possibly constrain the type.
            if !self.tcx.has_typeck_results(def_id) {
                debug!(
                    "find_opaque_ty_constraints: no constraint for `{:?}` at `{:?}`: no typeck results",
                    self.def_id, def_id,
                );
                return;
            }
            // Calling `mir_borrowck` can lead to cycle errors through
            // const-checking, avoid calling it if we don't have to.
            if !self.tcx.typeck(def_id).concrete_opaque_types.contains_key(&self.def_id) {
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
                        GenericArgKind::Type(ty) => matches!(ty.kind(), ty::Param(_)),
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
            if def_id.to_def_id() != self.def_id {
                self.check(def_id);
                intravisit::walk_item(self, it);
            }
        }
        fn visit_impl_item(&mut self, it: &'tcx ImplItem<'tcx>) {
            debug!("find_existential_constraints: visiting {:?}", it);
            let def_id = self.tcx.hir().local_def_id(it.hir_id);
            // The opaque type itself or its children are not within its reveal scope.
            if def_id.to_def_id() != self.def_id {
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

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let scope = tcx.hir().get_defining_scope(hir_id);
    let mut locator = ConstraintLocator { def_id: def_id.to_def_id(), tcx, found: None };

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
            tcx.ty_error()
        }
    }
}

/// Retrieve the inferred concrete type for let position impl trait.
///
/// This is different to other kinds of impl trait because:
///
/// 1. We know which function contains the defining use (the function that
///    contains the let statement)
/// 2. We do not currently allow (free) lifetimes in the return type. `let`
///    statements in some statically unreachable code are removed from the MIR
///    by the time we borrow check, and it's not clear how we should handle
///    those.
fn let_position_impl_trait_type(tcx: TyCtxt<'_>, opaque_ty_id: LocalDefId) -> Ty<'_> {
    let scope = tcx.hir().get_defining_scope(tcx.hir().local_def_id_to_hir_id(opaque_ty_id));
    let scope_def_id = tcx.hir().local_def_id(scope);

    let opaque_ty_def_id = opaque_ty_id.to_def_id();

    let owner_typeck_results = tcx.typeck(scope_def_id);
    let concrete_ty = owner_typeck_results
        .concrete_opaque_types
        .get(&opaque_ty_def_id)
        .map(|opaque| opaque.concrete_type)
        .unwrap_or_else(|| {
            tcx.sess.delay_span_bug(
                DUMMY_SP,
                &format!(
                    "owner {:?} has no opaque type for {:?} in its typeck results",
                    scope_def_id, opaque_ty_id
                ),
            );
            if let Some(ErrorReported) = owner_typeck_results.tainted_by_errors {
                // Some error in the owner fn prevented us from populating the
                // `concrete_opaque_types` table.
                tcx.ty_error()
            } else {
                // We failed to resolve the opaque type or it resolves to
                // itself. Return the non-revealed type, which should result in
                // E0720.
                tcx.mk_opaque(
                    opaque_ty_def_id,
                    InternalSubsts::identity_for_item(tcx, opaque_ty_def_id),
                )
            }
        });
    debug!("concrete_ty = {:?}", concrete_ty);
    if concrete_ty.has_erased_regions() {
        // FIXME(impl_trait_in_bindings) Handle this case.
        tcx.sess.span_fatal(
            tcx.hir().span(tcx.hir().local_def_id_to_hir_id(opaque_ty_id)),
            "lifetimes in impl Trait types in bindings are not currently supported",
        );
    }
    concrete_ty
}

fn infer_placeholder_type(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    body_id: hir::BodyId,
    span: Span,
    item_ident: Ident,
) -> Ty<'_> {
    let ty = tcx.diagnostic_only_typeck(def_id).node_type(body_id.hir_id);

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
            if !matches!(ty.kind(), ty::Error(_)) {
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
    tcx.fold_regions(ty, &mut false, |r, _| match r {
        ty::ReErased => tcx.lifetimes.re_static,
        _ => r,
    })
}

fn report_assoc_ty_on_inherent_impl(tcx: TyCtxt<'_>, span: Span) {
    tcx.sess.emit_err(AssocTypeOnInherentImpl { span });
}
