use rustc_errors::{Applicability, StashKey};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit;
use rustc_hir::intravisit::Visitor;
use rustc_hir::{HirId, Node};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_middle::ty::subst::InternalSubsts;
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::{
    self, ImplTraitInTraitData, IsSuggestable, Ty, TyCtxt, TypeFolder, TypeSuperFoldable,
    TypeVisitableExt,
};
use rustc_span::symbol::Ident;
use rustc_span::{Span, DUMMY_SP};

use super::ItemCtxt;
use super::{bad_placeholder, is_suggestable_infer_ty};
use crate::errors::UnconstrainedOpaqueType;

/// Computes the relevant generic parameter for a potential generic const argument.
///
/// This should be called using the query `tcx.opt_const_param_of`.
pub(super) fn opt_const_param_of(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Option<DefId> {
    use hir::*;
    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

    match tcx.hir().get(hir_id) {
        Node::AnonConst(_) => (),
        _ => return None,
    };

    let parent_node_id = tcx.hir().parent_id(hir_id);
    let parent_node = tcx.hir().get(parent_node_id);

    let (generics, arg_idx) = match parent_node {
        // This match arm is for when the def_id appears in a GAT whose
        // path can't be resolved without typechecking e.g.
        //
        // trait Foo {
        //   type Assoc<const N: usize>;
        //   fn foo() -> Self::Assoc<3>;
        // }
        //
        // In the above code we would call this query with the def_id of 3 and
        // the parent_node we match on would be the hir node for Self::Assoc<3>
        //
        // `Self::Assoc<3>` cant be resolved without typechecking here as we
        // didnt write <Self as Foo>::Assoc<3>. If we did then another match
        // arm would handle this.
        //
        // I believe this match arm is only needed for GAT but I am not 100% sure - BoxyUwU
        Node::Ty(hir_ty @ Ty { kind: TyKind::Path(QPath::TypeRelative(_, segment)), .. }) => {
            // Find the Item containing the associated type so we can create an ItemCtxt.
            // Using the ItemCtxt convert the HIR for the unresolved assoc type into a
            // ty which is a fully resolved projection.
            // For the code example above, this would mean converting Self::Assoc<3>
            // into a ty::Alias(ty::Projection, <Self as Foo>::Assoc<3>)
            let item_def_id = tcx
                .hir()
                .parent_owner_iter(hir_id)
                .find(|(_, node)| matches!(node, OwnerNode::Item(_)))
                .unwrap()
                .0
                .to_def_id();
            let item_ctxt = &ItemCtxt::new(tcx, item_def_id) as &dyn crate::astconv::AstConv<'_>;
            let ty = item_ctxt.ast_ty_to_ty(hir_ty);

            // Iterate through the generics of the projection to find the one that corresponds to
            // the def_id that this query was called with. We filter to only type and const args here
            // as a precaution for if it's ever allowed to elide lifetimes in GAT's. It currently isn't
            // but it can't hurt to be safe ^^
            if let ty::Alias(ty::Projection, projection) = ty.kind() {
                let generics = tcx.generics_of(projection.def_id);

                let arg_index = segment
                    .args
                    .and_then(|args| {
                        args.args
                            .iter()
                            .filter(|arg| arg.is_ty_or_const())
                            .position(|arg| arg.hir_id() == hir_id)
                    })
                    .unwrap_or_else(|| {
                        bug!("no arg matching AnonConst in segment");
                    });

                (generics, arg_index)
            } else {
                // I dont think it's possible to reach this but I'm not 100% sure - BoxyUwU
                tcx.sess.delay_span_bug(
                    tcx.def_span(def_id),
                    "unexpected non-GAT usage of an anon const",
                );
                return None;
            }
        }
        Node::Expr(&Expr {
            kind:
                ExprKind::MethodCall(segment, ..) | ExprKind::Path(QPath::TypeRelative(_, segment)),
            ..
        }) => {
            let body_owner = tcx.hir().enclosing_body_owner(hir_id);
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
                        .filter(|arg| arg.is_ty_or_const())
                        .position(|arg| arg.hir_id() == hir_id)
                })
                .unwrap_or_else(|| {
                    bug!("no arg matching AnonConst in segment");
                });

            (tcx.generics_of(type_dependent_def), idx)
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
                    let body_owner = tcx.hir().enclosing_body_owner(hir_id);
                    let _tables = tcx.typeck(body_owner);
                    &*path
                }
                Node::Pat(pat) => {
                    if let Some(path) = get_path_containing_arg_in_pat(pat, hir_id) {
                        path
                    } else {
                        tcx.sess.delay_span_bug(
                            tcx.def_span(def_id),
                            &format!("unable to find const parent for {} in pat {:?}", hir_id, pat),
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
            let Some((arg_index, segment)) = path.segments.iter().find_map(|seg| {
                let args = seg.args?;
                args.args
                .iter()
                .filter(|arg| arg.is_ty_or_const())
                .position(|arg| arg.hir_id() == hir_id)
                .map(|index| (index, seg)).or_else(|| args.bindings
                    .iter()
                    .filter_map(TypeBinding::opt_const)
                    .position(|ct| ct.hir_id == hir_id)
                    .map(|idx| (idx, seg)))
            }) else {
                tcx.sess.delay_span_bug(
                    tcx.def_span(def_id),
                    "no arg matching AnonConst in path",
                );
                return None;
            };

            let generics = match tcx.res_generics_def_id(segment.res) {
                Some(def_id) => tcx.generics_of(def_id),
                None => {
                    tcx.sess.delay_span_bug(
                        tcx.def_span(def_id),
                        &format!("unexpected anon const res {:?} in path: {:?}", segment.res, path),
                    );
                    return None;
                }
            };

            (generics, arg_index)
        }
        _ => return None,
    };

    debug!(?parent_node);
    debug!(?generics, ?arg_idx);
    generics
        .params
        .iter()
        .filter(|param| param.kind.is_ty_or_const())
        .nth(match generics.has_self && generics.parent.is_none() {
            true => arg_idx + 1,
            false => arg_idx,
        })
        .and_then(|param| match param.kind {
            ty::GenericParamDefKind::Const { .. } => {
                debug!(?param);
                Some(param.def_id)
            }
            _ => None,
        })
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
            .any(|arg| arg.hir_id() == arg_id)
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

pub(super) fn type_of(tcx: TyCtxt<'_>, def_id: DefId) -> ty::EarlyBinder<Ty<'_>> {
    // If we are computing `type_of` the synthesized associated type for an RPITIT in the impl
    // side, use `collect_return_position_impl_trait_in_trait_tys` to infer the value of the
    // associated type in the impl.
    if let Some(ImplTraitInTraitData::Impl { fn_def_id, .. }) = tcx.opt_rpitit_info(def_id) {
        match tcx.collect_return_position_impl_trait_in_trait_tys(fn_def_id) {
            Ok(map) => {
                let assoc_item = tcx.associated_item(def_id);
                return ty::EarlyBinder(map[&assoc_item.trait_item_def_id.unwrap()]);
            }
            Err(_) => {
                return ty::EarlyBinder(tcx.ty_error_with_message(
                    DUMMY_SP,
                    "Could not collect return position impl trait in trait tys",
                ));
            }
        }
    }

    let def_id = def_id.expect_local();
    use rustc_hir::*;

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);

    let icx = ItemCtxt::new(tcx, def_id.to_def_id());

    let output = match tcx.hir().get(hir_id) {
        Node::TraitItem(item) => match item.kind {
            TraitItemKind::Fn(..) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                tcx.mk_fn_def(def_id.to_def_id(), substs)
            }
            TraitItemKind::Const(ty, body_id) => body_id
                .and_then(|body_id| {
                    is_suggestable_infer_ty(ty).then(|| {
                        infer_placeholder_type(
                            tcx, def_id, body_id, ty.span, item.ident, "constant",
                        )
                    })
                })
                .unwrap_or_else(|| icx.to_ty(ty)),
            TraitItemKind::Type(_, Some(ty)) => icx.to_ty(ty),
            TraitItemKind::Type(_, None) => {
                span_bug!(item.span, "associated type missing default");
            }
        },

        Node::ImplItem(item) => match item.kind {
            ImplItemKind::Fn(..) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                tcx.mk_fn_def(def_id.to_def_id(), substs)
            }
            ImplItemKind::Const(ty, body_id) => {
                if is_suggestable_infer_ty(ty) {
                    infer_placeholder_type(tcx, def_id, body_id, ty.span, item.ident, "constant")
                } else {
                    icx.to_ty(ty)
                }
            }
            ImplItemKind::Type(ty) => {
                if tcx.impl_trait_ref(tcx.hir().get_parent_item(hir_id)).is_none() {
                    check_feature_inherent_assoc_ty(tcx, item.span);
                }

                icx.to_ty(ty)
            }
        },

        Node::Item(item) => {
            match item.kind {
                ItemKind::Static(ty, .., body_id) => {
                    if is_suggestable_infer_ty(ty) {
                        infer_placeholder_type(
                            tcx,
                            def_id,
                            body_id,
                            ty.span,
                            item.ident,
                            "static variable",
                        )
                    } else {
                        icx.to_ty(ty)
                    }
                }
                ItemKind::Const(ty, body_id) => {
                    if is_suggestable_infer_ty(ty) {
                        infer_placeholder_type(
                            tcx, def_id, body_id, ty.span, item.ident, "constant",
                        )
                    } else {
                        icx.to_ty(ty)
                    }
                }
                ItemKind::TyAlias(self_ty, _) => icx.to_ty(self_ty),
                ItemKind::Impl(hir::Impl { self_ty, .. }) => match self_ty.find_self_aliases() {
                    spans if spans.len() > 0 => {
                        let guar = tcx.sess.emit_err(crate::errors::SelfInImplSelf {
                            span: spans.into(),
                            note: (),
                        });
                        tcx.ty_error(guar)
                    }
                    _ => icx.to_ty(*self_ty),
                },
                ItemKind::Fn(..) => {
                    let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                    tcx.mk_fn_def(def_id.to_def_id(), substs)
                }
                ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Union(..) => {
                    let def = tcx.adt_def(def_id);
                    let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                    tcx.mk_adt(def, substs)
                }
                ItemKind::OpaqueTy(OpaqueTy { origin: hir::OpaqueTyOrigin::TyAlias, .. }) => {
                    find_opaque_ty_constraints_for_tait(tcx, def_id)
                }
                // Opaque types desugared from `impl Trait`.
                ItemKind::OpaqueTy(OpaqueTy {
                    origin:
                        hir::OpaqueTyOrigin::FnReturn(owner) | hir::OpaqueTyOrigin::AsyncFn(owner),
                    in_trait,
                    ..
                }) => {
                    if in_trait && !tcx.impl_defaultness(owner).has_value() {
                        span_bug!(
                            tcx.def_span(def_id),
                            "tried to get type of this RPITIT with no definition"
                        );
                    }
                    find_opaque_ty_constraints_for_rpit(tcx, def_id, owner)
                }
                ItemKind::Trait(..)
                | ItemKind::TraitAlias(..)
                | ItemKind::Macro(..)
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
            ForeignItemKind::Static(t, _) => icx.to_ty(t),
            ForeignItemKind::Type => tcx.mk_foreign(def_id.to_def_id()),
        },

        Node::Ctor(def) | Node::Variant(Variant { data: def, .. }) => match def {
            VariantData::Unit(..) | VariantData::Struct(..) => {
                tcx.type_of(tcx.hir().get_parent_item(hir_id)).subst_identity()
            }
            VariantData::Tuple(..) => {
                let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                tcx.mk_fn_def(def_id.to_def_id(), substs)
            }
        },

        Node::Field(field) => icx.to_ty(field.ty),

        Node::Expr(&Expr { kind: ExprKind::Closure { .. }, .. }) => {
            tcx.typeck(def_id).node_type(hir_id)
        }

        Node::AnonConst(_) if let Some(param) = tcx.opt_const_param_of(def_id) => {
            // We defer to `type_of` of the corresponding parameter
            // for generic arguments.
            tcx.type_of(param).subst_identity()
        }

        Node::AnonConst(_) => {
            let parent_node = tcx.hir().get_parent(hir_id);
            match parent_node {
                Node::Ty(Ty { kind: TyKind::Array(_, constant), .. })
                | Node::Expr(Expr { kind: ExprKind::Repeat(_, constant), .. })
                    if constant.hir_id() == hir_id =>
                {
                    tcx.types.usize
                }
                Node::Ty(Ty { kind: TyKind::Typeof(e), .. }) if e.hir_id == hir_id => {
                    tcx.typeck(def_id).node_type(e.hir_id)
                }

                Node::Expr(Expr { kind: ExprKind::ConstBlock(anon_const), .. })
                    if anon_const.hir_id == hir_id =>
                {
                    let substs = InternalSubsts::identity_for_item(tcx, def_id.to_def_id());
                    substs.as_inline_const().ty()
                }

                Node::Expr(&Expr { kind: ExprKind::InlineAsm(asm), .. })
                | Node::Item(&Item { kind: ItemKind::GlobalAsm(asm), .. })
                    if asm.operands.iter().any(|(op, _op_sp)| match op {
                        hir::InlineAsmOperand::Const { anon_const }
                        | hir::InlineAsmOperand::SymFn { anon_const } => {
                            anon_const.hir_id == hir_id
                        }
                        _ => false,
                    }) =>
                {
                    tcx.typeck(def_id).node_type(hir_id)
                }

                Node::Variant(Variant { disr_expr: Some(e), .. }) if e.hir_id == hir_id => {
                    tcx.adt_def(tcx.hir().get_parent_item(hir_id)).repr().discr_type().to_ty(tcx)
                }

                Node::TypeBinding(TypeBinding {
                    hir_id: binding_id,
                    kind: TypeBindingKind::Equality { term: Term::Const(e) },
                    ident,
                    ..
                }) if let Node::TraitRef(trait_ref) = tcx.hir().get_parent(*binding_id)
                    && e.hir_id == hir_id =>
                {
                    let Some(trait_def_id) = trait_ref.trait_def_id() else {
                        return ty::EarlyBinder(tcx.ty_error_with_message(DUMMY_SP, "Could not find trait"));
                    };
                    let assoc_items = tcx.associated_items(trait_def_id);
                    let assoc_item = assoc_items.find_by_name_and_kind(
                        tcx,
                        *ident,
                        ty::AssocKind::Const,
                        def_id.to_def_id(),
                    );
                    if let Some(assoc_item) = assoc_item {
                        tcx.type_of(assoc_item.def_id)
                            .no_bound_vars()
                            .expect("const parameter types cannot be generic")
                    } else {
                        // FIXME(associated_const_equality): add a useful error message here.
                        tcx.ty_error_with_message(
                            DUMMY_SP,
                            "Could not find associated const on trait",
                        )
                    }
                }

                Node::TypeBinding(TypeBinding {
                    hir_id: binding_id,
                    gen_args,
                    kind,
                    ident,
                    ..
                }) if let Node::TraitRef(trait_ref) = tcx.hir().get_parent(*binding_id)
                    && let Some((idx, _)) =
                        gen_args.args.iter().enumerate().find(|(_, arg)| {
                            if let GenericArg::Const(ct) = arg {
                                ct.value.hir_id == hir_id
                            } else {
                                false
                            }
                        }) =>
                {
                    let Some(trait_def_id) = trait_ref.trait_def_id() else {
                        return ty::EarlyBinder(tcx.ty_error_with_message(DUMMY_SP, "Could not find trait"));
                    };
                    let assoc_items = tcx.associated_items(trait_def_id);
                    let assoc_item = assoc_items.find_by_name_and_kind(
                        tcx,
                        *ident,
                        match kind {
                            // I think `<A: T>` type bindings requires that `A` is a type
                            TypeBindingKind::Constraint { .. }
                            | TypeBindingKind::Equality { term: Term::Ty(..) } => {
                                ty::AssocKind::Type
                            }
                            TypeBindingKind::Equality { term: Term::Const(..) } => {
                                ty::AssocKind::Const
                            }
                        },
                        def_id.to_def_id(),
                    );
                    if let Some(assoc_item) = assoc_item
                        && let param = &tcx.generics_of(assoc_item.def_id).params[idx]
                        && matches!(param.kind, ty::GenericParamDefKind::Const { .. })
                    {
                        tcx.type_of(param.def_id)
                            .no_bound_vars()
                            .expect("const parameter types cannot be generic")
                    } else {
                        // FIXME(associated_const_equality): add a useful error message here.
                        tcx.ty_error_with_message(
                            DUMMY_SP,
                            "Could not find const param on associated item",
                        )
                    }
                }

                Node::GenericParam(&GenericParam {
                    def_id: param_def_id,
                    kind: GenericParamKind::Const { default: Some(ct), .. },
                    ..
                }) if ct.hir_id == hir_id => tcx.type_of(param_def_id).subst_identity(),

                x => tcx.ty_error_with_message(
                    DUMMY_SP,
                    &format!("unexpected const parent in type_of(): {x:?}"),
                ),
            }
        }

        Node::GenericParam(param) => match &param.kind {
            GenericParamKind::Type { default: Some(ty), .. }
            | GenericParamKind::Const { ty, .. } => icx.to_ty(ty),
            x => bug!("unexpected non-type Node::GenericParam: {:?}", x),
        },

        x => {
            bug!("unexpected sort of node in type_of(): {:?}", x);
        }
    };
    ty::EarlyBinder(output)
}

#[instrument(skip(tcx), level = "debug")]
/// Checks "defining uses" of opaque `impl Trait` types to ensure that they meet the restrictions
/// laid for "higher-order pattern unification".
/// This ensures that inference is tractable.
/// In particular, definitions of opaque types can only use other generics as arguments,
/// and they cannot repeat an argument. Example:
///
/// ```ignore (illustrative)
/// type Foo<A, B> = impl Bar<A, B>;
///
/// // Okay -- `Foo` is applied to two distinct, generic types.
/// fn a<T, U>() -> Foo<T, U> { .. }
///
/// // Not okay -- `Foo` is applied to `T` twice.
/// fn b<T>() -> Foo<T, T> { .. }
///
/// // Not okay -- `Foo` is applied to a non-generic type.
/// fn b<T>() -> Foo<T, u32> { .. }
/// ```
///
fn find_opaque_ty_constraints_for_tait(tcx: TyCtxt<'_>, def_id: LocalDefId) -> Ty<'_> {
    use rustc_hir::{Expr, ImplItem, Item, TraitItem};

    struct ConstraintLocator<'tcx> {
        tcx: TyCtxt<'tcx>,

        /// def_id of the opaque type whose defining uses are being checked
        def_id: LocalDefId,

        /// as we walk the defining uses, we are checking that all of them
        /// define the same hidden type. This variable is set to `Some`
        /// with the first type that we find, and then later types are
        /// checked against it (we also carry the span of that first
        /// type).
        found: Option<ty::OpaqueHiddenType<'tcx>>,

        /// In the presence of dead code, typeck may figure out a hidden type
        /// while borrowck will now. We collect these cases here and check at
        /// the end that we actually found a type that matches (modulo regions).
        typeck_types: Vec<ty::OpaqueHiddenType<'tcx>>,
    }

    impl ConstraintLocator<'_> {
        #[instrument(skip(self), level = "debug")]
        fn check(&mut self, item_def_id: LocalDefId) {
            // Don't try to check items that cannot possibly constrain the type.
            if !self.tcx.has_typeck_results(item_def_id) {
                debug!("no constraint: no typeck results");
                return;
            }
            // Calling `mir_borrowck` can lead to cycle errors through
            // const-checking, avoid calling it if we don't have to.
            // ```rust
            // type Foo = impl Fn() -> usize; // when computing type for this
            // const fn bar() -> Foo {
            //     || 0usize
            // }
            // const BAZR: Foo = bar(); // we would mir-borrowck this, causing cycles
            // // because we again need to reveal `Foo` so we can check whether the
            // // constant does not contain interior mutability.
            // ```
            let tables = self.tcx.typeck(item_def_id);
            if let Some(guar) = tables.tainted_by_errors {
                self.found =
                    Some(ty::OpaqueHiddenType { span: DUMMY_SP, ty: self.tcx.ty_error(guar) });
                return;
            }
            let Some(&typeck_hidden_ty) = tables.concrete_opaque_types.get(&self.def_id) else {
                debug!("no constraints in typeck results");
                return;
            };
            if self.typeck_types.iter().all(|prev| prev.ty != typeck_hidden_ty.ty) {
                self.typeck_types.push(typeck_hidden_ty);
            }

            // Use borrowck to get the type with unerased regions.
            let concrete_opaque_types = &self.tcx.mir_borrowck(item_def_id).concrete_opaque_types;
            debug!(?concrete_opaque_types);
            if let Some(&concrete_type) = concrete_opaque_types.get(&self.def_id) {
                debug!(?concrete_type, "found constraint");
                if let Some(prev) = &mut self.found {
                    if concrete_type.ty != prev.ty && !(concrete_type, prev.ty).references_error() {
                        let guar = prev.report_mismatch(&concrete_type, self.tcx);
                        prev.ty = self.tcx.ty_error(guar);
                    }
                } else {
                    self.found = Some(concrete_type);
                }
            }
        }
    }

    impl<'tcx> intravisit::Visitor<'tcx> for ConstraintLocator<'tcx> {
        type NestedFilter = nested_filter::All;

        fn nested_visit_map(&mut self) -> Self::Map {
            self.tcx.hir()
        }
        fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
            if let hir::ExprKind::Closure(closure) = ex.kind {
                self.check(closure.def_id);
            }
            intravisit::walk_expr(self, ex);
        }
        fn visit_item(&mut self, it: &'tcx Item<'tcx>) {
            trace!(?it.owner_id);
            // The opaque type itself or its children are not within its reveal scope.
            if it.owner_id.def_id != self.def_id {
                self.check(it.owner_id.def_id);
                intravisit::walk_item(self, it);
            }
        }
        fn visit_impl_item(&mut self, it: &'tcx ImplItem<'tcx>) {
            trace!(?it.owner_id);
            // The opaque type itself or its children are not within its reveal scope.
            if it.owner_id.def_id != self.def_id {
                self.check(it.owner_id.def_id);
                intravisit::walk_impl_item(self, it);
            }
        }
        fn visit_trait_item(&mut self, it: &'tcx TraitItem<'tcx>) {
            trace!(?it.owner_id);
            self.check(it.owner_id.def_id);
            intravisit::walk_trait_item(self, it);
        }
    }

    let hir_id = tcx.hir().local_def_id_to_hir_id(def_id);
    let scope = tcx.hir().get_defining_scope(hir_id);
    let mut locator = ConstraintLocator { def_id, tcx, found: None, typeck_types: vec![] };

    debug!(?scope);

    if scope == hir::CRATE_HIR_ID {
        tcx.hir().walk_toplevel_module(&mut locator);
    } else {
        trace!("scope={:#?}", tcx.hir().get(scope));
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
            Node::Item(it) => locator.visit_item(it),
            Node::ImplItem(it) => locator.visit_impl_item(it),
            Node::TraitItem(it) => locator.visit_trait_item(it),
            other => bug!("{:?} is not a valid scope for an opaque type item", other),
        }
    }

    let Some(hidden) = locator.found else {
        let reported = tcx.sess.emit_err(UnconstrainedOpaqueType {
            span: tcx.def_span(def_id),
            name: tcx.item_name(tcx.local_parent(def_id).to_def_id()),
            what: match tcx.hir().get(scope) {
                _ if scope == hir::CRATE_HIR_ID => "module",
                Node::Item(hir::Item { kind: hir::ItemKind::Mod(_), .. }) => "module",
                Node::Item(hir::Item { kind: hir::ItemKind::Impl(_), .. }) => "impl",
                _ => "item",
            },
        });
        return tcx.ty_error(reported);
    };

    // Only check against typeck if we didn't already error
    if !hidden.ty.references_error() {
        for concrete_type in locator.typeck_types {
            if tcx.erase_regions(concrete_type.ty) != tcx.erase_regions(hidden.ty)
                && !(concrete_type, hidden).references_error()
            {
                hidden.report_mismatch(&concrete_type, tcx);
            }
        }
    }

    hidden.ty
}

fn find_opaque_ty_constraints_for_rpit(
    tcx: TyCtxt<'_>,
    def_id: LocalDefId,
    owner_def_id: LocalDefId,
) -> Ty<'_> {
    use rustc_hir::{Expr, ImplItem, Item, TraitItem};

    struct ConstraintChecker<'tcx> {
        tcx: TyCtxt<'tcx>,

        /// def_id of the opaque type whose defining uses are being checked
        def_id: LocalDefId,

        found: ty::OpaqueHiddenType<'tcx>,
    }

    impl ConstraintChecker<'_> {
        #[instrument(skip(self), level = "debug")]
        fn check(&self, def_id: LocalDefId) {
            // Use borrowck to get the type with unerased regions.
            let concrete_opaque_types = &self.tcx.mir_borrowck(def_id).concrete_opaque_types;
            debug!(?concrete_opaque_types);
            for &(def_id, concrete_type) in concrete_opaque_types {
                if def_id != self.def_id {
                    // Ignore constraints for other opaque types.
                    continue;
                }

                debug!(?concrete_type, "found constraint");

                if concrete_type.ty != self.found.ty
                    && !(concrete_type, self.found).references_error()
                {
                    self.found.report_mismatch(&concrete_type, self.tcx);
                }
            }
        }
    }

    impl<'tcx> intravisit::Visitor<'tcx> for ConstraintChecker<'tcx> {
        type NestedFilter = nested_filter::OnlyBodies;

        fn nested_visit_map(&mut self) -> Self::Map {
            self.tcx.hir()
        }
        fn visit_expr(&mut self, ex: &'tcx Expr<'tcx>) {
            if let hir::ExprKind::Closure(closure) = ex.kind {
                self.check(closure.def_id);
            }
            intravisit::walk_expr(self, ex);
        }
        fn visit_item(&mut self, it: &'tcx Item<'tcx>) {
            trace!(?it.owner_id);
            // The opaque type itself or its children are not within its reveal scope.
            if it.owner_id.def_id != self.def_id {
                self.check(it.owner_id.def_id);
                intravisit::walk_item(self, it);
            }
        }
        fn visit_impl_item(&mut self, it: &'tcx ImplItem<'tcx>) {
            trace!(?it.owner_id);
            // The opaque type itself or its children are not within its reveal scope.
            if it.owner_id.def_id != self.def_id {
                self.check(it.owner_id.def_id);
                intravisit::walk_impl_item(self, it);
            }
        }
        fn visit_trait_item(&mut self, it: &'tcx TraitItem<'tcx>) {
            trace!(?it.owner_id);
            self.check(it.owner_id.def_id);
            intravisit::walk_trait_item(self, it);
        }
    }

    let concrete = tcx.mir_borrowck(owner_def_id).concrete_opaque_types.get(&def_id).copied();

    if let Some(concrete) = concrete {
        let scope = tcx.hir().local_def_id_to_hir_id(owner_def_id);
        debug!(?scope);
        let mut locator = ConstraintChecker { def_id, tcx, found: concrete };

        match tcx.hir().get(scope) {
            Node::Item(it) => intravisit::walk_item(&mut locator, it),
            Node::ImplItem(it) => intravisit::walk_impl_item(&mut locator, it),
            Node::TraitItem(it) => intravisit::walk_trait_item(&mut locator, it),
            other => bug!("{:?} is not a valid scope for an opaque type item", other),
        }
    }

    concrete.map(|concrete| concrete.ty).unwrap_or_else(|| {
        let table = tcx.typeck(owner_def_id);
        if let Some(guar) = table.tainted_by_errors {
            // Some error in the
            // owner fn prevented us from populating
            // the `concrete_opaque_types` table.
            tcx.ty_error(guar)
        } else {
            table.concrete_opaque_types.get(&def_id).map(|ty| ty.ty).unwrap_or_else(|| {
                // We failed to resolve the opaque type or it
                // resolves to itself. We interpret this as the
                // no values of the hidden type ever being constructed,
                // so we can just make the hidden type be `!`.
                // For backwards compatibility reasons, we fall back to
                // `()` until we the diverging default is changed.
                tcx.mk_diverging_default()
            })
        }
    })
}

fn infer_placeholder_type<'a>(
    tcx: TyCtxt<'a>,
    def_id: LocalDefId,
    body_id: hir::BodyId,
    span: Span,
    item_ident: Ident,
    kind: &'static str,
) -> Ty<'a> {
    // Attempts to make the type nameable by turning FnDefs into FnPtrs.
    struct MakeNameable<'tcx> {
        tcx: TyCtxt<'tcx>,
    }

    impl<'tcx> TypeFolder<TyCtxt<'tcx>> for MakeNameable<'tcx> {
        fn interner(&self) -> TyCtxt<'tcx> {
            self.tcx
        }

        fn fold_ty(&mut self, ty: Ty<'tcx>) -> Ty<'tcx> {
            let ty = match *ty.kind() {
                ty::FnDef(def_id, substs) => {
                    self.tcx.mk_fn_ptr(self.tcx.fn_sig(def_id).subst(self.tcx, substs))
                }
                _ => ty,
            };

            ty.super_fold_with(self)
        }
    }

    let ty = tcx.diagnostic_only_typeck(def_id).node_type(body_id.hir_id);

    // If this came from a free `const` or `static mut?` item,
    // then the user may have written e.g. `const A = 42;`.
    // In this case, the parser has stashed a diagnostic for
    // us to improve in typeck so we do that now.
    match tcx.sess.diagnostic().steal_diagnostic(span, StashKey::ItemNoType) {
        Some(mut err) => {
            if !ty.references_error() {
                // Only suggest adding `:` if it was missing (and suggested by parsing diagnostic)
                let colon = if span == item_ident.span.shrink_to_hi() { ":" } else { "" };

                // The parser provided a sub-optimal `HasPlaceholders` suggestion for the type.
                // We are typeck and have the real type, so remove that and suggest the actual type.
                // FIXME(eddyb) this looks like it should be functionality on `Diagnostic`.
                if let Ok(suggestions) = &mut err.suggestions {
                    suggestions.clear();
                }

                if let Some(ty) = ty.make_suggestable(tcx, false) {
                    err.span_suggestion(
                        span,
                        &format!("provide a type for the {item}", item = kind),
                        format!("{colon} {ty}"),
                        Applicability::MachineApplicable,
                    );
                } else {
                    with_forced_trimmed_paths!(err.span_note(
                        tcx.hir().body(body_id).value.span,
                        &format!("however, the inferred type `{ty}` cannot be named"),
                    ));
                }
            }

            err.emit();
        }
        None => {
            let mut diag = bad_placeholder(tcx, vec![span], kind);

            if !ty.references_error() {
                if let Some(ty) = ty.make_suggestable(tcx, false) {
                    diag.span_suggestion(
                        span,
                        "replace with the correct type",
                        ty,
                        Applicability::MachineApplicable,
                    );
                } else {
                    with_forced_trimmed_paths!(diag.span_note(
                        tcx.hir().body(body_id).value.span,
                        &format!("however, the inferred type `{ty}` cannot be named"),
                    ));
                }
            }

            diag.emit();
        }
    }

    // Typeck doesn't expect erased regions to be returned from `type_of`.
    tcx.fold_regions(ty, |r, _| match *r {
        ty::ReErased => tcx.lifetimes.re_static,
        _ => r,
    })
}

fn check_feature_inherent_assoc_ty(tcx: TyCtxt<'_>, span: Span) {
    if !tcx.features().inherent_associated_types {
        use rustc_session::parse::feature_err;
        use rustc_span::symbol::sym;
        feature_err(
            &tcx.sess.parse_sess,
            sym::inherent_associated_types,
            span,
            "inherent associated types are unstable",
        )
        .emit();
    }
}
