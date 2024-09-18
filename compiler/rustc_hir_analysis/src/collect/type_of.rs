use core::ops::ControlFlow;

use rustc_errors::{Applicability, StashKey, Suggestions};
use rustc_hir as hir;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::HirId;
use rustc_middle::query::plumbing::CyclePlaceholder;
use rustc_middle::ty::print::with_forced_trimmed_paths;
use rustc_middle::ty::util::IntTypeExt;
use rustc_middle::ty::{self, Article, IsSuggestable, Ty, TyCtxt, TypeVisitableExt};
use rustc_middle::{bug, span_bug};
use rustc_span::symbol::Ident;
use rustc_span::{Span, DUMMY_SP};
use tracing::debug;

use super::{bad_placeholder, ItemCtxt};
use crate::errors::TypeofReservedKeywordUsed;
use crate::hir_ty_lowering::HirTyLowerer;

mod opaque;

fn anon_const_type_of<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> Ty<'tcx> {
    use hir::*;
    use rustc_middle::ty::Ty;
    let hir_id = tcx.local_def_id_to_hir_id(def_id);

    let node = tcx.hir_node(hir_id);
    let Node::AnonConst(&AnonConst { span, .. }) = node else {
        span_bug!(
            tcx.def_span(def_id),
            "expected anon const in `anon_const_type_of`, got {node:?}"
        );
    };

    let parent_node_id = tcx.parent_hir_id(hir_id);
    let parent_node = tcx.hir_node(parent_node_id);

    let find_sym_fn = |&(op, op_sp)| match op {
        hir::InlineAsmOperand::SymFn { anon_const } if anon_const.hir_id == hir_id => {
            Some((anon_const, op_sp))
        }
        _ => None,
    };

    let find_const = |&(op, op_sp)| match op {
        hir::InlineAsmOperand::Const { anon_const } if anon_const.hir_id == hir_id => {
            Some((anon_const, op_sp))
        }
        _ => None,
    };

    match parent_node {
        // Anon consts "inside" the type system.
        Node::ConstArg(&ConstArg {
            hir_id: arg_hir_id,
            kind: ConstArgKind::Anon(&AnonConst { hir_id: anon_hir_id, .. }),
            ..
        }) if anon_hir_id == hir_id => const_arg_anon_type_of(tcx, arg_hir_id, span),

        // Anon consts outside the type system.
        Node::Expr(&Expr { kind: ExprKind::InlineAsm(asm), .. })
        | Node::Item(&Item { kind: ItemKind::GlobalAsm(asm), .. })
            if let Some((anon_const, op_sp)) = asm.operands.iter().find_map(find_sym_fn) =>
        {
            let ty = tcx.typeck(def_id).node_type(hir_id);

            match ty.kind() {
                ty::Error(_) => ty,
                ty::FnDef(..) => ty,
                _ => {
                    let guar = tcx
                        .dcx()
                        .struct_span_err(op_sp, "invalid `sym` operand")
                        .with_span_label(
                            tcx.def_span(anon_const.def_id),
                            format!("is {} `{}`", ty.kind().article(), ty),
                        )
                        .with_help("`sym` operands must refer to either a function or a static")
                        .emit();

                    Ty::new_error(tcx, guar)
                }
            }
        }
        Node::Expr(&Expr { kind: ExprKind::InlineAsm(asm), .. })
        | Node::Item(&Item { kind: ItemKind::GlobalAsm(asm), .. })
            if let Some((anon_const, op_sp)) = asm.operands.iter().find_map(find_const) =>
        {
            let ty = tcx.typeck(def_id).node_type(hir_id);

            match ty.kind() {
                ty::Error(_) => ty,
                ty::Int(_) | ty::Uint(_) => ty,
                _ => {
                    let guar = tcx
                        .dcx()
                        .struct_span_err(op_sp, "invalid type for `const` operand")
                        .with_span_label(
                            tcx.def_span(anon_const.def_id),
                            format!("is {} `{}`", ty.kind().article(), ty),
                        )
                        .with_help("`const` operands must be of an integer type")
                        .emit();

                    Ty::new_error(tcx, guar)
                }
            }
        }
        Node::Variant(Variant { disr_expr: Some(ref e), .. }) if e.hir_id == hir_id => {
            tcx.adt_def(tcx.hir().get_parent_item(hir_id)).repr().discr_type().to_ty(tcx)
        }
        // Sort of affects the type system, but only for the purpose of diagnostics
        // so no need for ConstArg.
        Node::Ty(&hir::Ty { kind: TyKind::Typeof(ref e), span, .. }) if e.hir_id == hir_id => {
            let ty = tcx.typeck(def_id).node_type(tcx.local_def_id_to_hir_id(def_id));
            let ty = tcx.fold_regions(ty, |r, _| {
                if r.is_erased() { ty::Region::new_error_misc(tcx) } else { r }
            });
            let (ty, opt_sugg) = if let Some(ty) = ty.make_suggestable(tcx, false, None) {
                (ty, Some((span, Applicability::MachineApplicable)))
            } else {
                (ty, None)
            };
            tcx.dcx().emit_err(TypeofReservedKeywordUsed { span, ty, opt_sugg });
            return ty;
        }

        _ => Ty::new_error_with_message(
            tcx,
            span,
            format!("unexpected anon const parent in type_of(): {parent_node:?}"),
        ),
    }
}

fn const_arg_anon_type_of<'tcx>(tcx: TyCtxt<'tcx>, arg_hir_id: HirId, span: Span) -> Ty<'tcx> {
    use hir::*;
    use rustc_middle::ty::Ty;

    let parent_node_id = tcx.parent_hir_id(arg_hir_id);
    let parent_node = tcx.hir_node(parent_node_id);

    let (generics, arg_idx) = match parent_node {
        // Easy case: arrays repeat expressions.
        Node::Ty(&hir::Ty { kind: TyKind::Array(_, ref constant), .. })
        | Node::Expr(&Expr { kind: ExprKind::Repeat(_, ref constant), .. })
            if constant.hir_id() == arg_hir_id =>
        {
            return tcx.types.usize;
        }
        Node::GenericParam(&GenericParam {
            def_id: param_def_id,
            kind: GenericParamKind::Const { default: Some(ct), .. },
            ..
        }) if ct.hir_id == arg_hir_id => {
            return tcx
                .type_of(param_def_id)
                .no_bound_vars()
                .expect("const parameter types cannot be generic");
        }

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
        Node::Ty(hir_ty @ hir::Ty { kind: TyKind::Path(QPath::TypeRelative(_, segment)), .. }) => {
            // Find the Item containing the associated type so we can create an ItemCtxt.
            // Using the ItemCtxt lower the HIR for the unresolved assoc type into a
            // ty which is a fully resolved projection.
            // For the code example above, this would mean lowering `Self::Assoc<3>`
            // to a ty::Alias(ty::Projection, `<Self as Foo>::Assoc<3>`).
            let item_def_id = tcx
                .hir()
                .parent_owner_iter(arg_hir_id)
                .find(|(_, node)| matches!(node, OwnerNode::Item(_)))
                .unwrap()
                .0
                .def_id;
            let ty = ItemCtxt::new(tcx, item_def_id).lower_ty(hir_ty);

            // Iterate through the generics of the projection to find the one that corresponds to
            // the def_id that this query was called with. We filter to only type and const args here
            // as a precaution for if it's ever allowed to elide lifetimes in GAT's. It currently isn't
            // but it can't hurt to be safe ^^
            if let ty::Alias(ty::Projection | ty::Inherent, projection) = ty.kind() {
                let generics = tcx.generics_of(projection.def_id);

                let arg_index = segment
                    .args
                    .and_then(|args| {
                        args.args
                            .iter()
                            .filter(|arg| arg.is_ty_or_const())
                            .position(|arg| arg.hir_id() == arg_hir_id)
                    })
                    .unwrap_or_else(|| {
                        bug!("no arg matching AnonConst in segment");
                    });

                (generics, arg_index)
            } else {
                // I dont think it's possible to reach this but I'm not 100% sure - BoxyUwU
                return Ty::new_error_with_message(
                    tcx,
                    span,
                    "unexpected non-GAT usage of an anon const",
                );
            }
        }
        Node::Expr(&Expr {
            kind:
                ExprKind::MethodCall(segment, ..) | ExprKind::Path(QPath::TypeRelative(_, segment)),
            ..
        }) => {
            let body_owner = tcx.hir().enclosing_body_owner(arg_hir_id);
            let tables = tcx.typeck(body_owner);
            // This may fail in case the method/path does not actually exist.
            // As there is no relevant param for `def_id`, we simply return
            // `None` here.
            let Some(type_dependent_def) = tables.type_dependent_def_id(parent_node_id) else {
                return Ty::new_error_with_message(
                    tcx,
                    span,
                    format!("unable to find type-dependent def for {parent_node_id:?}"),
                );
            };
            let idx = segment
                .args
                .and_then(|args| {
                    args.args
                        .iter()
                        .filter(|arg| arg.is_ty_or_const())
                        .position(|arg| arg.hir_id() == arg_hir_id)
                })
                .unwrap_or_else(|| {
                    bug!("no arg matching ConstArg in segment");
                });

            (tcx.generics_of(type_dependent_def), idx)
        }

        Node::Ty(&hir::Ty { kind: TyKind::Path(_), .. })
        | Node::Expr(&Expr { kind: ExprKind::Path(_) | ExprKind::Struct(..), .. })
        | Node::TraitRef(..)
        | Node::Pat(_) => {
            let path = match parent_node {
                Node::Ty(&hir::Ty { kind: TyKind::Path(QPath::Resolved(_, path)), .. })
                | Node::TraitRef(&TraitRef { path, .. }) => &*path,
                Node::Expr(&Expr {
                    kind:
                        ExprKind::Path(QPath::Resolved(_, path))
                        | ExprKind::Struct(&QPath::Resolved(_, path), ..),
                    ..
                }) => {
                    let body_owner = tcx.hir().enclosing_body_owner(arg_hir_id);
                    let _tables = tcx.typeck(body_owner);
                    &*path
                }
                Node::Pat(pat) => {
                    if let Some(path) = get_path_containing_arg_in_pat(pat, arg_hir_id) {
                        path
                    } else {
                        return Ty::new_error_with_message(
                            tcx,
                            span,
                            format!("unable to find const parent for {arg_hir_id} in pat {pat:?}"),
                        );
                    }
                }
                _ => {
                    return Ty::new_error_with_message(
                        tcx,
                        span,
                        format!("unexpected const parent path {parent_node:?}"),
                    );
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
                    .position(|arg| arg.hir_id() == arg_hir_id)
                    .map(|index| (index, seg))
                    .or_else(|| {
                        args.constraints
                            .iter()
                            .copied()
                            .filter_map(AssocItemConstraint::ct)
                            .position(|ct| ct.hir_id == arg_hir_id)
                            .map(|idx| (idx, seg))
                    })
            }) else {
                return Ty::new_error_with_message(tcx, span, "no arg matching AnonConst in path");
            };

            let generics = match tcx.res_generics_def_id(segment.res) {
                Some(def_id) => tcx.generics_of(def_id),
                None => {
                    return Ty::new_error_with_message(
                        tcx,
                        span,
                        format!("unexpected anon const res {:?} in path: {:?}", segment.res, path),
                    );
                }
            };

            (generics, arg_index)
        }

        _ => {
            return Ty::new_error_with_message(
                tcx,
                span,
                format!("unexpected const arg parent in type_of(): {parent_node:?}"),
            );
        }
    };

    debug!(?parent_node);
    debug!(?generics, ?arg_idx);
    if let Some(param_def_id) = generics
        .own_params
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
    {
        tcx.type_of(param_def_id).no_bound_vars().expect("const parameter types cannot be generic")
    } else {
        return Ty::new_error_with_message(
            tcx,
            span,
            format!("const generic parameter not found in {generics:?} at position {arg_idx:?}"),
        );
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

pub(super) fn type_of(tcx: TyCtxt<'_>, def_id: LocalDefId) -> ty::EarlyBinder<'_, Ty<'_>> {
    use rustc_hir::*;
    use rustc_middle::ty::Ty;

    // If we are computing `type_of` the synthesized associated type for an RPITIT in the impl
    // side, use `collect_return_position_impl_trait_in_trait_tys` to infer the value of the
    // associated type in the impl.
    match tcx.opt_rpitit_info(def_id.to_def_id()) {
        Some(ty::ImplTraitInTraitData::Impl { fn_def_id }) => {
            match tcx.collect_return_position_impl_trait_in_trait_tys(fn_def_id) {
                Ok(map) => {
                    let assoc_item = tcx.associated_item(def_id);
                    return map[&assoc_item.trait_item_def_id.unwrap()];
                }
                Err(_) => {
                    return ty::EarlyBinder::bind(Ty::new_error_with_message(
                        tcx,
                        DUMMY_SP,
                        "Could not collect return position impl trait in trait tys",
                    ));
                }
            }
        }
        // For an RPITIT in a trait, just return the corresponding opaque.
        Some(ty::ImplTraitInTraitData::Trait { opaque_def_id, .. }) => {
            return ty::EarlyBinder::bind(Ty::new_opaque(
                tcx,
                opaque_def_id,
                ty::GenericArgs::identity_for_item(tcx, opaque_def_id),
            ));
        }
        None => {}
    }

    let hir_id = tcx.local_def_id_to_hir_id(def_id);

    let icx = ItemCtxt::new(tcx, def_id);

    let output = match tcx.hir_node(hir_id) {
        Node::TraitItem(item) => match item.kind {
            TraitItemKind::Fn(..) => {
                let args = ty::GenericArgs::identity_for_item(tcx, def_id);
                Ty::new_fn_def(tcx, def_id.to_def_id(), args)
            }
            TraitItemKind::Const(ty, body_id) => body_id
                .and_then(|body_id| {
                    ty.is_suggestable_infer_ty().then(|| {
                        infer_placeholder_type(
                            icx.lowerer(),
                            def_id,
                            body_id,
                            ty.span,
                            item.ident,
                            "associated constant",
                        )
                    })
                })
                .unwrap_or_else(|| icx.lower_ty(ty)),
            TraitItemKind::Type(_, Some(ty)) => icx.lower_ty(ty),
            TraitItemKind::Type(_, None) => {
                span_bug!(item.span, "associated type missing default");
            }
        },

        Node::ImplItem(item) => match item.kind {
            ImplItemKind::Fn(..) => {
                let args = ty::GenericArgs::identity_for_item(tcx, def_id);
                Ty::new_fn_def(tcx, def_id.to_def_id(), args)
            }
            ImplItemKind::Const(ty, body_id) => {
                if ty.is_suggestable_infer_ty() {
                    infer_placeholder_type(
                        icx.lowerer(),
                        def_id,
                        body_id,
                        ty.span,
                        item.ident,
                        "associated constant",
                    )
                } else {
                    icx.lower_ty(ty)
                }
            }
            ImplItemKind::Type(ty) => {
                if tcx.impl_trait_ref(tcx.hir().get_parent_item(hir_id)).is_none() {
                    check_feature_inherent_assoc_ty(tcx, item.span);
                }

                icx.lower_ty(ty)
            }
        },

        Node::Item(item) => match item.kind {
            ItemKind::Static(ty, .., body_id) => {
                if ty.is_suggestable_infer_ty() {
                    infer_placeholder_type(
                        icx.lowerer(),
                        def_id,
                        body_id,
                        ty.span,
                        item.ident,
                        "static variable",
                    )
                } else {
                    icx.lower_ty(ty)
                }
            }
            ItemKind::Const(ty, _, body_id) => {
                if ty.is_suggestable_infer_ty() {
                    infer_placeholder_type(
                        icx.lowerer(),
                        def_id,
                        body_id,
                        ty.span,
                        item.ident,
                        "constant",
                    )
                } else {
                    icx.lower_ty(ty)
                }
            }
            ItemKind::TyAlias(self_ty, _) => icx.lower_ty(self_ty),
            ItemKind::Impl(hir::Impl { self_ty, .. }) => match self_ty.find_self_aliases() {
                spans if spans.len() > 0 => {
                    let guar = tcx
                        .dcx()
                        .emit_err(crate::errors::SelfInImplSelf { span: spans.into(), note: () });
                    Ty::new_error(tcx, guar)
                }
                _ => icx.lower_ty(*self_ty),
            },
            ItemKind::Fn(..) => {
                let args = ty::GenericArgs::identity_for_item(tcx, def_id);
                Ty::new_fn_def(tcx, def_id.to_def_id(), args)
            }
            ItemKind::Enum(..) | ItemKind::Struct(..) | ItemKind::Union(..) => {
                let def = tcx.adt_def(def_id);
                let args = ty::GenericArgs::identity_for_item(tcx, def_id);
                Ty::new_adt(tcx, def, args)
            }
            ItemKind::OpaqueTy(..) => tcx.type_of_opaque(def_id).map_or_else(
                |CyclePlaceholder(guar)| Ty::new_error(tcx, guar),
                |ty| ty.instantiate_identity(),
            ),
            ItemKind::Trait(..)
            | ItemKind::TraitAlias(..)
            | ItemKind::Macro(..)
            | ItemKind::Mod(..)
            | ItemKind::ForeignMod { .. }
            | ItemKind::GlobalAsm(..)
            | ItemKind::ExternCrate(..)
            | ItemKind::Use(..) => {
                span_bug!(item.span, "compute_type_of_item: unexpected item type: {:?}", item.kind);
            }
        },

        Node::ForeignItem(foreign_item) => match foreign_item.kind {
            ForeignItemKind::Fn(..) => {
                let args = ty::GenericArgs::identity_for_item(tcx, def_id);
                Ty::new_fn_def(tcx, def_id.to_def_id(), args)
            }
            ForeignItemKind::Static(t, _, _) => icx.lower_ty(t),
            ForeignItemKind::Type => Ty::new_foreign(tcx, def_id.to_def_id()),
        },

        Node::Ctor(def) | Node::Variant(Variant { data: def, .. }) => match def {
            VariantData::Unit(..) | VariantData::Struct { .. } => {
                tcx.type_of(tcx.hir().get_parent_item(hir_id)).instantiate_identity()
            }
            VariantData::Tuple(_, _, ctor) => {
                let args = ty::GenericArgs::identity_for_item(tcx, def_id);
                Ty::new_fn_def(tcx, ctor.to_def_id(), args)
            }
        },

        Node::Field(field) => icx.lower_ty(field.ty),

        Node::Expr(&Expr { kind: ExprKind::Closure { .. }, .. }) => {
            tcx.typeck(def_id).node_type(hir_id)
        }

        Node::AnonConst(_) => anon_const_type_of(tcx, def_id),

        Node::ConstBlock(_) => {
            let args = ty::GenericArgs::identity_for_item(tcx, def_id.to_def_id());
            args.as_inline_const().ty()
        }

        Node::GenericParam(param) => match &param.kind {
            GenericParamKind::Type { default: Some(ty), .. }
            | GenericParamKind::Const { ty, .. } => icx.lower_ty(ty),
            x => bug!("unexpected non-type Node::GenericParam: {:?}", x),
        },

        Node::ArrayLenInfer(_) => tcx.types.usize,

        x => {
            bug!("unexpected sort of node in type_of(): {:?}", x);
        }
    };
    if let Err(e) = icx.check_tainted_by_errors()
        && !output.references_error()
    {
        ty::EarlyBinder::bind(Ty::new_error(tcx, e))
    } else {
        ty::EarlyBinder::bind(output)
    }
}

pub(super) fn type_of_opaque(
    tcx: TyCtxt<'_>,
    def_id: DefId,
) -> Result<ty::EarlyBinder<'_, Ty<'_>>, CyclePlaceholder> {
    if let Some(def_id) = def_id.as_local() {
        use rustc_hir::*;

        Ok(ty::EarlyBinder::bind(match tcx.hir_node_by_def_id(def_id) {
            Node::Item(item) => match item.kind {
                ItemKind::OpaqueTy(OpaqueTy {
                    origin: hir::OpaqueTyOrigin::TyAlias { in_assoc_ty: false, .. },
                    ..
                }) => opaque::find_opaque_ty_constraints_for_tait(tcx, def_id),
                ItemKind::OpaqueTy(OpaqueTy {
                    origin: hir::OpaqueTyOrigin::TyAlias { in_assoc_ty: true, .. },
                    ..
                }) => opaque::find_opaque_ty_constraints_for_impl_trait_in_assoc_type(tcx, def_id),
                // Opaque types desugared from `impl Trait`.
                ItemKind::OpaqueTy(&OpaqueTy {
                    origin:
                        hir::OpaqueTyOrigin::FnReturn(owner) | hir::OpaqueTyOrigin::AsyncFn(owner),
                    in_trait,
                    ..
                }) => {
                    if in_trait && !tcx.defaultness(owner).has_value() {
                        span_bug!(
                            tcx.def_span(def_id),
                            "tried to get type of this RPITIT with no definition"
                        );
                    }
                    opaque::find_opaque_ty_constraints_for_rpit(tcx, def_id, owner)
                }
                _ => {
                    span_bug!(item.span, "type_of_opaque: unexpected item type: {:?}", item.kind);
                }
            },

            x => {
                bug!("unexpected sort of node in type_of_opaque(): {:?}", x);
            }
        }))
    } else {
        // Foreign opaque type will go through the foreign provider
        // and load the type from metadata.
        Ok(tcx.type_of(def_id))
    }
}

fn infer_placeholder_type<'tcx>(
    cx: &dyn HirTyLowerer<'tcx>,
    def_id: LocalDefId,
    body_id: hir::BodyId,
    span: Span,
    item_ident: Ident,
    kind: &'static str,
) -> Ty<'tcx> {
    let tcx = cx.tcx();
    let ty = tcx.diagnostic_only_typeck(def_id).node_type(body_id.hir_id);

    // If this came from a free `const` or `static mut?` item,
    // then the user may have written e.g. `const A = 42;`.
    // In this case, the parser has stashed a diagnostic for
    // us to improve in typeck so we do that now.
    let guar = cx
        .dcx()
        .try_steal_modify_and_emit_err(span, StashKey::ItemNoType, |err| {
            if !ty.references_error() {
                // Only suggest adding `:` if it was missing (and suggested by parsing diagnostic).
                let colon = if span == item_ident.span.shrink_to_hi() { ":" } else { "" };

                // The parser provided a sub-optimal `HasPlaceholders` suggestion for the type.
                // We are typeck and have the real type, so remove that and suggest the actual type.
                if let Suggestions::Enabled(suggestions) = &mut err.suggestions {
                    suggestions.clear();
                }

                if let Some(ty) = ty.make_suggestable(tcx, false, None) {
                    err.span_suggestion(
                        span,
                        format!("provide a type for the {kind}"),
                        format!("{colon} {ty}"),
                        Applicability::MachineApplicable,
                    );
                } else {
                    with_forced_trimmed_paths!(err.span_note(
                        tcx.hir().body(body_id).value.span,
                        format!("however, the inferred type `{ty}` cannot be named"),
                    ));
                }
            }
        })
        .unwrap_or_else(|| {
            let mut diag = bad_placeholder(cx, vec![span], kind);

            if !ty.references_error() {
                if let Some(ty) = ty.make_suggestable(tcx, false, None) {
                    diag.span_suggestion(
                        span,
                        "replace with the correct type",
                        ty,
                        Applicability::MachineApplicable,
                    );
                } else {
                    with_forced_trimmed_paths!(diag.span_note(
                        tcx.hir().body(body_id).value.span,
                        format!("however, the inferred type `{ty}` cannot be named"),
                    ));
                }
            }
            diag.emit()
        });
    Ty::new_error(tcx, guar)
}

fn check_feature_inherent_assoc_ty(tcx: TyCtxt<'_>, span: Span) {
    if !tcx.features().inherent_associated_types {
        use rustc_session::parse::feature_err;
        use rustc_span::symbol::sym;
        feature_err(
            &tcx.sess,
            sym::inherent_associated_types,
            span,
            "inherent associated types are unstable",
        )
        .emit();
    }
}

pub(crate) fn type_alias_is_lazy<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId) -> bool {
    use hir::intravisit::Visitor;
    if tcx.features().lazy_type_alias {
        return true;
    }
    struct HasTait;
    impl<'tcx> Visitor<'tcx> for HasTait {
        type Result = ControlFlow<()>;
        fn visit_ty(&mut self, t: &'tcx hir::Ty<'tcx>) -> Self::Result {
            if let hir::TyKind::OpaqueDef(..) = t.kind {
                ControlFlow::Break(())
            } else {
                hir::intravisit::walk_ty(self, t)
            }
        }
    }
    HasTait.visit_ty(tcx.hir().expect_item(def_id).expect_ty_alias().0).is_break()
}
