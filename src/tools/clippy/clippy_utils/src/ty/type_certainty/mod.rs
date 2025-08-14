//! A heuristic to tell whether an expression's type can be determined purely from its
//! subexpressions, and the arguments and locals they use. Put another way, `expr_type_is_certain`
//! tries to tell whether an expression's type can be determined without appeal to the surrounding
//! context.
//!
//! This is, in some sense, a counterpart to `let_unit_value`'s `expr_needs_inferred_result`.
//! Intuitively, that function determines whether an expression's type is needed for type inference,
//! whereas `expr_type_is_certain` determines whether type inference is needed for an expression's
//! type.
//!
//! As a heuristic, `expr_type_is_certain` may produce false negatives, but a false positive should
//! be considered a bug.

use crate::paths::{PathNS, lookup_path};
use rustc_ast::{LitFloatType, LitIntType, LitKind};
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::DefId;
use rustc_hir::intravisit::{InferKind, Visitor, VisitorExt, walk_qpath, walk_ty};
use rustc_hir::{self as hir, AmbigArg, Expr, ExprKind, GenericArgs, HirId, Node, Param, PathSegment, QPath, TyKind};
use rustc_lint::LateContext;
use rustc_middle::ty::{self, AdtDef, GenericArgKind, Ty};
use rustc_span::Span;

mod certainty;
use certainty::{Certainty, Meet, join, meet};

pub fn expr_type_is_certain(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    expr_type_certainty(cx, expr, false).is_certain()
}

/// Determine the type certainty of `expr`. `in_arg` indicates that the expression happens within
/// the evaluation of a function or method call argument.
fn expr_type_certainty(cx: &LateContext<'_>, expr: &Expr<'_>, in_arg: bool) -> Certainty {
    let certainty = match &expr.kind {
        ExprKind::Unary(_, expr)
        | ExprKind::Field(expr, _)
        | ExprKind::Index(expr, _, _)
        | ExprKind::AddrOf(_, _, expr) => expr_type_certainty(cx, expr, in_arg),

        ExprKind::Array(exprs) => join(exprs.iter().map(|expr| expr_type_certainty(cx, expr, in_arg))),

        ExprKind::Call(callee, args) => {
            let lhs = expr_type_certainty(cx, callee, false);
            let rhs = if type_is_inferable_from_arguments(cx, expr) {
                meet(args.iter().map(|arg| expr_type_certainty(cx, arg, true)))
            } else {
                Certainty::Uncertain
            };
            lhs.join_clearing_def_ids(rhs)
        },

        ExprKind::MethodCall(method, receiver, args, _) => {
            let mut receiver_type_certainty = expr_type_certainty(cx, receiver, false);
            // Even if `receiver_type_certainty` is `Certain(Some(..))`, the `Self` type in the method
            // identified by `type_dependent_def_id(..)` can differ. This can happen as a result of a `deref`,
            // for example. So update the `DefId` in `receiver_type_certainty` (if any).
            if let Some(method_def_id) = cx.typeck_results().type_dependent_def_id(expr.hir_id)
                && let Some(self_ty_def_id) = adt_def_id(self_ty(cx, method_def_id))
            {
                receiver_type_certainty = receiver_type_certainty.with_def_id(self_ty_def_id);
            }
            let lhs = path_segment_certainty(cx, receiver_type_certainty, method, false);
            let rhs = if type_is_inferable_from_arguments(cx, expr) {
                meet(
                    std::iter::once(receiver_type_certainty)
                        .chain(args.iter().map(|arg| expr_type_certainty(cx, arg, true))),
                )
            } else {
                Certainty::Uncertain
            };
            lhs.join(rhs)
        },

        ExprKind::Tup(exprs) => meet(exprs.iter().map(|expr| expr_type_certainty(cx, expr, in_arg))),

        ExprKind::Binary(_, lhs, rhs) => {
            // If one of the side of the expression is uncertain, the certainty will come from the other side,
            // with no information on the type.
            match (
                expr_type_certainty(cx, lhs, in_arg),
                expr_type_certainty(cx, rhs, in_arg),
            ) {
                (Certainty::Uncertain, Certainty::Certain(_)) | (Certainty::Certain(_), Certainty::Uncertain) => {
                    Certainty::Certain(None)
                },
                (l, r) => l.meet(r),
            }
        },

        ExprKind::Lit(lit) => {
            if !in_arg
                && matches!(
                    lit.node,
                    LitKind::Int(_, LitIntType::Unsuffixed) | LitKind::Float(_, LitFloatType::Unsuffixed)
                )
            {
                Certainty::Uncertain
            } else {
                Certainty::Certain(None)
            }
        },

        ExprKind::Cast(_, ty) => type_certainty(cx, ty),

        ExprKind::If(_, if_expr, Some(else_expr)) => {
            expr_type_certainty(cx, if_expr, in_arg).join(expr_type_certainty(cx, else_expr, in_arg))
        },

        ExprKind::Path(qpath) => qpath_certainty(cx, qpath, false),

        ExprKind::Struct(qpath, _, _) => qpath_certainty(cx, qpath, true),

        _ => Certainty::Uncertain,
    };

    let expr_ty = cx.typeck_results().expr_ty(expr);
    if let Some(def_id) = adt_def_id(expr_ty) {
        certainty.with_def_id(def_id)
    } else {
        certainty.clear_def_id()
    }
}

struct CertaintyVisitor<'cx, 'tcx> {
    cx: &'cx LateContext<'tcx>,
    certainty: Certainty,
}

impl<'cx, 'tcx> CertaintyVisitor<'cx, 'tcx> {
    fn new(cx: &'cx LateContext<'tcx>) -> Self {
        Self {
            cx,
            certainty: Certainty::Certain(None),
        }
    }
}

impl<'cx> Visitor<'cx> for CertaintyVisitor<'cx, '_> {
    fn visit_qpath(&mut self, qpath: &'cx QPath<'_>, hir_id: HirId, _: Span) {
        self.certainty = self.certainty.meet(qpath_certainty(self.cx, qpath, true));
        if self.certainty != Certainty::Uncertain {
            walk_qpath(self, qpath, hir_id);
        }
    }

    fn visit_ty(&mut self, ty: &'cx hir::Ty<'_, AmbigArg>) {
        if self.certainty != Certainty::Uncertain {
            walk_ty(self, ty);
        }
    }

    fn visit_infer(&mut self, _inf_id: HirId, _inf_span: Span, _kind: InferKind<'cx>) -> Self::Result {
        self.certainty = Certainty::Uncertain;
    }
}

fn type_certainty(cx: &LateContext<'_>, ty: &hir::Ty<'_>) -> Certainty {
    // Handle `TyKind::Path` specially so that its `DefId` can be preserved.
    //
    // Note that `CertaintyVisitor::new` initializes the visitor's internal certainty to
    // `Certainty::Certain(None)`. Furthermore, if a `TyKind::Path` is encountered while traversing
    // `ty`, the result of the call to `qpath_certainty` is combined with the visitor's internal
    // certainty using `Certainty::meet`. Thus, if the `TyKind::Path` were not treated specially here,
    // the resulting certainty would be `Certainty::Certain(None)`.
    if let TyKind::Path(qpath) = &ty.kind {
        return qpath_certainty(cx, qpath, true);
    }

    let mut visitor = CertaintyVisitor::new(cx);
    visitor.visit_ty_unambig(ty);
    visitor.certainty
}

fn generic_args_certainty(cx: &LateContext<'_>, args: &GenericArgs<'_>) -> Certainty {
    let mut visitor = CertaintyVisitor::new(cx);
    visitor.visit_generic_args(args);
    visitor.certainty
}

/// Tries to tell whether a `QPath` resolves to something certain, e.g., whether all of its path
/// segments generic arguments are instantiated.
///
/// `qpath` could refer to either a type or a value. The heuristic never needs the `DefId` of a
/// value. So `DefId`s are retained only when `resolves_to_type` is true.
fn qpath_certainty(cx: &LateContext<'_>, qpath: &QPath<'_>, resolves_to_type: bool) -> Certainty {
    let certainty = match qpath {
        QPath::Resolved(ty, path) => {
            let len = path.segments.len();
            path.segments.iter().enumerate().fold(
                ty.map_or(Certainty::Uncertain, |ty| type_certainty(cx, ty)),
                |parent_certainty, (i, path_segment)| {
                    path_segment_certainty(cx, parent_certainty, path_segment, i != len - 1 || resolves_to_type)
                },
            )
        },

        QPath::TypeRelative(ty, path_segment) => {
            path_segment_certainty(cx, type_certainty(cx, ty), path_segment, resolves_to_type)
        },

        QPath::LangItem(lang_item, ..) => cx
            .tcx
            .lang_items()
            .get(*lang_item)
            .map_or(Certainty::Uncertain, |def_id| {
                let generics = cx.tcx.generics_of(def_id);
                if generics.is_empty() {
                    Certainty::Certain(if resolves_to_type { Some(def_id) } else { None })
                } else {
                    Certainty::Uncertain
                }
            }),
    };
    debug_assert!(resolves_to_type || certainty.to_def_id().is_none());
    certainty
}

/// Tries to tell whether `param` resolves to something certain, e.g., a non-wildcard type if
/// present. The certainty `DefId` is cleared before returning.
fn param_certainty(cx: &LateContext<'_>, param: &Param<'_>) -> Certainty {
    let owner_did = cx.tcx.hir_enclosing_body_owner(param.hir_id);
    let Some(fn_decl) = cx.tcx.hir_fn_decl_by_hir_id(cx.tcx.local_def_id_to_hir_id(owner_did)) else {
        return Certainty::Uncertain;
    };
    let inputs = fn_decl.inputs;
    let body_params = cx.tcx.hir_body_owned_by(owner_did).params;
    std::iter::zip(body_params, inputs)
        .find(|(p, _)| p.hir_id == param.hir_id)
        .map_or(Certainty::Uncertain, |(_, ty)| type_certainty(cx, ty).clear_def_id())
}

fn path_segment_certainty(
    cx: &LateContext<'_>,
    parent_certainty: Certainty,
    path_segment: &PathSegment<'_>,
    resolves_to_type: bool,
) -> Certainty {
    let certainty = match update_res(cx, parent_certainty, path_segment, resolves_to_type).unwrap_or(path_segment.res) {
        // A definition's type is certain if it refers to something without generics (e.g., a crate or module, or
        // an unparameterized type), or the generics are instantiated with arguments that are certain.
        //
        // If the parent is uncertain, then the current path segment must account for the parent's generic arguments.
        // Consider the following examples, where the current path segment is `None`:
        // - `Option::None`             // uncertain; parent (i.e., `Option`) is uncertain
        // - `Option::<Vec<u64>>::None` // certain; parent (i.e., `Option::<..>`) is certain
        // - `Option::None::<Vec<u64>>` // certain; parent (i.e., `Option`) is uncertain
        Res::Def(_, def_id) => {
            // Checking `res_generics_def_id(..)` before calling `generics_of` avoids an ICE.
            if cx.tcx.res_generics_def_id(path_segment.res).is_some() {
                let generics = cx.tcx.generics_of(def_id);

                let own_count = generics.own_params.len();
                let lhs = if (parent_certainty.is_certain() || generics.parent_count == 0) && own_count == 0 {
                    Certainty::Certain(None)
                } else {
                    Certainty::Uncertain
                };
                let rhs = path_segment
                    .args
                    .map_or(Certainty::Uncertain, |args| generic_args_certainty(cx, args));
                // See the comment preceding `qpath_certainty`. `def_id` could refer to a type or a value.
                let certainty = lhs.join_clearing_def_ids(rhs);
                if resolves_to_type {
                    if let DefKind::TyAlias = cx.tcx.def_kind(def_id) {
                        adt_def_id(cx.tcx.type_of(def_id).instantiate_identity())
                            .map_or(certainty, |def_id| certainty.with_def_id(def_id))
                    } else {
                        certainty.with_def_id(def_id)
                    }
                } else {
                    certainty
                }
            } else {
                Certainty::Certain(None)
            }
        },

        Res::PrimTy(_) | Res::SelfTyParam { .. } | Res::SelfTyAlias { .. } | Res::SelfCtor(_) => {
            Certainty::Certain(None)
        },

        // `get_parent` because `hir_id` refers to a `Pat`, and we're interested in the node containing the `Pat`.
        Res::Local(hir_id) => match cx.tcx.parent_hir_node(hir_id) {
            // A parameter's type is not always certain, as it may come from an untyped closure definition,
            // or from a wildcard in a typed closure definition.
            Node::Param(param) => param_certainty(cx, param),
            // A local's type is certain if its type annotation is certain or it has an initializer whose
            // type is certain.
            Node::LetStmt(local) => {
                let lhs = local.ty.map_or(Certainty::Uncertain, |ty| type_certainty(cx, ty));
                let rhs = local
                    .init
                    .map_or(Certainty::Uncertain, |init| expr_type_certainty(cx, init, false));
                let certainty = lhs.join(rhs);
                if resolves_to_type {
                    certainty
                } else {
                    certainty.clear_def_id()
                }
            },
            _ => Certainty::Uncertain,
        },

        _ => Certainty::Uncertain,
    };
    debug_assert!(resolves_to_type || certainty.to_def_id().is_none());
    certainty
}

/// For at least some `QPath::TypeRelative`, the path segment's `res` can be `Res::Err`.
/// `update_res` tries to fix the resolution when `parent_certainty` is `Certain(Some(..))`.
fn update_res(
    cx: &LateContext<'_>,
    parent_certainty: Certainty,
    path_segment: &PathSegment<'_>,
    resolves_to_type: bool,
) -> Option<Res> {
    if path_segment.res == Res::Err
        && let Some(def_id) = parent_certainty.to_def_id()
    {
        let mut def_path = cx.get_def_path(def_id);
        def_path.push(path_segment.ident.name);
        let ns = if resolves_to_type { PathNS::Type } else { PathNS::Value };
        if let &[id] = lookup_path(cx.tcx, ns, &def_path).as_slice() {
            return Some(Res::Def(cx.tcx.def_kind(id), id));
        }
    }

    None
}

#[allow(clippy::cast_possible_truncation)]
fn type_is_inferable_from_arguments(cx: &LateContext<'_>, expr: &Expr<'_>) -> bool {
    let Some(callee_def_id) = (match expr.kind {
        ExprKind::Call(callee, _) => {
            let callee_ty = cx.typeck_results().expr_ty(callee);
            if let ty::FnDef(callee_def_id, _) = callee_ty.kind() {
                Some(*callee_def_id)
            } else {
                None
            }
        },
        ExprKind::MethodCall(_, _, _, _) => cx.typeck_results().type_dependent_def_id(expr.hir_id),
        _ => None,
    }) else {
        return false;
    };

    let generics = cx.tcx.generics_of(callee_def_id);
    let fn_sig = cx.tcx.fn_sig(callee_def_id).skip_binder();

    // Check that all type parameters appear in the functions input types.
    (0..(generics.parent_count + generics.own_params.len()) as u32).all(|index| {
        fn_sig
            .inputs()
            .iter()
            .any(|input_ty| contains_param(*input_ty.skip_binder(), index))
    })
}

fn self_ty<'tcx>(cx: &LateContext<'tcx>, method_def_id: DefId) -> Ty<'tcx> {
    cx.tcx.fn_sig(method_def_id).skip_binder().inputs().skip_binder()[0]
}

fn adt_def_id(ty: Ty<'_>) -> Option<DefId> {
    ty.peel_refs().ty_adt_def().map(AdtDef::did)
}

fn contains_param(ty: Ty<'_>, index: u32) -> bool {
    ty.walk()
        .any(|arg| matches!(arg.kind(), GenericArgKind::Type(ty) if ty.is_param(index)))
}
