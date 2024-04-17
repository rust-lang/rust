use crate::FnCtxt;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::PatKind;
use rustc_infer::infer::type_variable::TypeVariableOrigin;
use rustc_middle::ty::Ty;
use rustc_middle::ty::UserType;
use rustc_span::def_id::LocalDefId;
use rustc_span::Span;
use rustc_trait_selection::traits;

/// Provides context for checking patterns in declarations. More specifically this
/// allows us to infer array types if the pattern is irrefutable and allows us to infer
/// the size of the array. See issue #76342.
#[derive(Debug, Copy, Clone)]
pub(super) enum DeclOrigin<'a> {
    // from an `if let` expression
    LetExpr,
    // from `let x = ..`
    LocalDecl { els: Option<&'a hir::Block<'a>> },
}

impl<'a> DeclOrigin<'a> {
    pub(super) fn try_get_else(&self) -> Option<&'a hir::Block<'a>> {
        match self {
            Self::LocalDecl { els } => *els,
            Self::LetExpr => None,
        }
    }
}

/// A declaration is an abstraction of [hir::LetStmt] and [hir::LetExpr].
///
/// It must have a hir_id, as this is how we connect gather_locals to the check functions.
pub(super) struct Declaration<'a> {
    pub hir_id: hir::HirId,
    pub pat: &'a hir::Pat<'a>,
    pub ty: Option<&'a hir::Ty<'a>>,
    pub span: Span,
    pub init: Option<&'a hir::Expr<'a>>,
    pub origin: DeclOrigin<'a>,
}

impl<'a> From<&'a hir::LetStmt<'a>> for Declaration<'a> {
    fn from(local: &'a hir::LetStmt<'a>) -> Self {
        let hir::LetStmt { hir_id, pat, ty, span, init, els, source: _ } = *local;
        Declaration { hir_id, pat, ty, span, init, origin: DeclOrigin::LocalDecl { els } }
    }
}

impl<'a> From<(&'a hir::LetExpr<'a>, hir::HirId)> for Declaration<'a> {
    fn from((let_expr, hir_id): (&'a hir::LetExpr<'a>, hir::HirId)) -> Self {
        let hir::LetExpr { pat, ty, span, init, is_recovered: _ } = *let_expr;
        Declaration { hir_id, pat, ty, span, init: Some(init), origin: DeclOrigin::LetExpr }
    }
}

pub(super) struct GatherLocalsVisitor<'a, 'tcx> {
    fcx: &'a FnCtxt<'a, 'tcx>,
    // parameters are special cases of patterns, but we want to handle them as
    // *distinct* cases. so track when we are hitting a pattern *within* an fn
    // parameter.
    outermost_fn_param_pat: Option<(Span, hir::HirId)>,
    pat_expr_map: FxHashMap<hir::HirId, Span>,
}

impl<'a, 'tcx> GatherLocalsVisitor<'a, 'tcx> {
    pub(super) fn new(fcx: &'a FnCtxt<'a, 'tcx>) -> Self {
        Self { fcx, outermost_fn_param_pat: None, pat_expr_map: Default::default() }
    }

    fn assign(&mut self, span: Span, nid: hir::HirId, ty_opt: Option<Ty<'tcx>>) -> Ty<'tcx> {
        match ty_opt {
            None => {
                // Infer the variable's type.
                let var_ty = self.fcx.next_ty_var(TypeVariableOrigin { param_def_id: None, span });
                self.fcx.locals.borrow_mut().insert(nid, var_ty);
                var_ty
            }
            Some(typ) => {
                // Take type that the user specified.
                self.fcx.locals.borrow_mut().insert(nid, typ);
                typ
            }
        }
    }

    /// Allocates a type for a declaration, which may have a type annotation. If it does have
    /// a type annotation, then the [`Ty`] stored will be the resolved type. This may be found
    /// again during type checking by querying [`FnCtxt::local_ty`] for the same hir_id.
    fn declare(&mut self, decl: Declaration<'tcx>) {
        let local_ty = match decl.ty {
            Some(ref hir_ty) => {
                let o_ty = self.fcx.lower_ty(hir_ty);

                let c_ty = self.fcx.infcx.canonicalize_user_type_annotation(UserType::Ty(o_ty.raw));
                debug!(?hir_ty.hir_id, ?o_ty, ?c_ty, "visit_local");
                self.fcx
                    .typeck_results
                    .borrow_mut()
                    .user_provided_types_mut()
                    .insert(hir_ty.hir_id, c_ty);

                let ty = o_ty.normalized;
                match decl.pat.kind {
                    // We explicitly allow `let ref x: str = *"";`
                    hir::PatKind::Binding(hir::BindingAnnotation(hir::ByRef::Yes(_), _), ..) => {}
                    // We explicitly allow `let _: dyn Trait;` and allow the `visit_pat` check to
                    // handle `let (x, _): (sized, str) = *r;`. Otherwise with the later we'd
                    // complain incorrectly about the `str` that is otherwise unused.
                    _ if {
                        let mut is_wild = false;
                        decl.pat.walk(|pat| {
                            if let hir::PatKind::Wild = pat.kind {
                                is_wild = true;
                                false
                            } else {
                                true
                            }
                        });
                        is_wild
                    } => {}
                    _ => {
                        if self.outermost_fn_param_pat.is_some() {
                            if !self.fcx.tcx.features().unsized_fn_params {
                                self.fcx.require_type_is_sized(
                                    ty,
                                    hir_ty.span,
                                    traits::SizedArgumentType(Some(decl.pat.hir_id)),
                                );
                            }
                        } else {
                            if !self.fcx.tcx.features().unsized_locals {
                                self.fcx.require_type_is_sized(
                                    ty,
                                    hir_ty.span,
                                    traits::VariableType(decl.pat.hir_id),
                                );
                            }
                        }
                    }
                }
                Some(ty)
            }
            None => None,
        };
        self.assign(decl.span, decl.hir_id, local_ty);

        debug!(
            "local variable {:?} is assigned type {}",
            decl.pat,
            self.fcx.ty_to_string(*self.fcx.locals.borrow().get(&decl.hir_id).unwrap())
        );
    }
}

/// Builds a correspondence mapping between the bindings in a pattern and the expression that
/// originates the value. This is then used on unsized locals errors to point at the sub expression
/// That corresponds to the sub-pattern with the `?Sized` type, instead of the binding.
///
/// This is somewhat limited, as it only supports bindings, tuples and structs for now, falling back
/// on pointing at the binding otherwise.
struct JointVisitorExpr<'hir> {
    pat: &'hir hir::Pat<'hir>,
    expr: &'hir hir::Expr<'hir>,
    map: FxHashMap<hir::HirId, Span>,
}

impl<'hir> JointVisitorExpr<'hir> {
    fn walk(&mut self) {
        match (self.pat.kind, self.expr.kind) {
            (hir::PatKind::Tuple(pat_fields, pos), hir::ExprKind::Tup(expr_fields))
                if pat_fields.len() == expr_fields.len() && pos.as_opt_usize() == None =>
            {
                for (pat, expr) in pat_fields.iter().zip(expr_fields.iter()) {
                    self.map.insert(pat.hir_id, expr.span);
                    let mut v = JointVisitorExpr { pat, expr, map: Default::default() };
                    v.walk();
                    self.map.extend(v.map);
                }
            }
            (hir::PatKind::Binding(..), hir::ExprKind::MethodCall(path, ..)) => {
                self.map.insert(self.pat.hir_id, path.ident.span);
            }
            (hir::PatKind::Binding(..), _) => {
                self.map.insert(self.pat.hir_id, self.expr.span);
            }
            (
                hir::PatKind::Struct(pat_path, pat_fields, _),
                hir::ExprKind::Struct(call_path, expr_fields, _),
            ) if pat_path.res() == call_path.res() && pat_path.res().is_some() => {
                for (pat_field, expr_field) in pat_fields.iter().zip(expr_fields.iter()) {
                    self.map.insert(pat_field.hir_id, expr_field.span);
                    let mut v = JointVisitorExpr {
                        pat: pat_field.pat,
                        expr: expr_field.expr,
                        map: Default::default(),
                    };
                    v.walk();
                    self.map.extend(v.map);
                }
            }
            (
                hir::PatKind::TupleStruct(pat_path, pat_fields, _),
                hir::ExprKind::Call(
                    hir::Expr { kind: hir::ExprKind::Path(expr_path), .. },
                    expr_fields,
                ),
            ) if pat_path.res() == expr_path.res() && pat_path.res().is_some() => {
                for (pat, expr) in pat_fields.iter().zip(expr_fields.iter()) {
                    self.map.insert(pat.hir_id, expr.span);
                    let mut v = JointVisitorExpr { pat: pat, expr: expr, map: Default::default() };
                    v.walk();
                    self.map.extend(v.map);
                }
            }
            _ => {}
        }
    }
}

impl<'a, 'tcx> Visitor<'tcx> for GatherLocalsVisitor<'a, 'tcx> {
    // Add explicitly-declared locals.
    fn visit_local(&mut self, local: &'tcx hir::LetStmt<'tcx>) {
        self.declare(local.into());
        if let Some(init) = local.init {
            let mut v = JointVisitorExpr { pat: &local.pat, expr: &init, map: Default::default() };
            v.walk();
            self.pat_expr_map.extend(v.map);
        }
        intravisit::walk_local(self, local);
    }

    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        if let hir::ExprKind::Let(let_expr) = expr.kind {
            let mut v = JointVisitorExpr {
                pat: &let_expr.pat,
                expr: &let_expr.init,
                map: Default::default(),
            };
            v.walk();
            self.pat_expr_map.extend(v.map);
            self.declare((let_expr, expr.hir_id).into());
        }
        intravisit::walk_expr(self, expr);
    }

    fn visit_param(&mut self, param: &'tcx hir::Param<'tcx>) {
        let old_outermost_fn_param_pat =
            self.outermost_fn_param_pat.replace((param.ty_span, param.hir_id));
        intravisit::walk_param(self, param);
        self.outermost_fn_param_pat = old_outermost_fn_param_pat;
    }

    // Add pattern bindings.
    fn visit_pat(&mut self, p: &'tcx hir::Pat<'tcx>) {
        if let PatKind::Binding(_, _, ident, _) = p.kind {
            let var_ty = self.assign(p.span, p.hir_id, None);

            if let Some((ty_span, hir_id)) = self.outermost_fn_param_pat {
                if !self.fcx.tcx.features().unsized_fn_params {
                    self.fcx.require_type_is_sized(
                        var_ty,
                        ty_span,
                        // ty_span == ident.span iff this is a closure parameter with no type
                        // ascription, or if it's an implicit `self` parameter
                        traits::SizedArgumentType(
                            if ty_span == ident.span
                                && self.fcx.tcx.is_closure_like(self.fcx.body_id.into())
                            {
                                None
                            } else {
                                Some(hir_id)
                            },
                        ),
                    );
                }
            } else {
                if !self.fcx.tcx.features().unsized_locals {
                    let span = *self.pat_expr_map.get(&p.hir_id).unwrap_or(&p.span);
                    self.fcx.require_type_is_sized(var_ty, span, traits::VariableType(p.hir_id));
                }
            }

            debug!(
                "pattern binding {} is assigned to {} with type {:?}",
                ident,
                self.fcx.ty_to_string(*self.fcx.locals.borrow().get(&p.hir_id).unwrap()),
                var_ty
            );
        }
        let old_outermost_fn_param_pat = self.outermost_fn_param_pat.take();
        intravisit::walk_pat(self, p);
        self.outermost_fn_param_pat = old_outermost_fn_param_pat;
    }

    // Don't descend into the bodies of nested closures.
    fn visit_fn(
        &mut self,
        _: intravisit::FnKind<'tcx>,
        _: &'tcx hir::FnDecl<'tcx>,
        _: hir::BodyId,
        _: Span,
        _: LocalDefId,
    ) {
    }
}
