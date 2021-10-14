use super::FnCtxt;
use crate::astconv::AstConv;

use rustc_ast::util::parser::ExprPrecedence;
use rustc_span::{self, MultiSpan, Span};

use rustc_errors::{Applicability, DiagnosticBuilder};
use rustc_hir as hir;
use rustc_hir::def::{CtorOf, DefKind};
use rustc_hir::lang_items::LangItem;
use rustc_hir::{
    Expr, ExprKind, GenericBound, ItemKind, Node, Path, QPath, Stmt, StmtKind, TyKind,
    WherePredicate,
};
use rustc_infer::infer::{self, TyCtxtInferExt};

use rustc_middle::lint::in_external_macro;
use rustc_middle::ty::{self, Binder, Ty};
use rustc_span::symbol::{kw, sym};

use rustc_middle::ty::subst::GenericArgKind;
use std::iter;

impl<'a, 'tcx> FnCtxt<'a, 'tcx> {
    pub(in super::super) fn suggest_semicolon_at_end(
        &self,
        span: Span,
        err: &mut DiagnosticBuilder<'_>,
    ) {
        err.span_suggestion_short(
            span.shrink_to_hi(),
            "consider using a semicolon here",
            ";".to_string(),
            Applicability::MachineApplicable,
        );
    }

    /// On implicit return expressions with mismatched types, provides the following suggestions:
    ///
    /// - Points out the method's return type as the reason for the expected type.
    /// - Possible missing semicolon.
    /// - Possible missing return type if the return type is the default, and not `fn main()`.
    pub fn suggest_mismatched_types_on_tail(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        blk_id: hir::HirId,
    ) -> bool {
        let expr = expr.peel_drop_temps();
        // If the expression is from an external macro, then do not suggest
        // adding a semicolon, because there's nowhere to put it.
        // See issue #81943.
        if expr.can_have_side_effects() && !in_external_macro(self.tcx.sess, expr.span) {
            self.suggest_missing_semicolon(err, expr, expected);
        }
        let mut pointing_at_return_type = false;
        if let Some((fn_decl, can_suggest)) = self.get_fn_decl(blk_id) {
            let fn_id = self.tcx.hir().get_return_block(blk_id).unwrap();
            pointing_at_return_type = self.suggest_missing_return_type(
                err,
                &fn_decl,
                expected,
                found,
                can_suggest,
                fn_id,
            );
            self.suggest_missing_break_or_return_expr(
                err, expr, &fn_decl, expected, found, blk_id, fn_id,
            );
        }
        pointing_at_return_type
    }

    /// When encountering an fn-like ctor that needs to unify with a value, check whether calling
    /// the ctor would successfully solve the type mismatch and if so, suggest it:
    /// ```
    /// fn foo(x: usize) -> usize { x }
    /// let x: usize = foo;  // suggest calling the `foo` function: `foo(42)`
    /// ```
    fn suggest_fn_call(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) -> bool {
        let hir = self.tcx.hir();
        let (def_id, sig) = match *found.kind() {
            ty::FnDef(def_id, _) => (def_id, found.fn_sig(self.tcx)),
            ty::Closure(def_id, substs) => (def_id, substs.as_closure().sig()),
            _ => return false,
        };

        let sig = self.replace_bound_vars_with_fresh_vars(expr.span, infer::FnCall, sig).0;
        let sig = self.normalize_associated_types_in(expr.span, sig);
        if self.can_coerce(sig.output(), expected) {
            let (mut sugg_call, applicability) = if sig.inputs().is_empty() {
                (String::new(), Applicability::MachineApplicable)
            } else {
                ("...".to_string(), Applicability::HasPlaceholders)
            };
            let mut msg = "call this function";
            match hir.get_if_local(def_id) {
                Some(
                    Node::Item(hir::Item { kind: ItemKind::Fn(.., body_id), .. })
                    | Node::ImplItem(hir::ImplItem {
                        kind: hir::ImplItemKind::Fn(_, body_id), ..
                    })
                    | Node::TraitItem(hir::TraitItem {
                        kind: hir::TraitItemKind::Fn(.., hir::TraitFn::Provided(body_id)),
                        ..
                    }),
                ) => {
                    let body = hir.body(*body_id);
                    sugg_call = body
                        .params
                        .iter()
                        .map(|param| match &param.pat.kind {
                            hir::PatKind::Binding(_, _, ident, None)
                                if ident.name != kw::SelfLower =>
                            {
                                ident.to_string()
                            }
                            _ => "_".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                }
                Some(Node::Expr(hir::Expr {
                    kind: ExprKind::Closure(_, _, body_id, _, _),
                    span: full_closure_span,
                    ..
                })) => {
                    if *full_closure_span == expr.span {
                        return false;
                    }
                    msg = "call this closure";
                    let body = hir.body(*body_id);
                    sugg_call = body
                        .params
                        .iter()
                        .map(|param| match &param.pat.kind {
                            hir::PatKind::Binding(_, _, ident, None)
                                if ident.name != kw::SelfLower =>
                            {
                                ident.to_string()
                            }
                            _ => "_".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                }
                Some(Node::Ctor(hir::VariantData::Tuple(fields, _))) => {
                    sugg_call = fields.iter().map(|_| "_").collect::<Vec<_>>().join(", ");
                    match def_id.as_local().map(|def_id| hir.def_kind(def_id)) {
                        Some(DefKind::Ctor(hir::def::CtorOf::Variant, _)) => {
                            msg = "instantiate this tuple variant";
                        }
                        Some(DefKind::Ctor(CtorOf::Struct, _)) => {
                            msg = "instantiate this tuple struct";
                        }
                        _ => {}
                    }
                }
                Some(Node::ForeignItem(hir::ForeignItem {
                    kind: hir::ForeignItemKind::Fn(_, idents, _),
                    ..
                })) => {
                    sugg_call = idents
                        .iter()
                        .map(|ident| {
                            if ident.name != kw::SelfLower {
                                ident.to_string()
                            } else {
                                "_".to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                }
                Some(Node::TraitItem(hir::TraitItem {
                    kind: hir::TraitItemKind::Fn(.., hir::TraitFn::Required(idents)),
                    ..
                })) => {
                    sugg_call = idents
                        .iter()
                        .map(|ident| {
                            if ident.name != kw::SelfLower {
                                ident.to_string()
                            } else {
                                "_".to_string()
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                }
                _ => {}
            }
            err.span_suggestion_verbose(
                expr.span.shrink_to_hi(),
                &format!("use parentheses to {}", msg),
                format!("({})", sugg_call),
                applicability,
            );
            return true;
        }
        false
    }

    pub fn suggest_deref_ref_or_into(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'tcx>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        expected_ty_expr: Option<&'tcx hir::Expr<'tcx>>,
    ) {
        let expr = expr.peel_blocks();
        if let Some((sp, msg, suggestion, applicability, verbose)) =
            self.check_ref(expr, found, expected)
        {
            if verbose {
                err.span_suggestion_verbose(sp, msg, suggestion, applicability);
            } else {
                err.span_suggestion(sp, msg, suggestion, applicability);
            }
        } else if let (ty::FnDef(def_id, ..), true) =
            (&found.kind(), self.suggest_fn_call(err, expr, expected, found))
        {
            if let Some(sp) = self.tcx.hir().span_if_local(*def_id) {
                let sp = self.sess().source_map().guess_head_span(sp);
                err.span_label(sp, &format!("{} defined here", found));
            }
        } else if !self.check_for_cast(err, expr, found, expected, expected_ty_expr) {
            let is_struct_pat_shorthand_field =
                self.maybe_get_struct_pattern_shorthand_field(expr).is_some();
            let methods = self.get_conversion_methods(expr.span, expected, found, expr.hir_id);
            if !methods.is_empty() {
                if let Ok(expr_text) = self.sess().source_map().span_to_snippet(expr.span) {
                    let mut suggestions = iter::zip(iter::repeat(&expr_text), &methods)
                        .filter_map(|(receiver, method)| {
                            let method_call = format!(".{}()", method.name);
                            if receiver.ends_with(&method_call) {
                                None // do not suggest code that is already there (#53348)
                            } else {
                                let method_call_list = [".to_vec()", ".to_string()"];
                                let mut sugg = if receiver.ends_with(".clone()")
                                    && method_call_list.contains(&method_call.as_str())
                                {
                                    let max_len = receiver.rfind('.').unwrap();
                                    vec![(
                                        expr.span,
                                        format!("{}{}", &receiver[..max_len], method_call),
                                    )]
                                } else {
                                    if expr.precedence().order()
                                        < ExprPrecedence::MethodCall.order()
                                    {
                                        vec![
                                            (expr.span.shrink_to_lo(), "(".to_string()),
                                            (expr.span.shrink_to_hi(), format!("){}", method_call)),
                                        ]
                                    } else {
                                        vec![(expr.span.shrink_to_hi(), method_call)]
                                    }
                                };
                                if is_struct_pat_shorthand_field {
                                    sugg.insert(
                                        0,
                                        (expr.span.shrink_to_lo(), format!("{}: ", receiver)),
                                    );
                                }
                                Some(sugg)
                            }
                        })
                        .peekable();
                    if suggestions.peek().is_some() {
                        err.multipart_suggestions(
                            "try using a conversion method",
                            suggestions,
                            Applicability::MaybeIncorrect,
                        );
                    }
                }
            } else if found.to_string().starts_with("Option<")
                && expected.to_string() == "Option<&str>"
            {
                if let ty::Adt(_def, subst) = found.kind() {
                    if subst.len() != 0 {
                        if let GenericArgKind::Type(ty) = subst[0].unpack() {
                            let peeled = ty.peel_refs().to_string();
                            if peeled == "String" {
                                let ref_cnt = ty.to_string().len() - peeled.len();
                                let result = format!(".map(|x| &*{}x)", "*".repeat(ref_cnt));
                                err.span_suggestion_verbose(
                                    expr.span.shrink_to_hi(),
                                    "try converting the passed type into a `&str`",
                                    result,
                                    Applicability::MaybeIncorrect,
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// When encountering the expected boxed value allocated in the stack, suggest allocating it
    /// in the heap by calling `Box::new()`.
    pub(in super::super) fn suggest_boxing_when_appropriate(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) {
        if self.tcx.hir().is_inside_const_context(expr.hir_id) {
            // Do not suggest `Box::new` in const context.
            return;
        }
        if !expected.is_box() || found.is_box() {
            return;
        }
        let boxed_found = self.tcx.mk_box(found);
        if self.can_coerce(boxed_found, expected) {
            err.multipart_suggestion(
                "store this in the heap by calling `Box::new`",
                vec![
                    (expr.span.shrink_to_lo(), "Box::new(".to_string()),
                    (expr.span.shrink_to_hi(), ")".to_string()),
                ],
                Applicability::MachineApplicable,
            );
            err.note(
                "for more on the distinction between the stack and the heap, read \
                 https://doc.rust-lang.org/book/ch15-01-box.html, \
                 https://doc.rust-lang.org/rust-by-example/std/box.html, and \
                 https://doc.rust-lang.org/std/boxed/index.html",
            );
        }
    }

    /// When encountering a closure that captures variables, where a FnPtr is expected,
    /// suggest a non-capturing closure
    pub(in super::super) fn suggest_no_capture_closure(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) {
        if let (ty::FnPtr(_), ty::Closure(def_id, _)) = (expected.kind(), found.kind()) {
            if let Some(upvars) = self.tcx.upvars_mentioned(*def_id) {
                // Report upto four upvars being captured to reduce the amount error messages
                // reported back to the user.
                let spans_and_labels = upvars
                    .iter()
                    .take(4)
                    .map(|(var_hir_id, upvar)| {
                        let var_name = self.tcx.hir().name(*var_hir_id).to_string();
                        let msg = format!("`{}` captured here", var_name);
                        (upvar.span, msg)
                    })
                    .collect::<Vec<_>>();

                let mut multi_span: MultiSpan =
                    spans_and_labels.iter().map(|(sp, _)| *sp).collect::<Vec<_>>().into();
                for (sp, label) in spans_and_labels {
                    multi_span.push_span_label(sp, label);
                }
                err.span_note(
                    multi_span,
                    "closures can only be coerced to `fn` types if they do not capture any variables"
                );
            }
        }
    }

    /// When encountering an `impl Future` where `BoxFuture` is expected, suggest `Box::pin`.
    #[instrument(skip(self, err))]
    pub(in super::super) fn suggest_calling_boxed_future_when_appropriate(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
    ) -> bool {
        // Handle #68197.

        if self.tcx.hir().is_inside_const_context(expr.hir_id) {
            // Do not suggest `Box::new` in const context.
            return false;
        }
        let pin_did = self.tcx.lang_items().pin_type();
        // This guards the `unwrap` and `mk_box` below.
        if pin_did.is_none() || self.tcx.lang_items().owned_box().is_none() {
            return false;
        }
        let box_found = self.tcx.mk_box(found);
        let pin_box_found = self.tcx.mk_lang_item(box_found, LangItem::Pin).unwrap();
        let pin_found = self.tcx.mk_lang_item(found, LangItem::Pin).unwrap();
        match expected.kind() {
            ty::Adt(def, _) if Some(def.did) == pin_did => {
                if self.can_coerce(pin_box_found, expected) {
                    debug!("can coerce {:?} to {:?}, suggesting Box::pin", pin_box_found, expected);
                    match found.kind() {
                        ty::Adt(def, _) if def.is_box() => {
                            err.help("use `Box::pin`");
                        }
                        _ => {
                            err.multipart_suggestion(
                                "you need to pin and box this expression",
                                vec![
                                    (expr.span.shrink_to_lo(), "Box::pin(".to_string()),
                                    (expr.span.shrink_to_hi(), ")".to_string()),
                                ],
                                Applicability::MaybeIncorrect,
                            );
                        }
                    }
                    true
                } else if self.can_coerce(pin_found, expected) {
                    match found.kind() {
                        ty::Adt(def, _) if def.is_box() => {
                            err.help("use `Box::pin`");
                            true
                        }
                        _ => false,
                    }
                } else {
                    false
                }
            }
            ty::Adt(def, _) if def.is_box() && self.can_coerce(box_found, expected) => {
                // Check if the parent expression is a call to Pin::new.  If it
                // is and we were expecting a Box, ergo Pin<Box<expected>>, we
                // can suggest Box::pin.
                let parent = self.tcx.hir().get_parent_node(expr.hir_id);
                let fn_name = match self.tcx.hir().find(parent) {
                    Some(Node::Expr(Expr { kind: ExprKind::Call(fn_name, _), .. })) => fn_name,
                    _ => return false,
                };
                match fn_name.kind {
                    ExprKind::Path(QPath::TypeRelative(
                        hir::Ty {
                            kind: TyKind::Path(QPath::Resolved(_, Path { res: recv_ty, .. })),
                            ..
                        },
                        method,
                    )) if recv_ty.opt_def_id() == pin_did && method.ident.name == sym::new => {
                        err.span_suggestion(
                            fn_name.span,
                            "use `Box::pin` to pin and box this expression",
                            "Box::pin".to_string(),
                            Applicability::MachineApplicable,
                        );
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    /// A common error is to forget to add a semicolon at the end of a block, e.g.,
    ///
    /// ```
    /// fn foo() {
    ///     bar_that_returns_u32()
    /// }
    /// ```
    ///
    /// This routine checks if the return expression in a block would make sense on its own as a
    /// statement and the return type has been left as default or has been specified as `()`. If so,
    /// it suggests adding a semicolon.
    fn suggest_missing_semicolon(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expression: &'tcx hir::Expr<'tcx>,
        expected: Ty<'tcx>,
    ) {
        if expected.is_unit() {
            // `BlockTailExpression` only relevant if the tail expr would be
            // useful on its own.
            match expression.kind {
                ExprKind::Call(..)
                | ExprKind::MethodCall(..)
                | ExprKind::Loop(..)
                | ExprKind::If(..)
                | ExprKind::Match(..)
                | ExprKind::Block(..)
                    if expression.can_have_side_effects() =>
                {
                    err.span_suggestion(
                        expression.span.shrink_to_hi(),
                        "consider using a semicolon here",
                        ";".to_string(),
                        Applicability::MachineApplicable,
                    );
                }
                _ => (),
            }
        }
    }

    /// A possible error is to forget to add a return type that is needed:
    ///
    /// ```
    /// fn foo() {
    ///     bar_that_returns_u32()
    /// }
    /// ```
    ///
    /// This routine checks if the return type is left as default, the method is not part of an
    /// `impl` block and that it isn't the `main` method. If so, it suggests setting the return
    /// type.
    pub(in super::super) fn suggest_missing_return_type(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        fn_decl: &hir::FnDecl<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        can_suggest: bool,
        fn_id: hir::HirId,
    ) -> bool {
        // Only suggest changing the return type for methods that
        // haven't set a return type at all (and aren't `fn main()` or an impl).
        match (&fn_decl.output, found.is_suggestable(), can_suggest, expected.is_unit()) {
            (&hir::FnRetTy::DefaultReturn(span), true, true, true) => {
                err.span_suggestion(
                    span,
                    "try adding a return type",
                    format!("-> {} ", self.resolve_vars_with_obligations(found)),
                    Applicability::MachineApplicable,
                );
                true
            }
            (&hir::FnRetTy::DefaultReturn(span), false, true, true) => {
                err.span_label(span, "possibly return type missing here?");
                true
            }
            (&hir::FnRetTy::DefaultReturn(span), _, false, true) => {
                // `fn main()` must return `()`, do not suggest changing return type
                err.span_label(span, "expected `()` because of default return type");
                true
            }
            // expectation was caused by something else, not the default return
            (&hir::FnRetTy::DefaultReturn(_), _, _, false) => false,
            (&hir::FnRetTy::Return(ref ty), _, _, _) => {
                // Only point to return type if the expected type is the return type, as if they
                // are not, the expectation must have been caused by something else.
                debug!("suggest_missing_return_type: return type {:?} node {:?}", ty, ty.kind);
                let sp = ty.span;
                let ty = <dyn AstConv<'_>>::ast_ty_to_ty(self, ty);
                debug!("suggest_missing_return_type: return type {:?}", ty);
                debug!("suggest_missing_return_type: expected type {:?}", ty);
                let bound_vars = self.tcx.late_bound_vars(fn_id);
                let ty = Binder::bind_with_vars(ty, bound_vars);
                let ty = self.normalize_associated_types_in(sp, ty);
                let ty = self.tcx.erase_late_bound_regions(ty);
                if self.can_coerce(expected, ty) {
                    err.span_label(sp, format!("expected `{}` because of return type", expected));
                    self.try_suggest_return_impl_trait(err, expected, ty, fn_id);
                    return true;
                }
                false
            }
        }
    }

    /// check whether the return type is a generic type with a trait bound
    /// only suggest this if the generic param is not present in the arguments
    /// if this is true, hint them towards changing the return type to `impl Trait`
    /// ```
    /// fn cant_name_it<T: Fn() -> u32>() -> T {
    ///     || 3
    /// }
    /// ```
    fn try_suggest_return_impl_trait(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        fn_id: hir::HirId,
    ) {
        // Only apply the suggestion if:
        //  - the return type is a generic parameter
        //  - the generic param is not used as a fn param
        //  - the generic param has at least one bound
        //  - the generic param doesn't appear in any other bounds where it's not the Self type
        // Suggest:
        //  - Changing the return type to be `impl <all bounds>`

        debug!("try_suggest_return_impl_trait, expected = {:?}, found = {:?}", expected, found);

        let ty::Param(expected_ty_as_param) = expected.kind() else { return };

        let fn_node = self.tcx.hir().find(fn_id);

        let Some(hir::Node::Item(hir::Item {
            kind:
                hir::ItemKind::Fn(
                    hir::FnSig { decl: hir::FnDecl { inputs: fn_parameters, output: fn_return, .. }, .. },
                    hir::Generics { params, where_clause, .. },
                    _body_id,
                ),
            ..
        })) = fn_node else { return };

        let Some(expected_generic_param) = params.get(expected_ty_as_param.index as usize) else { return };

        // get all where BoundPredicates here, because they are used in to cases below
        let where_predicates = where_clause
            .predicates
            .iter()
            .filter_map(|p| match p {
                WherePredicate::BoundPredicate(hir::WhereBoundPredicate {
                    bounds,
                    bounded_ty,
                    ..
                }) => {
                    // FIXME: Maybe these calls to `ast_ty_to_ty` can be removed (and the ones below)
                    let ty = <dyn AstConv<'_>>::ast_ty_to_ty(self, bounded_ty);
                    Some((ty, bounds))
                }
                _ => None,
            })
            .map(|(ty, bounds)| match ty.kind() {
                ty::Param(param_ty) if param_ty == expected_ty_as_param => Ok(Some(bounds)),
                // check whether there is any predicate that contains our `T`, like `Option<T>: Send`
                _ => match ty.contains(expected) {
                    true => Err(()),
                    false => Ok(None),
                },
            })
            .collect::<Result<Vec<_>, _>>();

        let Ok(where_predicates) =  where_predicates else { return };

        // now get all predicates in the same types as the where bounds, so we can chain them
        let predicates_from_where =
            where_predicates.iter().flatten().map(|bounds| bounds.iter()).flatten();

        // extract all bounds from the source code using their spans
        let all_matching_bounds_strs = expected_generic_param
            .bounds
            .iter()
            .chain(predicates_from_where)
            .filter_map(|bound| match bound {
                GenericBound::Trait(_, _) => {
                    self.tcx.sess.source_map().span_to_snippet(bound.span()).ok()
                }
                _ => None,
            })
            .collect::<Vec<String>>();

        if all_matching_bounds_strs.len() == 0 {
            return;
        }

        let all_bounds_str = all_matching_bounds_strs.join(" + ");

        let ty_param_used_in_fn_params = fn_parameters.iter().any(|param| {
                let ty = <dyn AstConv<'_>>::ast_ty_to_ty(self, param);
                matches!(ty.kind(), ty::Param(fn_param_ty_param) if expected_ty_as_param == fn_param_ty_param)
            });

        if ty_param_used_in_fn_params {
            return;
        }

        err.span_suggestion(
            fn_return.span(),
            "consider using an impl return type",
            format!("impl {}", all_bounds_str),
            Applicability::MaybeIncorrect,
        );
    }

    pub(in super::super) fn suggest_missing_break_or_return_expr(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &'tcx hir::Expr<'tcx>,
        fn_decl: &hir::FnDecl<'_>,
        expected: Ty<'tcx>,
        found: Ty<'tcx>,
        id: hir::HirId,
        fn_id: hir::HirId,
    ) {
        if !expected.is_unit() {
            return;
        }
        let found = self.resolve_vars_with_obligations(found);

        let in_loop = self.is_loop(id)
            || self.tcx.hir().parent_iter(id).any(|(parent_id, _)| self.is_loop(parent_id));

        let in_local_statement = self.is_local_statement(id)
            || self
                .tcx
                .hir()
                .parent_iter(id)
                .any(|(parent_id, _)| self.is_local_statement(parent_id));

        if in_loop && in_local_statement {
            err.multipart_suggestion(
                "you might have meant to break the loop with this value",
                vec![
                    (expr.span.shrink_to_lo(), "break ".to_string()),
                    (expr.span.shrink_to_hi(), ";".to_string()),
                ],
                Applicability::MaybeIncorrect,
            );
            return;
        }

        if let hir::FnRetTy::Return(ty) = fn_decl.output {
            let ty = <dyn AstConv<'_>>::ast_ty_to_ty(self, ty);
            let bound_vars = self.tcx.late_bound_vars(fn_id);
            let ty = self.tcx.erase_late_bound_regions(Binder::bind_with_vars(ty, bound_vars));
            let ty = self.normalize_associated_types_in(expr.span, ty);
            let ty = match self.tcx.asyncness(fn_id.owner) {
                hir::IsAsync::Async => self
                    .tcx
                    .infer_ctxt()
                    .enter(|infcx| {
                        infcx.get_impl_future_output_ty(ty).unwrap_or_else(|| {
                            span_bug!(
                                fn_decl.output.span(),
                                "failed to get output type of async function"
                            )
                        })
                    })
                    .skip_binder(),
                hir::IsAsync::NotAsync => ty,
            };
            if self.can_coerce(found, ty) {
                err.multipart_suggestion(
                    "you might have meant to return this value",
                    vec![
                        (expr.span.shrink_to_lo(), "return ".to_string()),
                        (expr.span.shrink_to_hi(), ";".to_string()),
                    ],
                    Applicability::MaybeIncorrect,
                );
            }
        }
    }

    pub(in super::super) fn suggest_missing_parentheses(
        &self,
        err: &mut DiagnosticBuilder<'_>,
        expr: &hir::Expr<'_>,
    ) {
        let sp = self.tcx.sess.source_map().start_point(expr.span);
        if let Some(sp) = self.tcx.sess.parse_sess.ambiguous_block_expr_parse.borrow().get(&sp) {
            // `{ 42 } &&x` (#61475) or `{ 42 } && if x { 1 } else { 0 }`
            self.tcx.sess.parse_sess.expr_parentheses_needed(err, *sp);
        }
    }

    fn is_loop(&self, id: hir::HirId) -> bool {
        let node = self.tcx.hir().get(id);
        matches!(node, Node::Expr(Expr { kind: ExprKind::Loop(..), .. }))
    }

    fn is_local_statement(&self, id: hir::HirId) -> bool {
        let node = self.tcx.hir().get(id);
        matches!(node, Node::Stmt(Stmt { kind: StmtKind::Local(..), .. }))
    }
}
