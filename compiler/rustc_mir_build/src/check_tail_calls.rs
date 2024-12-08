use rustc_abi::ExternAbi;
use rustc_errors::Applicability;
use rustc_hir::LangItem;
use rustc_hir::def::DefKind;
use rustc_middle::span_bug;
use rustc_middle::thir::visit::{self, Visitor};
use rustc_middle::thir::{BodyTy, Expr, ExprId, ExprKind, Thir};
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_span::def_id::{DefId, LocalDefId};
use rustc_span::{DUMMY_SP, ErrorGuaranteed, Span};

pub(crate) fn check_tail_calls(tcx: TyCtxt<'_>, def: LocalDefId) -> Result<(), ErrorGuaranteed> {
    let (thir, expr) = tcx.thir_body(def)?;
    let thir = &thir.borrow();

    // If `thir` is empty, a type error occurred, skip this body.
    if thir.exprs.is_empty() {
        return Ok(());
    }

    let is_closure = matches!(tcx.def_kind(def), DefKind::Closure);
    let caller_ty = tcx.type_of(def).skip_binder();

    let mut visitor = TailCallCkVisitor {
        tcx,
        thir,
        found_errors: Ok(()),
        // FIXME(#132279): we're clearly in a body here.
        typing_env: ty::TypingEnv::non_body_analysis(tcx, def),
        is_closure,
        caller_ty,
    };

    visitor.visit_expr(&thir[expr]);

    visitor.found_errors
}

struct TailCallCkVisitor<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    thir: &'a Thir<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    /// Whatever the currently checked body is one of a closure
    is_closure: bool,
    /// The result of the checks, `Err(_)` if there was a problem with some
    /// tail call, `Ok(())` if all of them were fine.
    found_errors: Result<(), ErrorGuaranteed>,
    /// Type of the caller function.
    caller_ty: Ty<'tcx>,
}

impl<'tcx> TailCallCkVisitor<'_, 'tcx> {
    fn check_tail_call(&mut self, call: &Expr<'_>, expr: &Expr<'_>) {
        if self.is_closure {
            self.report_in_closure(expr);
            return;
        }

        let BodyTy::Fn(caller_sig) = self.thir.body_type else {
            span_bug!(
                call.span,
                "`become` outside of functions should have been disallowed by hit_typeck"
            )
        };

        let ExprKind::Scope { value, .. } = call.kind else {
            span_bug!(call.span, "expected scope, found: {call:?}")
        };
        let value = &self.thir[value];

        if matches!(
            value.kind,
            ExprKind::Binary { .. }
                | ExprKind::Unary { .. }
                | ExprKind::AssignOp { .. }
                | ExprKind::Index { .. }
        ) {
            self.report_builtin_op(call, expr);
            return;
        }

        let ExprKind::Call { ty, fun, ref args, from_hir_call, fn_span } = value.kind else {
            self.report_non_call(value, expr);
            return;
        };

        if !from_hir_call {
            self.report_op(ty, args, fn_span, expr);
        }

        // Closures in thir look something akin to
        // `for<'a> extern "rust-call" fn(&'a [closure@...], ()) -> <[closure@...] as FnOnce<()>>::Output {<[closure@...] as Fn<()>>::call}`
        // So we have to check for them in this weird way...
        if let &ty::FnDef(did, args) = ty.kind() {
            let parent = self.tcx.parent(did);
            if self.tcx.fn_trait_kind_from_def_id(parent).is_some()
                && args.first().and_then(|arg| arg.as_type()).is_some_and(Ty::is_closure)
            {
                self.report_calling_closure(&self.thir[fun], args[1].as_type().unwrap(), expr);

                // Tail calling is likely to cause unrelated errors (ABI, argument mismatches),
                // skip them, producing an error about calling a closure is enough.
                return;
            };
        }

        // Erase regions since tail calls don't care about lifetimes
        let callee_sig =
            self.tcx.normalize_erasing_late_bound_regions(self.typing_env, ty.fn_sig(self.tcx));

        if caller_sig.abi != callee_sig.abi {
            self.report_abi_mismatch(expr.span, caller_sig.abi, callee_sig.abi);
        }

        if caller_sig.inputs_and_output != callee_sig.inputs_and_output {
            if caller_sig.inputs() != callee_sig.inputs() {
                self.report_arguments_mismatch(expr.span, caller_sig, callee_sig);
            }

            // FIXME(explicit_tail_calls): this currenly fails for cases where opaques are used.
            // e.g.
            // ```
            // fn a() -> impl Sized { become b() } // ICE
            // fn b() -> u8 { 0 }
            // ```
            // we should think what is the expected behavior here.
            // (we should probably just accept this by revealing opaques?)
            if caller_sig.output() != callee_sig.output() {
                span_bug!(expr.span, "hir typeck should have checked the return type already");
            }
        }

        {
            let caller_needs_location = self.needs_location(self.caller_ty);
            let callee_needs_location = self.needs_location(ty);

            if caller_needs_location != callee_needs_location {
                self.report_track_caller_mismatch(expr.span, caller_needs_location);
            }
        }

        if caller_sig.c_variadic {
            self.report_c_variadic_caller(expr.span);
        }

        if callee_sig.c_variadic {
            self.report_c_variadic_callee(expr.span);
        }
    }

    /// Returns true if function of type `ty` needs location argument
    /// (i.e. if a function is marked as `#[track_caller]`)
    fn needs_location(&self, ty: Ty<'tcx>) -> bool {
        if let &ty::FnDef(did, substs) = ty.kind() {
            let instance =
                ty::Instance::expect_resolve(self.tcx, self.typing_env, did, substs, DUMMY_SP);

            instance.def.requires_caller_location(self.tcx)
        } else {
            false
        }
    }

    fn report_in_closure(&mut self, expr: &Expr<'_>) {
        let err = self.tcx.dcx().span_err(expr.span, "`become` is not allowed in closures");
        self.found_errors = Err(err);
    }

    fn report_builtin_op(&mut self, value: &Expr<'_>, expr: &Expr<'_>) {
        let err = self
            .tcx
            .dcx()
            .struct_span_err(value.span, "`become` does not support operators")
            .with_note("using `become` on a builtin operator is not useful")
            .with_span_suggestion(
                value.span.until(expr.span),
                "try using `return` instead",
                "return ",
                Applicability::MachineApplicable,
            )
            .emit();
        self.found_errors = Err(err);
    }

    fn report_op(&mut self, fun_ty: Ty<'_>, args: &[ExprId], fn_span: Span, expr: &Expr<'_>) {
        let mut err =
            self.tcx.dcx().struct_span_err(fn_span, "`become` does not support operators");

        if let &ty::FnDef(did, _substs) = fun_ty.kind()
            && let parent = self.tcx.parent(did)
            && matches!(self.tcx.def_kind(parent), DefKind::Trait)
            && let Some(method) = op_trait_as_method_name(self.tcx, parent)
        {
            match args {
                &[arg] => {
                    let arg = &self.thir[arg];

                    err.multipart_suggestion(
                        "try using the method directly",
                        vec![
                            (fn_span.shrink_to_lo().until(arg.span), "(".to_owned()),
                            (arg.span.shrink_to_hi(), format!(").{method}()")),
                        ],
                        Applicability::MaybeIncorrect,
                    );
                }
                &[lhs, rhs] => {
                    let lhs = &self.thir[lhs];
                    let rhs = &self.thir[rhs];

                    err.multipart_suggestion(
                        "try using the method directly",
                        vec![
                            (lhs.span.shrink_to_lo(), format!("(")),
                            (lhs.span.between(rhs.span), format!(").{method}(")),
                            (rhs.span.between(expr.span.shrink_to_hi()), ")".to_owned()),
                        ],
                        Applicability::MaybeIncorrect,
                    );
                }
                _ => span_bug!(expr.span, "operator with more than 2 args? {args:?}"),
            }
        }

        self.found_errors = Err(err.emit());
    }

    fn report_non_call(&mut self, value: &Expr<'_>, expr: &Expr<'_>) {
        let err = self
            .tcx
            .dcx()
            .struct_span_err(value.span, "`become` requires a function call")
            .with_span_note(value.span, "not a function call")
            .with_span_suggestion(
                value.span.until(expr.span),
                "try using `return` instead",
                "return ",
                Applicability::MaybeIncorrect,
            )
            .emit();
        self.found_errors = Err(err);
    }

    fn report_calling_closure(&mut self, fun: &Expr<'_>, tupled_args: Ty<'_>, expr: &Expr<'_>) {
        let underscored_args = match tupled_args.kind() {
            ty::Tuple(tys) if tys.is_empty() => "".to_owned(),
            ty::Tuple(tys) => std::iter::repeat("_, ").take(tys.len() - 1).chain(["_"]).collect(),
            _ => "_".to_owned(),
        };

        let err = self
            .tcx
            .dcx()
            .struct_span_err(expr.span, "tail calling closures directly is not allowed")
            .with_multipart_suggestion(
                "try casting the closure to a function pointer type",
                vec![
                    (fun.span.shrink_to_lo(), "(".to_owned()),
                    (fun.span.shrink_to_hi(), format!(" as fn({underscored_args}) -> _)")),
                ],
                Applicability::MaybeIncorrect,
            )
            .emit();
        self.found_errors = Err(err);
    }

    fn report_abi_mismatch(&mut self, sp: Span, caller_abi: ExternAbi, callee_abi: ExternAbi) {
        let err = self
            .tcx
            .dcx()
            .struct_span_err(sp, "mismatched function ABIs")
            .with_note("`become` requires caller and callee to have the same ABI")
            .with_note(format!("caller ABI is `{caller_abi}`, while callee ABI is `{callee_abi}`"))
            .emit();
        self.found_errors = Err(err);
    }

    fn report_arguments_mismatch(
        &mut self,
        sp: Span,
        caller_sig: ty::FnSig<'_>,
        callee_sig: ty::FnSig<'_>,
    ) {
        let err = self
            .tcx
            .dcx()
            .struct_span_err(sp, "mismatched signatures")
            .with_note("`become` requires caller and callee to have matching signatures")
            .with_note(format!("caller signature: `{caller_sig}`"))
            .with_note(format!("callee signature: `{callee_sig}`"))
            .emit();
        self.found_errors = Err(err);
    }

    fn report_track_caller_mismatch(&mut self, sp: Span, caller_needs_location: bool) {
        let err = match caller_needs_location {
            true => self
                .tcx
                .dcx()
                .struct_span_err(
                    sp,
                    "a function marked with `#[track_caller]` cannot tail-call one that is not",
                )
                .emit(),
            false => self
                .tcx
                .dcx()
                .struct_span_err(
                    sp,
                    "a function mot marked with `#[track_caller]` cannot tail-call one that is",
                )
                .emit(),
        };

        self.found_errors = Err(err);
    }

    fn report_c_variadic_caller(&mut self, sp: Span) {
        let err = self
            .tcx
            .dcx()
            // FIXME(explicit_tail_calls): highlight the `...`
            .struct_span_err(sp, "tail-calls are not allowed in c-variadic functions")
            .emit();

        self.found_errors = Err(err);
    }

    fn report_c_variadic_callee(&mut self, sp: Span) {
        let err = self
            .tcx
            .dcx()
            // FIXME(explicit_tail_calls): highlight the function or something...
            .struct_span_err(sp, "c-variadic functions can't be tail-called")
            .emit();

        self.found_errors = Err(err);
    }
}

impl<'a, 'tcx> Visitor<'a, 'tcx> for TailCallCkVisitor<'a, 'tcx> {
    fn thir(&self) -> &'a Thir<'tcx> {
        &self.thir
    }

    fn visit_expr(&mut self, expr: &'a Expr<'tcx>) {
        if let ExprKind::Become { value } = expr.kind {
            let call = &self.thir[value];
            self.check_tail_call(call, expr);
        }

        visit::walk_expr(self, expr);
    }
}

fn op_trait_as_method_name(tcx: TyCtxt<'_>, trait_did: DefId) -> Option<&'static str> {
    let m = match tcx.as_lang_item(trait_did)? {
        LangItem::Add => "add",
        LangItem::Sub => "sub",
        LangItem::Mul => "mul",
        LangItem::Div => "div",
        LangItem::Rem => "rem",
        LangItem::Neg => "neg",
        LangItem::Not => "not",
        LangItem::BitXor => "bitxor",
        LangItem::BitAnd => "bitand",
        LangItem::BitOr => "bitor",
        LangItem::Shl => "shl",
        LangItem::Shr => "shr",
        LangItem::AddAssign => "add_assign",
        LangItem::SubAssign => "sub_assign",
        LangItem::MulAssign => "mul_assign",
        LangItem::DivAssign => "div_assign",
        LangItem::RemAssign => "rem_assign",
        LangItem::BitXorAssign => "bitxor_assign",
        LangItem::BitAndAssign => "bitand_assign",
        LangItem::BitOrAssign => "bitor_assign",
        LangItem::ShlAssign => "shl_assign",
        LangItem::ShrAssign => "shr_assign",
        LangItem::Index => "index",
        LangItem::IndexMut => "index_mut",
        _ => return None,
    };

    Some(m)
}
