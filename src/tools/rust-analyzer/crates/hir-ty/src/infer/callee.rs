//! Inference of calls.

use std::iter;

use intern::sym;
use tracing::debug;

use hir_def::{CallableDefId, hir::ExprId, signatures::FunctionSignature};
use rustc_type_ir::{
    InferTy, Interner,
    inherent::{GenericArgs as _, IntoKind, Ty as _},
};

use crate::{
    Adjust, Adjustment, AutoBorrow, FnAbi,
    autoderef::{GeneralAutoderef, InferenceContextAutoderef},
    infer::{
        AllowTwoPhase, AutoBorrowMutability, Expectation, InferenceContext, InferenceDiagnostic,
        expr::{ExprIsRead, TupleArgumentsFlag},
    },
    method_resolution::{MethodCallee, TreatNotYetDefinedOpaques},
    next_solver::{
        FnSig, Ty, TyKind,
        infer::{BoundRegionConversionTime, traits::ObligationCause},
    },
};

#[derive(Debug)]
enum CallStep<'db> {
    Builtin(Ty<'db>),
    DeferredClosure(ExprId, FnSig<'db>),
    /// Call overloading when callee implements one of the Fn* traits.
    Overloaded(MethodCallee<'db>),
}

impl<'db> InferenceContext<'_, 'db> {
    pub(crate) fn infer_call(
        &mut self,
        call_expr: ExprId,
        callee_expr: ExprId,
        arg_exprs: &[ExprId],
        expected: &Expectation<'db>,
    ) -> Ty<'db> {
        let original_callee_ty = self.infer_expr_no_expect(callee_expr, ExprIsRead::Yes);

        let expr_ty = self.table.try_structurally_resolve_type(original_callee_ty);

        let mut autoderef = GeneralAutoderef::new_from_inference_context(self, expr_ty);
        let mut result = None;
        while result.is_none() && autoderef.next().is_some() {
            result =
                Self::try_overloaded_call_step(call_expr, callee_expr, arg_exprs, &mut autoderef);
        }

        // FIXME: rustc does some ABI checks here, but the ABI mapping is in rustc_target and we don't have access to that crate.

        let obligations = autoderef.take_obligations();
        self.table.register_predicates(obligations);

        let output = match result {
            None => {
                // Check all of the arg expressions, but with no expectations
                // since we don't have a signature to compare them to.
                for &arg in arg_exprs {
                    self.infer_expr_no_expect(arg, ExprIsRead::Yes);
                }

                self.push_diagnostic(InferenceDiagnostic::ExpectedFunction {
                    call_expr,
                    found: original_callee_ty.store(),
                });

                self.types.types.error
            }

            Some(CallStep::Builtin(callee_ty)) => {
                self.confirm_builtin_call(call_expr, callee_ty, arg_exprs, expected)
            }

            Some(CallStep::DeferredClosure(_def_id, fn_sig)) => {
                self.confirm_deferred_closure_call(call_expr, arg_exprs, expected, fn_sig)
            }

            Some(CallStep::Overloaded(method_callee)) => {
                self.confirm_overloaded_call(call_expr, arg_exprs, expected, method_callee)
            }
        };

        // we must check that return type of called functions is WF:
        self.table.register_wf_obligation(output.into(), ObligationCause::new());

        output
    }

    fn try_overloaded_call_step(
        call_expr: ExprId,
        callee_expr: ExprId,
        arg_exprs: &[ExprId],
        autoderef: &mut InferenceContextAutoderef<'_, '_, 'db>,
    ) -> Option<CallStep<'db>> {
        let final_ty = autoderef.final_ty();
        let adjusted_ty = autoderef.ctx().table.try_structurally_resolve_type(final_ty);

        // If the callee is a function pointer or a closure, then we're all set.
        match adjusted_ty.kind() {
            TyKind::FnDef(..) | TyKind::FnPtr(..) => {
                let adjust_steps = autoderef.adjust_steps_as_infer_ok();
                let adjustments =
                    autoderef.ctx().table.register_infer_ok(adjust_steps).into_boxed_slice();
                autoderef.ctx().write_expr_adj(callee_expr, adjustments);
                return Some(CallStep::Builtin(adjusted_ty));
            }

            // Check whether this is a call to a closure where we
            // haven't yet decided on whether the closure is fn vs
            // fnmut vs fnonce. If so, we have to defer further processing.
            TyKind::Closure(def_id, args)
                if autoderef.ctx().infcx().closure_kind(adjusted_ty).is_none() =>
            {
                let closure_sig = args.as_closure().sig();
                let closure_sig = autoderef.ctx().infcx().instantiate_binder_with_fresh_vars(
                    BoundRegionConversionTime::FnCall,
                    closure_sig,
                );
                let adjust_steps = autoderef.adjust_steps_as_infer_ok();
                let adjustments = autoderef.ctx().table.register_infer_ok(adjust_steps);
                let def_id = def_id.0.loc(autoderef.ctx().db).1;
                autoderef.ctx().record_deferred_call_resolution(
                    def_id,
                    DeferredCallResolution {
                        call_expr,
                        callee_expr,
                        closure_ty: adjusted_ty,
                        adjustments,
                        fn_sig: closure_sig,
                    },
                );
                return Some(CallStep::DeferredClosure(def_id, closure_sig));
            }

            // When calling a `CoroutineClosure` that is local to the body, we will
            // not know what its `closure_kind` is yet. Instead, just fill in the
            // signature with an infer var for the `tupled_upvars_ty` of the coroutine,
            // and record a deferred call resolution which will constrain that var
            // as part of `AsyncFn*` trait confirmation.
            TyKind::CoroutineClosure(def_id, args)
                if autoderef.ctx().infcx().closure_kind(adjusted_ty).is_none() =>
            {
                let closure_args = args.as_coroutine_closure();
                let coroutine_closure_sig =
                    autoderef.ctx().infcx().instantiate_binder_with_fresh_vars(
                        BoundRegionConversionTime::FnCall,
                        closure_args.coroutine_closure_sig(),
                    );
                let tupled_upvars_ty = autoderef.ctx().table.next_ty_var();
                // We may actually receive a coroutine back whose kind is different
                // from the closure that this dispatched from. This is because when
                // we have no captures, we automatically implement `FnOnce`. This
                // impl forces the closure kind to `FnOnce` i.e. `u8`.
                let kind_ty = autoderef.ctx().table.next_ty_var();
                let interner = autoderef.ctx().interner();

                // Ignore splatting, it is unsupported on closures.
                let call_sig = interner.mk_fn_sig(
                    [coroutine_closure_sig.tupled_inputs_ty],
                    coroutine_closure_sig.to_coroutine(
                        interner,
                        closure_args.parent_args(),
                        kind_ty,
                        interner.coroutine_for_closure(def_id),
                        tupled_upvars_ty,
                    ),
                    coroutine_closure_sig.c_variadic,
                    coroutine_closure_sig.safety,
                    coroutine_closure_sig.abi,
                );
                let adjust_steps = autoderef.adjust_steps_as_infer_ok();
                let adjustments = autoderef.ctx().table.register_infer_ok(adjust_steps);
                let def_id = def_id.0.loc(autoderef.ctx().db).1;
                autoderef.ctx().record_deferred_call_resolution(
                    def_id,
                    DeferredCallResolution {
                        call_expr,
                        callee_expr,
                        closure_ty: adjusted_ty,
                        adjustments,
                        fn_sig: call_sig,
                    },
                );
                return Some(CallStep::DeferredClosure(def_id, call_sig));
            }

            // Hack: we know that there are traits implementing Fn for &F
            // where F:Fn and so forth. In the particular case of types
            // like `f: &mut FnMut()`, if there is a call `f()`, we would
            // normally translate to `FnMut::call_mut(&mut f, ())`, but
            // that winds up potentially requiring the user to mark their
            // variable as `mut` which feels unnecessary and unexpected.
            //
            //     fn foo(f: &mut impl FnMut()) { f() }
            //            ^ without this hack `f` would have to be declared as mutable
            //
            // The simplest fix by far is to just ignore this case and deref again,
            // so we wind up with `FnMut::call_mut(&mut *f, ())`.
            TyKind::Ref(..) if autoderef.step_count() == 0 => {
                return None;
            }

            TyKind::Infer(InferTy::TyVar(vid))
                // If we end up with an inference variable which is not the hidden type of
                // an opaque, emit an error.
                if !autoderef.ctx().infcx().has_opaques_with_sub_unified_hidden_type(vid) => {
                    autoderef
                        .ctx()
                        .type_must_be_known_at_this_point(callee_expr.into(), adjusted_ty);
                    return None;
                }

            TyKind::Error(_) => {
                return None;
            }

            _ => {}
        }

        // Now, we look for the implementation of a Fn trait on the object's type.
        // We first do it with the explicit instruction to look for an impl of
        // `Fn<Tuple>`, with the tuple `Tuple` having an arity corresponding
        // to the number of call parameters.
        // If that fails (or_else branch), we try again without specifying the
        // shape of the tuple (hence the None). This allows to detect an Fn trait
        // is implemented, and use this information for diagnostic.
        autoderef
            .ctx()
            .try_overloaded_call_traits(adjusted_ty, Some(arg_exprs))
            .or_else(|| autoderef.ctx().try_overloaded_call_traits(adjusted_ty, None))
            .map(|(autoref, method)| {
                let adjustments = autoderef.adjust_steps_as_infer_ok();
                let mut adjustments = autoderef.ctx().table.register_infer_ok(adjustments);
                adjustments.extend(autoref);
                autoderef.ctx().write_expr_adj(callee_expr, adjustments.into_boxed_slice());
                CallStep::Overloaded(method)
            })
    }

    fn try_overloaded_call_traits(
        &mut self,
        adjusted_ty: Ty<'db>,
        opt_arg_exprs: Option<&[ExprId]>,
    ) -> Option<(Option<Adjustment>, MethodCallee<'db>)> {
        // HACK(async_closures): For async closures, prefer `AsyncFn*`
        // over `Fn*`, since all async closures implement `FnOnce`, but
        // choosing that over `AsyncFn`/`AsyncFnMut` would be more restrictive.
        // For other callables, just prefer `Fn*` for perf reasons.
        //
        // The order of trait choices here is not that big of a deal,
        // since it just guides inference (and our choice of autoref).
        // Though in the future, I'd like typeck to choose:
        // `Fn > AsyncFn > FnMut > AsyncFnMut > FnOnce > AsyncFnOnce`
        // ...or *ideally*, we just have `LendingFn`/`LendingFnMut`, which
        // would naturally unify these two trait hierarchies in the most
        // general way.
        let call_trait_choices = if self.shallow_resolve(adjusted_ty).is_coroutine_closure() {
            [
                (self.lang_items.AsyncFn, sym::async_call, true),
                (self.lang_items.AsyncFnMut, sym::async_call_mut, true),
                (self.lang_items.AsyncFnOnce, sym::async_call_once, false),
                (self.lang_items.Fn, sym::call, true),
                (self.lang_items.FnMut, sym::call_mut, true),
                (self.lang_items.FnOnce, sym::call_once, false),
            ]
        } else {
            [
                (self.lang_items.Fn, sym::call, true),
                (self.lang_items.FnMut, sym::call_mut, true),
                (self.lang_items.FnOnce, sym::call_once, false),
                (self.lang_items.AsyncFn, sym::async_call, true),
                (self.lang_items.AsyncFnMut, sym::async_call_mut, true),
                (self.lang_items.AsyncFnOnce, sym::async_call_once, false),
            ]
        };

        // Try the options that are least restrictive on the caller first.
        for (opt_trait_def_id, method_name, borrow) in call_trait_choices {
            let Some(trait_def_id) = opt_trait_def_id else {
                continue;
            };

            let opt_input_type = opt_arg_exprs.map(|arg_exprs| {
                Ty::new_tup_from_iter(
                    self.interner(),
                    arg_exprs.iter().map(|_| self.table.next_ty_var()),
                )
            });

            // We use `TreatNotYetDefinedOpaques::AsRigid` here so that if the `adjusted_ty`
            // is `Box<impl FnOnce()>` we choose  `FnOnce` instead of `Fn`.
            //
            // We try all the different call traits in order and choose the first
            // one which may apply. So if we treat opaques as inference variables
            // `Box<impl FnOnce()>: Fn` is considered ambiguous and chosen.
            if let Some(ok) = self.table.lookup_method_for_operator(
                ObligationCause::new(),
                method_name,
                trait_def_id,
                adjusted_ty,
                opt_input_type,
                TreatNotYetDefinedOpaques::AsRigid,
            ) {
                let method = self.table.register_infer_ok(ok);
                let mut autoref = None;
                if borrow {
                    // Check for &self vs &mut self in the method signature. Since this is either
                    // the Fn or FnMut trait, it should be one of those.
                    let TyKind::Ref(_, _, mutbl) = method.sig.inputs_and_output.inputs()[0].kind()
                    else {
                        panic!("Expected `FnMut`/`Fn` to take receiver by-ref/by-mut")
                    };

                    // For initial two-phase borrow
                    // deployment, conservatively omit
                    // overloaded function call ops.
                    let mutbl = AutoBorrowMutability::new(mutbl, AllowTwoPhase::No);

                    autoref = Some(Adjustment {
                        kind: Adjust::Borrow(AutoBorrow::Ref(mutbl)),
                        target: method.sig.inputs_and_output.inputs()[0].store(),
                    });
                }

                return Some((autoref, method));
            }
        }

        None
    }

    /// Returns the argument indices to skip.
    fn check_legacy_const_generics(
        &mut self,
        callee: Option<CallableDefId>,
        args: &[ExprId],
    ) -> Box<[u32]> {
        let func = match callee {
            Some(CallableDefId::FunctionId(func)) => func,
            _ => return Default::default(),
        };

        let data = FunctionSignature::of(self.db, func);
        let Some(legacy_const_generics_indices) = data.legacy_const_generics_indices(self.db, func)
        else {
            return Default::default();
        };
        let mut legacy_const_generics_indices = Box::<[u32]>::from(legacy_const_generics_indices);

        // only use legacy const generics if the param count matches with them
        if data.params.len() + legacy_const_generics_indices.len() != args.len() {
            if args.len() <= data.params.len() {
                return Default::default();
            } else {
                // there are more parameters than there should be without legacy
                // const params; use them
                legacy_const_generics_indices.sort_unstable();
                return legacy_const_generics_indices;
            }
        }

        // check legacy const parameters
        for arg_idx in legacy_const_generics_indices.iter().copied() {
            if arg_idx >= args.len() as u32 {
                continue;
            }
            let expected = Expectation::none(); // FIXME use actual const ty, when that is lowered correctly
            self.infer_expr(args[arg_idx as usize], &expected, ExprIsRead::Yes);
            // FIXME: evaluate and unify with the const
        }
        legacy_const_generics_indices.sort_unstable();
        legacy_const_generics_indices
    }

    fn confirm_builtin_call(
        &mut self,
        call_expr: ExprId,
        callee_ty: Ty<'db>,
        arg_exprs: &[ExprId],
        expected: &Expectation<'db>,
    ) -> Ty<'db> {
        let (fn_sig, def_id) = match callee_ty.kind() {
            TyKind::FnDef(def_id, args) => {
                let fn_sig =
                    self.db.callable_item_signature(def_id.0).instantiate(self.interner(), args);
                (fn_sig, Some(def_id.0))
            }

            // FIXME(const_trait_impl): these arms should error because we can't enforce them
            TyKind::FnPtr(sig_tys, hdr) => (sig_tys.with(hdr), None),

            _ => unreachable!(),
        };

        // Replace any late-bound regions that appear in the function
        // signature with region variables. We also have to
        // renormalize the associated types at this point, since they
        // previously appeared within a `Binder<>` and hence would not
        // have been normalized before.
        let fn_sig = self
            .infcx()
            .instantiate_binder_with_fresh_vars(BoundRegionConversionTime::FnCall, fn_sig);

        let indices_to_skip = self.check_legacy_const_generics(def_id, arg_exprs);
        self.check_call_arguments(
            call_expr,
            fn_sig.inputs(),
            fn_sig.output(),
            expected,
            arg_exprs,
            &indices_to_skip,
            fn_sig.c_variadic,
            TupleArgumentsFlag::DontTupleArguments,
        );

        if fn_sig.abi == FnAbi::RustCall
            && let Some(ty) = fn_sig.inputs().last().copied()
            && let Some(tuple_trait) = self.lang_items.Tuple
        {
            self.table.register_bound(ty, tuple_trait, ObligationCause::new());
            self.require_type_is_sized(ty);
        }

        fn_sig.output()
    }

    fn confirm_deferred_closure_call(
        &mut self,
        call_expr: ExprId,
        arg_exprs: &[ExprId],
        expected: &Expectation<'db>,
        fn_sig: FnSig<'db>,
    ) -> Ty<'db> {
        // `fn_sig` is the *signature* of the closure being called. We
        // don't know the full details yet (`Fn` vs `FnMut` etc), but we
        // do know the types expected for each argument and the return
        // type.
        self.check_call_arguments(
            call_expr,
            fn_sig.inputs(),
            fn_sig.output(),
            expected,
            arg_exprs,
            &[],
            fn_sig.c_variadic,
            TupleArgumentsFlag::TupleArguments,
        );

        fn_sig.output()
    }

    fn confirm_overloaded_call(
        &mut self,
        call_expr: ExprId,
        arg_exprs: &[ExprId],
        expected: &Expectation<'db>,
        method: MethodCallee<'db>,
    ) -> Ty<'db> {
        self.check_call_arguments(
            call_expr,
            &method.sig.inputs()[1..],
            method.sig.output(),
            expected,
            arg_exprs,
            &[],
            method.sig.c_variadic,
            TupleArgumentsFlag::TupleArguments,
        );

        self.write_method_resolution(call_expr, method.def_id, method.args);

        method.sig.output()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DeferredCallResolution<'db> {
    call_expr: ExprId,
    callee_expr: ExprId,
    closure_ty: Ty<'db>,
    adjustments: Vec<Adjustment>,
    fn_sig: FnSig<'db>,
}

impl<'a, 'db> DeferredCallResolution<'db> {
    pub(crate) fn resolve(self, ctx: &mut InferenceContext<'a, 'db>) {
        debug!("DeferredCallResolution::resolve() {:?}", self);

        // we should not be invoked until the closure kind has been
        // determined by upvar inference
        assert!(ctx.infcx().closure_kind(self.closure_ty).is_some());

        // We may now know enough to figure out fn vs fnmut etc.
        match ctx.try_overloaded_call_traits(self.closure_ty, None) {
            Some((autoref, method_callee)) => {
                // One problem is that when we get here, we are going
                // to have a newly instantiated function signature
                // from the call trait. This has to be reconciled with
                // the older function signature we had before. In
                // principle we *should* be able to fn_sigs(), but we
                // can't because of the annoying need for a TypeTrace.
                // (This always bites me, should find a way to
                // refactor it.)
                let method_sig = method_callee.sig;

                debug!("attempt_resolution: method_callee={:?}", method_callee);

                for (method_arg_ty, self_arg_ty) in
                    iter::zip(method_sig.inputs().iter().skip(1), self.fn_sig.inputs())
                {
                    _ = ctx.demand_eqtype(self.call_expr.into(), *self_arg_ty, *method_arg_ty);
                }

                _ = ctx.demand_eqtype(
                    self.call_expr.into(),
                    method_sig.output(),
                    self.fn_sig.output(),
                );

                let mut adjustments = self.adjustments;
                adjustments.extend(autoref);
                ctx.write_expr_adj(self.callee_expr, adjustments.into_boxed_slice());

                ctx.write_method_resolution(
                    self.call_expr,
                    method_callee.def_id,
                    method_callee.args,
                );
            }
            None => {
                assert!(
                    ctx.lang_items.FnOnce.is_none(),
                    "Expected to find a suitable `Fn`/`FnMut`/`FnOnce` implementation for `{:?}`",
                    self.closure_ty
                )
            }
        }
    }
}
