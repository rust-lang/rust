use itertools::Itertools;
use rustc_abi::ExternAbi;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, EarlyBinder, GenericArgsRef, Ty, TyCtxt};
use rustc_session::lint::builtin::FUNCTION_ITEM_REFERENCES;
use rustc_span::Span;
use rustc_span::source_map::Spanned;
use rustc_span::symbol::sym;

use crate::errors;

pub(super) struct FunctionItemReferences;

impl<'tcx> crate::MirLint<'tcx> for FunctionItemReferences {
    fn run_lint(&self, tcx: TyCtxt<'tcx>, body: &Body<'tcx>) {
        let mut checker = FunctionItemRefChecker { tcx, body };
        checker.visit_body(body);
    }
}

struct FunctionItemRefChecker<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
}

impl<'tcx> Visitor<'tcx> for FunctionItemRefChecker<'_, 'tcx> {
    /// Emits a lint for function reference arguments bound by `fmt::Pointer` or passed to
    /// `transmute`. This only handles arguments in calls outside macro expansions to avoid double
    /// counting function references formatted as pointers by macros.
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        if let TerminatorKind::Call {
            func,
            args,
            destination: _,
            target: _,
            unwind: _,
            call_source: _,
            fn_span: _,
        } = &terminator.kind
        {
            let source_info = *self.body.source_info(location);
            let func_ty = func.ty(self.body, self.tcx);
            if let ty::FnDef(def_id, args_ref) = *func_ty.kind() {
                // Handle calls to `transmute`
                if self.tcx.is_diagnostic_item(sym::transmute, def_id) {
                    let arg_ty = args[0].node.ty(self.body, self.tcx);
                    for inner_ty in arg_ty.walk().filter_map(|arg| arg.as_type()) {
                        if let Some((fn_id, fn_args)) = FunctionItemRefChecker::is_fn_ref(inner_ty)
                        {
                            let span = self.nth_arg_span(args, 0);
                            self.emit_lint(fn_id, fn_args, source_info, span);
                        }
                    }
                } else {
                    self.check_bound_args(def_id, args_ref, args, source_info);
                }
            }
        }
        self.super_terminator(terminator, location);
    }
}

impl<'tcx> FunctionItemRefChecker<'_, 'tcx> {
    /// Emits a lint for function reference arguments bound by `fmt::Pointer` in calls to the
    /// function defined by `def_id` with the generic parameters `args_ref`.
    fn check_bound_args(
        &self,
        def_id: DefId,
        args_ref: GenericArgsRef<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        source_info: SourceInfo,
    ) {
        let param_env = self.tcx.param_env(def_id);
        let bounds = param_env.caller_bounds();
        for bound in bounds {
            if let Some(bound_ty) = self.is_pointer_trait(bound) {
                // Get the argument types as they appear in the function signature.
                let arg_defs =
                    self.tcx.fn_sig(def_id).instantiate_identity().skip_binder().inputs();
                for (arg_num, arg_def) in arg_defs.iter().enumerate() {
                    // For all types reachable from the argument type in the fn sig
                    for inner_ty in arg_def.walk().filter_map(|arg| arg.as_type()) {
                        // If the inner type matches the type bound by `Pointer`
                        if inner_ty == bound_ty {
                            // Do an instantiation using the parameters from the callsite
                            let instantiated_ty =
                                EarlyBinder::bind(inner_ty).instantiate(self.tcx, args_ref);
                            if let Some((fn_id, fn_args)) =
                                FunctionItemRefChecker::is_fn_ref(instantiated_ty)
                            {
                                let mut span = self.nth_arg_span(args, arg_num);
                                if span.from_expansion() {
                                    // The operand's ctxt wouldn't display the lint since it's
                                    // inside a macro so we have to use the callsite's ctxt.
                                    let callsite_ctxt = span.source_callsite().ctxt();
                                    span = span.with_ctxt(callsite_ctxt);
                                }
                                self.emit_lint(fn_id, fn_args, source_info, span);
                            }
                        }
                    }
                }
            }
        }
    }

    /// If the given predicate is the trait `fmt::Pointer`, returns the bound parameter type.
    fn is_pointer_trait(&self, bound: ty::Clause<'tcx>) -> Option<Ty<'tcx>> {
        if let ty::ClauseKind::Trait(predicate) = bound.kind().skip_binder() {
            self.tcx
                .is_diagnostic_item(sym::Pointer, predicate.def_id())
                .then(|| predicate.trait_ref.self_ty())
        } else {
            None
        }
    }

    /// If a type is a reference or raw pointer to the anonymous type of a function definition,
    /// returns that function's `DefId` and `GenericArgsRef`.
    fn is_fn_ref(ty: Ty<'tcx>) -> Option<(DefId, GenericArgsRef<'tcx>)> {
        let referent_ty = match ty.kind() {
            ty::Ref(_, referent_ty, _) => Some(referent_ty),
            ty::RawPtr(referent_ty, _) => Some(referent_ty),
            _ => None,
        };
        referent_ty
            .map(|ref_ty| {
                if let ty::FnDef(def_id, args_ref) = *ref_ty.kind() {
                    Some((def_id, args_ref))
                } else {
                    None
                }
            })
            .unwrap_or(None)
    }

    fn nth_arg_span(&self, args: &[Spanned<Operand<'tcx>>], n: usize) -> Span {
        match &args[n].node {
            Operand::Copy(place) | Operand::Move(place) => {
                self.body.local_decls[place.local].source_info.span
            }
            Operand::Constant(constant) => constant.span,
        }
    }

    fn emit_lint(
        &self,
        fn_id: DefId,
        fn_args: GenericArgsRef<'tcx>,
        source_info: SourceInfo,
        span: Span,
    ) {
        let lint_root = self.body.source_scopes[source_info.scope]
            .local_data
            .as_ref()
            .assert_crate_local()
            .lint_root;
        // FIXME: use existing printing routines to print the function signature
        let fn_sig = self.tcx.fn_sig(fn_id).instantiate(self.tcx, fn_args);
        let unsafety = fn_sig.safety().prefix_str();
        let abi = match fn_sig.abi() {
            ExternAbi::Rust => String::from(""),
            other_abi => {
                let mut s = String::from("extern \"");
                s.push_str(other_abi.name());
                s.push_str("\" ");
                s
            }
        };
        let ident = self.tcx.item_name(fn_id).to_ident_string();
        let ty_params = fn_args.types().map(|ty| format!("{ty}"));
        let const_params = fn_args.consts().map(|c| format!("{c}"));
        let params = ty_params.chain(const_params).join(", ");
        let num_args = fn_sig.inputs().map_bound(|inputs| inputs.len()).skip_binder();
        let variadic = if fn_sig.c_variadic() { ", ..." } else { "" };
        let ret = if fn_sig.output().skip_binder().is_unit() { "" } else { " -> _" };
        let sugg = format!(
            "{} as {}{}fn({}{}){}",
            if params.is_empty() { ident.clone() } else { format!("{ident}::<{params}>") },
            unsafety,
            abi,
            vec!["_"; num_args].join(", "),
            variadic,
            ret,
        );

        self.tcx.emit_node_span_lint(
            FUNCTION_ITEM_REFERENCES,
            lint_root,
            span,
            errors::FnItemRef { span, sugg, ident },
        );
    }
}
