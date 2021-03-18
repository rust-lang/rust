use rustc_errors::Applicability;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::visit::Visitor;
use rustc_middle::mir::*;
use rustc_middle::ty::{
    self,
    subst::{GenericArgKind, Subst, SubstsRef},
    PredicateKind, Ty, TyCtxt, TyS,
};
use rustc_session::lint::builtin::FUNCTION_ITEM_REFERENCES;
use rustc_span::{symbol::sym, Span};
use rustc_target::spec::abi::Abi;

use crate::transform::MirPass;

pub struct FunctionItemReferences;

impl<'tcx> MirPass<'tcx> for FunctionItemReferences {
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let mut checker = FunctionItemRefChecker { tcx, body };
        checker.visit_body(&body);
    }
}

struct FunctionItemRefChecker<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    body: &'a Body<'tcx>,
}

impl<'a, 'tcx> Visitor<'tcx> for FunctionItemRefChecker<'a, 'tcx> {
    /// Emits a lint for function reference arguments bound by `fmt::Pointer` or passed to
    /// `transmute`. This only handles arguments in calls outside macro expansions to avoid double
    /// counting function references formatted as pointers by macros.
    fn visit_terminator(&mut self, terminator: &Terminator<'tcx>, location: Location) {
        if let TerminatorKind::Call {
            func,
            args,
            destination: _,
            cleanup: _,
            from_hir_call: _,
            fn_span: _,
        } = &terminator.kind
        {
            let source_info = *self.body.source_info(location);
            // Only handle function calls outside macros
            if !source_info.span.from_expansion() {
                let func_ty = func.ty(self.body, self.tcx);
                if let ty::FnDef(def_id, substs_ref) = *func_ty.kind() {
                    // Handle calls to `transmute`
                    if self.tcx.is_diagnostic_item(sym::transmute, def_id) {
                        let arg_ty = args[0].ty(self.body, self.tcx);
                        for generic_inner_ty in arg_ty.walk() {
                            if let GenericArgKind::Type(inner_ty) = generic_inner_ty.unpack() {
                                if let Some((fn_id, fn_substs)) =
                                    FunctionItemRefChecker::is_fn_ref(inner_ty)
                                {
                                    let span = self.nth_arg_span(&args, 0);
                                    self.emit_lint(fn_id, fn_substs, source_info, span);
                                }
                            }
                        }
                    } else {
                        self.check_bound_args(def_id, substs_ref, &args, source_info);
                    }
                }
            }
        }
        self.super_terminator(terminator, location);
    }

    /// Emits a lint for function references formatted with `fmt::Pointer::fmt` by macros. These
    /// cases are handled as operands instead of call terminators to avoid any dependence on
    /// unstable, internal formatting details like whether `fmt` is called directly or not.
    fn visit_operand(&mut self, operand: &Operand<'tcx>, location: Location) {
        let source_info = *self.body.source_info(location);
        if source_info.span.from_expansion() {
            let op_ty = operand.ty(self.body, self.tcx);
            if let ty::FnDef(def_id, substs_ref) = *op_ty.kind() {
                if self.tcx.is_diagnostic_item(sym::pointer_trait_fmt, def_id) {
                    let param_ty = substs_ref.type_at(0);
                    if let Some((fn_id, fn_substs)) = FunctionItemRefChecker::is_fn_ref(param_ty) {
                        // The operand's ctxt wouldn't display the lint since it's inside a macro so
                        // we have to use the callsite's ctxt.
                        let callsite_ctxt = source_info.span.source_callsite().ctxt();
                        let span = source_info.span.with_ctxt(callsite_ctxt);
                        self.emit_lint(fn_id, fn_substs, source_info, span);
                    }
                }
            }
        }
        self.super_operand(operand, location);
    }
}

impl<'a, 'tcx> FunctionItemRefChecker<'a, 'tcx> {
    /// Emits a lint for function reference arguments bound by `fmt::Pointer` in calls to the
    /// function defined by `def_id` with the substitutions `substs_ref`.
    fn check_bound_args(
        &self,
        def_id: DefId,
        substs_ref: SubstsRef<'tcx>,
        args: &[Operand<'tcx>],
        source_info: SourceInfo,
    ) {
        let param_env = self.tcx.param_env(def_id);
        let bounds = param_env.caller_bounds();
        for bound in bounds {
            if let Some(bound_ty) = self.is_pointer_trait(&bound.kind().skip_binder()) {
                // Get the argument types as they appear in the function signature.
                let arg_defs = self.tcx.fn_sig(def_id).skip_binder().inputs();
                for (arg_num, arg_def) in arg_defs.iter().enumerate() {
                    // For all types reachable from the argument type in the fn sig
                    for generic_inner_ty in arg_def.walk() {
                        if let GenericArgKind::Type(inner_ty) = generic_inner_ty.unpack() {
                            // If the inner type matches the type bound by `Pointer`
                            if TyS::same_type(inner_ty, bound_ty) {
                                // Do a substitution using the parameters from the callsite
                                let subst_ty = inner_ty.subst(self.tcx, substs_ref);
                                if let Some((fn_id, fn_substs)) =
                                    FunctionItemRefChecker::is_fn_ref(subst_ty)
                                {
                                    let span = self.nth_arg_span(args, arg_num);
                                    self.emit_lint(fn_id, fn_substs, source_info, span);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// If the given predicate is the trait `fmt::Pointer`, returns the bound parameter type.
    fn is_pointer_trait(&self, bound: &PredicateKind<'tcx>) -> Option<Ty<'tcx>> {
        if let ty::PredicateKind::Trait(predicate, _) = bound {
            if self.tcx.is_diagnostic_item(sym::pointer_trait, predicate.def_id()) {
                Some(predicate.trait_ref.self_ty())
            } else {
                None
            }
        } else {
            None
        }
    }

    /// If a type is a reference or raw pointer to the anonymous type of a function definition,
    /// returns that function's `DefId` and `SubstsRef`.
    fn is_fn_ref(ty: Ty<'tcx>) -> Option<(DefId, SubstsRef<'tcx>)> {
        let referent_ty = match ty.kind() {
            ty::Ref(_, referent_ty, _) => Some(referent_ty),
            ty::RawPtr(ty_and_mut) => Some(&ty_and_mut.ty),
            _ => None,
        };
        referent_ty
            .map(|ref_ty| {
                if let ty::FnDef(def_id, substs_ref) = *ref_ty.kind() {
                    Some((def_id, substs_ref))
                } else {
                    None
                }
            })
            .unwrap_or(None)
    }

    fn nth_arg_span(&self, args: &[Operand<'tcx>], n: usize) -> Span {
        match &args[n] {
            Operand::Copy(place) | Operand::Move(place) => {
                self.body.local_decls[place.local].source_info.span
            }
            Operand::Constant(constant) => constant.span,
        }
    }

    fn emit_lint(
        &self,
        fn_id: DefId,
        fn_substs: SubstsRef<'tcx>,
        source_info: SourceInfo,
        span: Span,
    ) {
        let lint_root = self.body.source_scopes[source_info.scope]
            .local_data
            .as_ref()
            .assert_crate_local()
            .lint_root;
        let fn_sig = self.tcx.fn_sig(fn_id);
        let unsafety = fn_sig.unsafety().prefix_str();
        let abi = match fn_sig.abi() {
            Abi::Rust => String::from(""),
            other_abi => {
                let mut s = String::from("extern \"");
                s.push_str(other_abi.name());
                s.push_str("\" ");
                s
            }
        };
        let ident = self.tcx.item_name(fn_id).to_ident_string();
        let ty_params = fn_substs.types().map(|ty| format!("{}", ty));
        let const_params = fn_substs.consts().map(|c| format!("{}", c));
        let params = ty_params.chain(const_params).collect::<Vec<String>>().join(", ");
        let num_args = fn_sig.inputs().map_bound(|inputs| inputs.len()).skip_binder();
        let variadic = if fn_sig.c_variadic() { ", ..." } else { "" };
        let ret = if fn_sig.output().skip_binder().is_unit() { "" } else { " -> _" };
        self.tcx.struct_span_lint_hir(FUNCTION_ITEM_REFERENCES, lint_root, span, |lint| {
            lint.build("taking a reference to a function item does not give a function pointer")
                .span_suggestion(
                    span,
                    &format!("cast `{}` to obtain a function pointer", ident),
                    format!(
                        "{} as {}{}fn({}{}){}",
                        if params.is_empty() { ident } else { format!("{}::<{}>", ident, params) },
                        unsafety,
                        abi,
                        vec!["_"; num_args].join(", "),
                        variadic,
                        ret,
                    ),
                    Applicability::Unspecified,
                )
                .emit();
        });
    }
}
