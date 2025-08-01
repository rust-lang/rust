//! Performs various peephole optimizations.

use rustc_abi::ExternAbi;
use rustc_ast::attr;
use rustc_hir::LangItem;
use rustc_middle::bug;
use rustc_middle::mir::*;
use rustc_middle::ty::layout::ValidityRequirement;
use rustc_middle::ty::{self, GenericArgsRef, Ty, TyCtxt, layout};
use rustc_span::{DUMMY_SP, Symbol, sym};

use crate::simplify::simplify_duplicate_switch_targets;

pub(super) enum InstSimplify {
    BeforeInline,
    AfterSimplifyCfg,
}

impl<'tcx> crate::MirPass<'tcx> for InstSimplify {
    fn name(&self) -> &'static str {
        match self {
            InstSimplify::BeforeInline => "InstSimplify-before-inline",
            InstSimplify::AfterSimplifyCfg => "InstSimplify-after-simplifycfg",
        }
    }

    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() > 0
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        let ctx = InstSimplifyContext {
            tcx,
            local_decls: &body.local_decls,
            typing_env: body.typing_env(tcx),
        };
        let preserve_ub_checks =
            attr::contains_name(tcx.hir_krate_attrs(), sym::rustc_preserve_ub_checks);
        for block in body.basic_blocks.as_mut() {
            for statement in block.statements.iter_mut() {
                let StatementKind::Assign(box (.., rvalue)) = &mut statement.kind else {
                    continue;
                };

                if !preserve_ub_checks {
                    ctx.simplify_ub_check(rvalue);
                }
                ctx.simplify_bool_cmp(rvalue);
                ctx.simplify_ref_deref(rvalue);
                ctx.simplify_ptr_aggregate(rvalue);
                ctx.simplify_cast(rvalue);
                ctx.simplify_repeated_aggregate(rvalue);
                ctx.simplify_repeat_once(rvalue);
            }

            let terminator = block.terminator.as_mut().unwrap();
            ctx.simplify_primitive_clone(terminator, &mut block.statements);
            ctx.simplify_align_of_slice_val(terminator, &mut block.statements);
            ctx.simplify_intrinsic_assert(terminator);
            ctx.simplify_nounwind_call(terminator);
            simplify_duplicate_switch_targets(terminator);
        }
    }

    fn is_required(&self) -> bool {
        false
    }
}

struct InstSimplifyContext<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
}

impl<'tcx> InstSimplifyContext<'_, 'tcx> {
    /// Transform aggregates like [0, 0, 0, 0, 0] into [0; 5].
    /// GVN can also do this optimization, but GVN is only run at mir-opt-level 2 so having this in
    /// InstSimplify helps unoptimized builds.
    fn simplify_repeated_aggregate(&self, rvalue: &mut Rvalue<'tcx>) {
        let Rvalue::Aggregate(box AggregateKind::Array(_), fields) = &*rvalue else {
            return;
        };
        if fields.len() < 5 {
            return;
        }
        let (first, rest) = fields[..].split_first().unwrap();
        let Operand::Constant(first) = first else {
            return;
        };
        let Ok(first_val) = first.const_.eval(self.tcx, self.typing_env, first.span) else {
            return;
        };
        if rest.iter().all(|field| {
            let Operand::Constant(field) = field else {
                return false;
            };
            let field = field.const_.eval(self.tcx, self.typing_env, field.span);
            field == Ok(first_val)
        }) {
            let len = ty::Const::from_target_usize(self.tcx, fields.len().try_into().unwrap());
            *rvalue = Rvalue::Repeat(Operand::Constant(first.clone()), len);
        }
    }

    /// Transform boolean comparisons into logical operations.
    fn simplify_bool_cmp(&self, rvalue: &mut Rvalue<'tcx>) {
        let Rvalue::BinaryOp(op @ (BinOp::Eq | BinOp::Ne), box (a, b)) = &*rvalue else { return };
        *rvalue = match (op, self.try_eval_bool(a), self.try_eval_bool(b)) {
            // Transform "Eq(a, true)" ==> "a"
            (BinOp::Eq, _, Some(true)) => Rvalue::Use(a.clone()),

            // Transform "Ne(a, false)" ==> "a"
            (BinOp::Ne, _, Some(false)) => Rvalue::Use(a.clone()),

            // Transform "Eq(true, b)" ==> "b"
            (BinOp::Eq, Some(true), _) => Rvalue::Use(b.clone()),

            // Transform "Ne(false, b)" ==> "b"
            (BinOp::Ne, Some(false), _) => Rvalue::Use(b.clone()),

            // Transform "Eq(false, b)" ==> "Not(b)"
            (BinOp::Eq, Some(false), _) => Rvalue::UnaryOp(UnOp::Not, b.clone()),

            // Transform "Ne(true, b)" ==> "Not(b)"
            (BinOp::Ne, Some(true), _) => Rvalue::UnaryOp(UnOp::Not, b.clone()),

            // Transform "Eq(a, false)" ==> "Not(a)"
            (BinOp::Eq, _, Some(false)) => Rvalue::UnaryOp(UnOp::Not, a.clone()),

            // Transform "Ne(a, true)" ==> "Not(a)"
            (BinOp::Ne, _, Some(true)) => Rvalue::UnaryOp(UnOp::Not, a.clone()),

            _ => return,
        };
    }

    fn try_eval_bool(&self, a: &Operand<'_>) -> Option<bool> {
        let a = a.constant()?;
        if a.const_.ty().is_bool() { a.const_.try_to_bool() } else { None }
    }

    /// Transform `&(*a)` ==> `a`.
    fn simplify_ref_deref(&self, rvalue: &mut Rvalue<'tcx>) {
        if let Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) = rvalue
            && let Some((base, ProjectionElem::Deref)) = place.as_ref().last_projection()
            && rvalue.ty(self.local_decls, self.tcx) == base.ty(self.local_decls, self.tcx).ty
        {
            *rvalue = Rvalue::Use(Operand::Copy(Place {
                local: base.local,
                projection: self.tcx.mk_place_elems(base.projection),
            }));
        }
    }

    /// Transform `Aggregate(RawPtr, [p, ()])` ==> `Cast(PtrToPtr, p)`.
    fn simplify_ptr_aggregate(&self, rvalue: &mut Rvalue<'tcx>) {
        if let Rvalue::Aggregate(box AggregateKind::RawPtr(pointee_ty, mutability), fields) = rvalue
            && let meta_ty = fields.raw[1].ty(self.local_decls, self.tcx)
            && meta_ty.is_unit()
        {
            // The mutable borrows we're holding prevent printing `rvalue` here
            let mut fields = std::mem::take(fields);
            let _meta = fields.pop().unwrap();
            let data = fields.pop().unwrap();
            let ptr_ty = Ty::new_ptr(self.tcx, *pointee_ty, *mutability);
            *rvalue = Rvalue::Cast(CastKind::PtrToPtr, data, ptr_ty);
        }
    }

    fn simplify_ub_check(&self, rvalue: &mut Rvalue<'tcx>) {
        let Rvalue::NullaryOp(NullOp::UbChecks, _) = *rvalue else { return };

        let const_ = Const::from_bool(self.tcx, self.tcx.sess.ub_checks());
        let constant = ConstOperand { span: DUMMY_SP, const_, user_ty: None };
        *rvalue = Rvalue::Use(Operand::Constant(Box::new(constant)));
    }

    fn simplify_cast(&self, rvalue: &mut Rvalue<'tcx>) {
        let Rvalue::Cast(kind, operand, cast_ty) = rvalue else { return };

        let operand_ty = operand.ty(self.local_decls, self.tcx);
        if operand_ty == *cast_ty {
            *rvalue = Rvalue::Use(operand.clone());
        } else if *kind == CastKind::Transmute
            // Transmuting an integer to another integer is just a signedness cast
            && let (ty::Int(int), ty::Uint(uint)) | (ty::Uint(uint), ty::Int(int)) =
                (operand_ty.kind(), cast_ty.kind())
            && int.bit_width() == uint.bit_width()
        {
            // The width check isn't strictly necessary, as different widths
            // are UB and thus we'd be allowed to turn it into a cast anyway.
            // But let's keep the UB around for codegen to exploit later.
            // (If `CastKind::Transmute` ever becomes *not* UB for mismatched sizes,
            // then the width check is necessary for big-endian correctness.)
            *kind = CastKind::IntToInt;
        }
    }

    /// Simplify `[x; 1]` to just `[x]`.
    fn simplify_repeat_once(&self, rvalue: &mut Rvalue<'tcx>) {
        if let Rvalue::Repeat(operand, count) = rvalue
            && let Some(1) = count.try_to_target_usize(self.tcx)
        {
            *rvalue = Rvalue::Aggregate(
                Box::new(AggregateKind::Array(operand.ty(self.local_decls, self.tcx))),
                [operand.clone()].into(),
            );
        }
    }

    fn simplify_primitive_clone(
        &self,
        terminator: &mut Terminator<'tcx>,
        statements: &mut Vec<Statement<'tcx>>,
    ) {
        let TerminatorKind::Call {
            func, args, destination, target: Some(destination_block), ..
        } = &terminator.kind
        else {
            return;
        };

        // It's definitely not a clone if there are multiple arguments
        let [arg] = &args[..] else { return };

        // Only bother looking more if it's easy to know what we're calling
        let Some((fn_def_id, ..)) = func.const_fn_def() else { return };

        // These types are easily available from locals, so check that before
        // doing DefId lookups to figure out what we're actually calling.
        let arg_ty = arg.node.ty(self.local_decls, self.tcx);

        let ty::Ref(_region, inner_ty, Mutability::Not) = *arg_ty.kind() else { return };

        if !self.tcx.is_lang_item(fn_def_id, LangItem::CloneFn)
            || !inner_ty.is_trivially_pure_clone_copy()
        {
            return;
        }

        let Some(arg_place) = arg.node.place() else { return };

        statements.push(Statement::new(
            terminator.source_info,
            StatementKind::Assign(Box::new((
                *destination,
                Rvalue::Use(Operand::Copy(
                    arg_place.project_deeper(&[ProjectionElem::Deref], self.tcx),
                )),
            ))),
        ));
        terminator.kind = TerminatorKind::Goto { target: *destination_block };
    }

    // Convert `align_of_val::<[T]>(ptr)` to `align_of::<T>()`, since the
    // alignment of a slice doesn't actually depend on metadata at all
    // and the element type is always `Sized`.
    //
    // This is here so it can run after inlining, where it's more useful.
    // (LowerIntrinsics is done in cleanup, before the optimization passes.)
    fn simplify_align_of_slice_val(
        &self,
        terminator: &mut Terminator<'tcx>,
        statements: &mut Vec<Statement<'tcx>>,
    ) {
        if let TerminatorKind::Call {
            func, args, destination, target: Some(destination_block), ..
        } = &terminator.kind
            && args.len() == 1
            && let Some((fn_def_id, generics)) = func.const_fn_def()
            && self.tcx.is_intrinsic(fn_def_id, sym::align_of_val)
            && let ty::Slice(elem_ty) = *generics.type_at(0).kind()
        {
            statements.push(Statement::new(
                terminator.source_info,
                StatementKind::Assign(Box::new((
                    *destination,
                    Rvalue::NullaryOp(NullOp::AlignOf, elem_ty),
                ))),
            ));
            terminator.kind = TerminatorKind::Goto { target: *destination_block };
        }
    }

    fn simplify_nounwind_call(&self, terminator: &mut Terminator<'tcx>) {
        let TerminatorKind::Call { ref func, ref mut unwind, .. } = terminator.kind else {
            return;
        };

        let Some((def_id, _)) = func.const_fn_def() else {
            return;
        };

        let body_ty = self.tcx.type_of(def_id).skip_binder();
        let body_abi = match body_ty.kind() {
            ty::FnDef(..) => body_ty.fn_sig(self.tcx).abi(),
            ty::Closure(..) => ExternAbi::RustCall,
            ty::Coroutine(..) => ExternAbi::Rust,
            _ => bug!("unexpected body ty: {body_ty:?}"),
        };

        if !layout::fn_can_unwind(self.tcx, Some(def_id), body_abi) {
            *unwind = UnwindAction::Unreachable;
        }
    }

    fn simplify_intrinsic_assert(&self, terminator: &mut Terminator<'tcx>) {
        let TerminatorKind::Call { ref func, target: ref mut target @ Some(target_block), .. } =
            terminator.kind
        else {
            return;
        };
        let func_ty = func.ty(self.local_decls, self.tcx);
        let Some((intrinsic_name, args)) = resolve_rust_intrinsic(self.tcx, func_ty) else {
            return;
        };
        // The intrinsics we are interested in have one generic parameter
        let [arg, ..] = args[..] else { return };

        let known_is_valid =
            intrinsic_assert_panics(self.tcx, self.typing_env, arg, intrinsic_name);
        match known_is_valid {
            // We don't know the layout or it's not validity assertion at all, don't touch it
            None => {}
            Some(true) => {
                // If we know the assert panics, indicate to later opts that the call diverges
                *target = None;
            }
            Some(false) => {
                // If we know the assert does not panic, turn the call into a Goto
                terminator.kind = TerminatorKind::Goto { target: target_block };
            }
        }
    }
}

fn intrinsic_assert_panics<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    arg: ty::GenericArg<'tcx>,
    intrinsic_name: Symbol,
) -> Option<bool> {
    let requirement = ValidityRequirement::from_intrinsic(intrinsic_name)?;
    let ty = arg.expect_ty();
    Some(!tcx.check_validity_requirement((requirement, typing_env.as_query_input(ty))).ok()?)
}

fn resolve_rust_intrinsic<'tcx>(
    tcx: TyCtxt<'tcx>,
    func_ty: Ty<'tcx>,
) -> Option<(Symbol, GenericArgsRef<'tcx>)> {
    let ty::FnDef(def_id, args) = *func_ty.kind() else { return None };
    let intrinsic = tcx.intrinsic(def_id)?;
    Some((intrinsic.name, args))
}
