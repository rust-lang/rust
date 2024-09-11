//! Performs various peephole optimizations.

use rustc_ast::attr;
use rustc_hir::LangItem;
use rustc_middle::bug;
use rustc_middle::mir::*;
use rustc_middle::ty::layout::ValidityRequirement;
use rustc_middle::ty::{self, layout, GenericArgsRef, ParamEnv, Ty, TyCtxt};
use rustc_span::sym;
use rustc_span::symbol::Symbol;
use rustc_target::spec::abi::Abi;

use crate::simplify::simplify_duplicate_switch_targets;
use crate::take_array;

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
            param_env: tcx.param_env_reveal_all_normalized(body.source.def_id()),
        };
        let preserve_ub_checks =
            attr::contains_name(tcx.hir().krate_attrs(), sym::rustc_preserve_ub_checks);
        for block in body.basic_blocks.as_mut() {
            for statement in block.statements.iter_mut() {
                match statement.kind {
                    StatementKind::Assign(box (_place, ref mut rvalue)) => {
                        if !preserve_ub_checks {
                            ctx.simplify_ub_check(&statement.source_info, rvalue);
                        }
                        ctx.simplify_bool_cmp(&statement.source_info, rvalue);
                        ctx.simplify_ref_deref(&statement.source_info, rvalue);
                        ctx.simplify_len(&statement.source_info, rvalue);
                        ctx.simplify_ptr_aggregate(&statement.source_info, rvalue);
                        ctx.simplify_cast(rvalue);
                    }
                    _ => {}
                }
            }

            ctx.simplify_primitive_clone(block.terminator.as_mut().unwrap(), &mut block.statements);
            ctx.simplify_intrinsic_assert(block.terminator.as_mut().unwrap());
            ctx.simplify_nounwind_call(block.terminator.as_mut().unwrap());
            simplify_duplicate_switch_targets(block.terminator.as_mut().unwrap());
        }
    }
}

struct InstSimplifyContext<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    local_decls: &'a LocalDecls<'tcx>,
    param_env: ParamEnv<'tcx>,
}

impl<'tcx> InstSimplifyContext<'_, 'tcx> {
    fn should_simplify(&self, source_info: &SourceInfo, rvalue: &Rvalue<'tcx>) -> bool {
        self.should_simplify_custom(source_info, "Rvalue", rvalue)
    }

    fn should_simplify_custom(
        &self,
        source_info: &SourceInfo,
        label: &str,
        value: impl std::fmt::Debug,
    ) -> bool {
        self.tcx.consider_optimizing(|| {
            format!("InstSimplify - {label}: {value:?} SourceInfo: {source_info:?}")
        })
    }

    /// Transform boolean comparisons into logical operations.
    fn simplify_bool_cmp(&self, source_info: &SourceInfo, rvalue: &mut Rvalue<'tcx>) {
        match rvalue {
            Rvalue::BinaryOp(op @ (BinOp::Eq | BinOp::Ne), box (a, b)) => {
                let new = match (op, self.try_eval_bool(a), self.try_eval_bool(b)) {
                    // Transform "Eq(a, true)" ==> "a"
                    (BinOp::Eq, _, Some(true)) => Some(Rvalue::Use(a.clone())),

                    // Transform "Ne(a, false)" ==> "a"
                    (BinOp::Ne, _, Some(false)) => Some(Rvalue::Use(a.clone())),

                    // Transform "Eq(true, b)" ==> "b"
                    (BinOp::Eq, Some(true), _) => Some(Rvalue::Use(b.clone())),

                    // Transform "Ne(false, b)" ==> "b"
                    (BinOp::Ne, Some(false), _) => Some(Rvalue::Use(b.clone())),

                    // Transform "Eq(false, b)" ==> "Not(b)"
                    (BinOp::Eq, Some(false), _) => Some(Rvalue::UnaryOp(UnOp::Not, b.clone())),

                    // Transform "Ne(true, b)" ==> "Not(b)"
                    (BinOp::Ne, Some(true), _) => Some(Rvalue::UnaryOp(UnOp::Not, b.clone())),

                    // Transform "Eq(a, false)" ==> "Not(a)"
                    (BinOp::Eq, _, Some(false)) => Some(Rvalue::UnaryOp(UnOp::Not, a.clone())),

                    // Transform "Ne(a, true)" ==> "Not(a)"
                    (BinOp::Ne, _, Some(true)) => Some(Rvalue::UnaryOp(UnOp::Not, a.clone())),

                    _ => None,
                };

                if let Some(new) = new
                    && self.should_simplify(source_info, rvalue)
                {
                    *rvalue = new;
                }
            }

            _ => {}
        }
    }

    fn try_eval_bool(&self, a: &Operand<'_>) -> Option<bool> {
        let a = a.constant()?;
        if a.const_.ty().is_bool() { a.const_.try_to_bool() } else { None }
    }

    /// Transform `&(*a)` ==> `a`.
    fn simplify_ref_deref(&self, source_info: &SourceInfo, rvalue: &mut Rvalue<'tcx>) {
        if let Rvalue::Ref(_, _, place) | Rvalue::RawPtr(_, place) = rvalue {
            if let Some((base, ProjectionElem::Deref)) = place.as_ref().last_projection() {
                if rvalue.ty(self.local_decls, self.tcx) != base.ty(self.local_decls, self.tcx).ty {
                    return;
                }

                if !self.should_simplify(source_info, rvalue) {
                    return;
                }

                *rvalue = Rvalue::Use(Operand::Copy(Place {
                    local: base.local,
                    projection: self.tcx.mk_place_elems(base.projection),
                }));
            }
        }
    }

    /// Transform `Len([_; N])` ==> `N`.
    fn simplify_len(&self, source_info: &SourceInfo, rvalue: &mut Rvalue<'tcx>) {
        if let Rvalue::Len(ref place) = *rvalue {
            let place_ty = place.ty(self.local_decls, self.tcx).ty;
            if let ty::Array(_, len) = *place_ty.kind() {
                if !self.should_simplify(source_info, rvalue) {
                    return;
                }

                let const_ = Const::from_ty_const(len, self.tcx.types.usize, self.tcx);
                let constant = ConstOperand { span: source_info.span, const_, user_ty: None };
                *rvalue = Rvalue::Use(Operand::Constant(Box::new(constant)));
            }
        }
    }

    /// Transform `Aggregate(RawPtr, [p, ()])` ==> `Cast(PtrToPtr, p)`.
    fn simplify_ptr_aggregate(&self, source_info: &SourceInfo, rvalue: &mut Rvalue<'tcx>) {
        if let Rvalue::Aggregate(box AggregateKind::RawPtr(pointee_ty, mutability), fields) = rvalue
        {
            let meta_ty = fields.raw[1].ty(self.local_decls, self.tcx);
            if meta_ty.is_unit() {
                // The mutable borrows we're holding prevent printing `rvalue` here
                if !self.should_simplify_custom(
                    source_info,
                    "Aggregate::RawPtr",
                    (&pointee_ty, *mutability, &fields),
                ) {
                    return;
                }

                let mut fields = std::mem::take(fields);
                let _meta = fields.pop().unwrap();
                let data = fields.pop().unwrap();
                let ptr_ty = Ty::new_ptr(self.tcx, *pointee_ty, *mutability);
                *rvalue = Rvalue::Cast(CastKind::PtrToPtr, data, ptr_ty);
            }
        }
    }

    fn simplify_ub_check(&self, source_info: &SourceInfo, rvalue: &mut Rvalue<'tcx>) {
        if let Rvalue::NullaryOp(NullOp::UbChecks, _) = *rvalue {
            let const_ = Const::from_bool(self.tcx, self.tcx.sess.ub_checks());
            let constant = ConstOperand { span: source_info.span, const_, user_ty: None };
            *rvalue = Rvalue::Use(Operand::Constant(Box::new(constant)));
        }
    }

    fn simplify_cast(&self, rvalue: &mut Rvalue<'tcx>) {
        if let Rvalue::Cast(kind, operand, cast_ty) = rvalue {
            let operand_ty = operand.ty(self.local_decls, self.tcx);
            if operand_ty == *cast_ty {
                *rvalue = Rvalue::Use(operand.clone());
            } else if *kind == CastKind::Transmute {
                // Transmuting an integer to another integer is just a signedness cast
                if let (ty::Int(int), ty::Uint(uint)) | (ty::Uint(uint), ty::Int(int)) =
                    (operand_ty.kind(), cast_ty.kind())
                    && int.bit_width() == uint.bit_width()
                {
                    // The width check isn't strictly necessary, as different widths
                    // are UB and thus we'd be allowed to turn it into a cast anyway.
                    // But let's keep the UB around for codegen to exploit later.
                    // (If `CastKind::Transmute` ever becomes *not* UB for mismatched sizes,
                    // then the width check is necessary for big-endian correctness.)
                    *kind = CastKind::IntToInt;
                    return;
                }

                // Transmuting a transparent struct/union to a field's type is a projection
                if let ty::Adt(adt_def, args) = operand_ty.kind()
                    && adt_def.repr().transparent()
                    && (adt_def.is_struct() || adt_def.is_union())
                    && let Some(place) = operand.place()
                {
                    let variant = adt_def.non_enum_variant();
                    for (i, field) in variant.fields.iter_enumerated() {
                        let field_ty = field.ty(self.tcx, args);
                        if field_ty == *cast_ty {
                            let place = place
                                .project_deeper(&[ProjectionElem::Field(i, *cast_ty)], self.tcx);
                            let operand = if operand.is_move() {
                                Operand::Move(place)
                            } else {
                                Operand::Copy(place)
                            };
                            *rvalue = Rvalue::Use(operand);
                            return;
                        }
                    }
                }
            }
        }
    }

    fn simplify_primitive_clone(
        &self,
        terminator: &mut Terminator<'tcx>,
        statements: &mut Vec<Statement<'tcx>>,
    ) {
        let TerminatorKind::Call { func, args, destination, target, .. } = &mut terminator.kind
        else {
            return;
        };

        // It's definitely not a clone if there are multiple arguments
        let [arg] = &args[..] else { return };

        let Some(destination_block) = *target else { return };

        // Only bother looking more if it's easy to know what we're calling
        let Some((fn_def_id, fn_args)) = func.const_fn_def() else { return };

        // Clone needs one arg, so we can cheaply rule out other stuff
        if fn_args.len() != 1 {
            return;
        }

        // These types are easily available from locals, so check that before
        // doing DefId lookups to figure out what we're actually calling.
        let arg_ty = arg.node.ty(self.local_decls, self.tcx);

        let ty::Ref(_region, inner_ty, Mutability::Not) = *arg_ty.kind() else { return };

        if !inner_ty.is_trivially_pure_clone_copy() {
            return;
        }

        if !self.tcx.is_lang_item(fn_def_id, LangItem::CloneFn) {
            return;
        }

        if !self.tcx.consider_optimizing(|| {
            format!(
                "InstSimplify - Call: {:?} SourceInfo: {:?}",
                (fn_def_id, fn_args),
                terminator.source_info
            )
        }) {
            return;
        }

        let Ok([arg]) = take_array(args) else { return };
        let Some(arg_place) = arg.node.place() else { return };

        statements.push(Statement {
            source_info: terminator.source_info,
            kind: StatementKind::Assign(Box::new((
                *destination,
                Rvalue::Use(Operand::Copy(
                    arg_place.project_deeper(&[ProjectionElem::Deref], self.tcx),
                )),
            ))),
        });
        terminator.kind = TerminatorKind::Goto { target: destination_block };
    }

    fn simplify_nounwind_call(&self, terminator: &mut Terminator<'tcx>) {
        let TerminatorKind::Call { func, unwind, .. } = &mut terminator.kind else {
            return;
        };

        let Some((def_id, _)) = func.const_fn_def() else {
            return;
        };

        let body_ty = self.tcx.type_of(def_id).skip_binder();
        let body_abi = match body_ty.kind() {
            ty::FnDef(..) => body_ty.fn_sig(self.tcx).abi(),
            ty::Closure(..) => Abi::RustCall,
            ty::Coroutine(..) => Abi::Rust,
            _ => bug!("unexpected body ty: {:?}", body_ty),
        };

        if !layout::fn_can_unwind(self.tcx, Some(def_id), body_abi) {
            *unwind = UnwindAction::Unreachable;
        }
    }

    fn simplify_intrinsic_assert(&self, terminator: &mut Terminator<'tcx>) {
        let TerminatorKind::Call { func, target, .. } = &mut terminator.kind else {
            return;
        };
        let Some(target_block) = target else {
            return;
        };
        let func_ty = func.ty(self.local_decls, self.tcx);
        let Some((intrinsic_name, args)) = resolve_rust_intrinsic(self.tcx, func_ty) else {
            return;
        };
        // The intrinsics we are interested in have one generic parameter
        if args.is_empty() {
            return;
        }

        let known_is_valid =
            intrinsic_assert_panics(self.tcx, self.param_env, args[0], intrinsic_name);
        match known_is_valid {
            // We don't know the layout or it's not validity assertion at all, don't touch it
            None => {}
            Some(true) => {
                // If we know the assert panics, indicate to later opts that the call diverges
                *target = None;
            }
            Some(false) => {
                // If we know the assert does not panic, turn the call into a Goto
                terminator.kind = TerminatorKind::Goto { target: *target_block };
            }
        }
    }
}

fn intrinsic_assert_panics<'tcx>(
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    arg: ty::GenericArg<'tcx>,
    intrinsic_name: Symbol,
) -> Option<bool> {
    let requirement = ValidityRequirement::from_intrinsic(intrinsic_name)?;
    let ty = arg.expect_ty();
    Some(!tcx.check_validity_requirement((requirement, param_env.and(ty))).ok()?)
}

fn resolve_rust_intrinsic<'tcx>(
    tcx: TyCtxt<'tcx>,
    func_ty: Ty<'tcx>,
) -> Option<(Symbol, GenericArgsRef<'tcx>)> {
    if let ty::FnDef(def_id, args) = *func_ty.kind() {
        let intrinsic = tcx.intrinsic(def_id)?;
        return Some((intrinsic.name, args));
    }
    None
}
