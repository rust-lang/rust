use rustc_ast::InlineAsmTemplatePiece;
use rustc_data_structures::fx::FxHashSet;
use rustc_hir as hir;
use rustc_middle::ty::{self, Article, FloatTy, IntTy, Ty, TyCtxt, TypeVisitable, UintTy};
use rustc_session::lint;
use rustc_span::def_id::LocalDefId;
use rustc_span::{Symbol, DUMMY_SP};
use rustc_target::asm::{InlineAsmReg, InlineAsmRegClass, InlineAsmRegOrRegClass, InlineAsmType};

pub struct InlineAsmCtxt<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
    get_operand_ty: Box<dyn Fn(&'tcx hir::Expr<'tcx>) -> Ty<'tcx> + 'a>,
}

impl<'a, 'tcx> InlineAsmCtxt<'a, 'tcx> {
    pub fn new_global_asm(tcx: TyCtxt<'tcx>) -> Self {
        InlineAsmCtxt {
            tcx,
            param_env: ty::ParamEnv::empty(),
            get_operand_ty: Box::new(|e| bug!("asm operand in global asm: {e:?}")),
        }
    }

    pub fn new_in_fn(
        tcx: TyCtxt<'tcx>,
        param_env: ty::ParamEnv<'tcx>,
        get_operand_ty: impl Fn(&'tcx hir::Expr<'tcx>) -> Ty<'tcx> + 'a,
    ) -> Self {
        InlineAsmCtxt { tcx, param_env, get_operand_ty: Box::new(get_operand_ty) }
    }

    // FIXME(compiler-errors): This could use `<$ty as Pointee>::Metadata == ()`
    fn is_thin_ptr_ty(&self, ty: Ty<'tcx>) -> bool {
        // Type still may have region variables, but `Sized` does not depend
        // on those, so just erase them before querying.
        if ty.is_sized(self.tcx, self.param_env) {
            return true;
        }
        if let ty::Foreign(..) = ty.kind() {
            return true;
        }
        false
    }

    fn check_asm_operand_type(
        &self,
        idx: usize,
        reg: InlineAsmRegOrRegClass,
        expr: &'tcx hir::Expr<'tcx>,
        template: &[InlineAsmTemplatePiece],
        is_input: bool,
        tied_input: Option<(&'tcx hir::Expr<'tcx>, Option<InlineAsmType>)>,
        target_features: &FxHashSet<Symbol>,
    ) -> Option<InlineAsmType> {
        let ty = (self.get_operand_ty)(expr);
        if ty.has_non_region_infer() {
            bug!("inference variable in asm operand ty: {:?} {:?}", expr, ty);
        }
        let asm_ty_isize = match self.tcx.sess.target.pointer_width {
            16 => InlineAsmType::I16,
            32 => InlineAsmType::I32,
            64 => InlineAsmType::I64,
            _ => unreachable!(),
        };

        let asm_ty = match *ty.kind() {
            // `!` is allowed for input but not for output (issue #87802)
            ty::Never if is_input => return None,
            ty::Error(_) => return None,
            ty::Int(IntTy::I8) | ty::Uint(UintTy::U8) => Some(InlineAsmType::I8),
            ty::Int(IntTy::I16) | ty::Uint(UintTy::U16) => Some(InlineAsmType::I16),
            ty::Int(IntTy::I32) | ty::Uint(UintTy::U32) => Some(InlineAsmType::I32),
            ty::Int(IntTy::I64) | ty::Uint(UintTy::U64) => Some(InlineAsmType::I64),
            ty::Int(IntTy::I128) | ty::Uint(UintTy::U128) => Some(InlineAsmType::I128),
            ty::Int(IntTy::Isize) | ty::Uint(UintTy::Usize) => Some(asm_ty_isize),
            ty::Float(FloatTy::F32) => Some(InlineAsmType::F32),
            ty::Float(FloatTy::F64) => Some(InlineAsmType::F64),
            ty::FnPtr(_) => Some(asm_ty_isize),
            ty::RawPtr(ty::TypeAndMut { ty, mutbl: _ }) if self.is_thin_ptr_ty(ty) => {
                Some(asm_ty_isize)
            }
            ty::Adt(adt, substs) if adt.repr().simd() => {
                let fields = &adt.non_enum_variant().fields;
                let elem_ty = fields[0].ty(self.tcx, substs);
                match elem_ty.kind() {
                    ty::Never | ty::Error(_) => return None,
                    ty::Int(IntTy::I8) | ty::Uint(UintTy::U8) => {
                        Some(InlineAsmType::VecI8(fields.len() as u64))
                    }
                    ty::Int(IntTy::I16) | ty::Uint(UintTy::U16) => {
                        Some(InlineAsmType::VecI16(fields.len() as u64))
                    }
                    ty::Int(IntTy::I32) | ty::Uint(UintTy::U32) => {
                        Some(InlineAsmType::VecI32(fields.len() as u64))
                    }
                    ty::Int(IntTy::I64) | ty::Uint(UintTy::U64) => {
                        Some(InlineAsmType::VecI64(fields.len() as u64))
                    }
                    ty::Int(IntTy::I128) | ty::Uint(UintTy::U128) => {
                        Some(InlineAsmType::VecI128(fields.len() as u64))
                    }
                    ty::Int(IntTy::Isize) | ty::Uint(UintTy::Usize) => {
                        Some(match self.tcx.sess.target.pointer_width {
                            16 => InlineAsmType::VecI16(fields.len() as u64),
                            32 => InlineAsmType::VecI32(fields.len() as u64),
                            64 => InlineAsmType::VecI64(fields.len() as u64),
                            _ => unreachable!(),
                        })
                    }
                    ty::Float(FloatTy::F32) => Some(InlineAsmType::VecF32(fields.len() as u64)),
                    ty::Float(FloatTy::F64) => Some(InlineAsmType::VecF64(fields.len() as u64)),
                    _ => None,
                }
            }
            ty::Infer(_) => unreachable!(),
            _ => None,
        };
        let Some(asm_ty) = asm_ty else {
            let msg = &format!("cannot use value of type `{ty}` for inline assembly");
            let mut err = self.tcx.sess.struct_span_err(expr.span, msg);
            err.note(
                "only integers, floats, SIMD vectors, pointers and function pointers \
                 can be used as arguments for inline assembly",
            );
            err.emit();
            return None;
        };

        // Check that the type implements Copy. The only case where this can
        // possibly fail is for SIMD types which don't #[derive(Copy)].
        if !ty.is_copy_modulo_regions(self.tcx, self.param_env) {
            let msg = "arguments for inline assembly must be copyable";
            let mut err = self.tcx.sess.struct_span_err(expr.span, msg);
            err.note(&format!("`{ty}` does not implement the Copy trait"));
            err.emit();
        }

        // Ideally we wouldn't need to do this, but LLVM's register allocator
        // really doesn't like it when tied operands have different types.
        //
        // This is purely an LLVM limitation, but we have to live with it since
        // there is no way to hide this with implicit conversions.
        //
        // For the purposes of this check we only look at the `InlineAsmType`,
        // which means that pointers and integers are treated as identical (modulo
        // size).
        if let Some((in_expr, Some(in_asm_ty))) = tied_input {
            if in_asm_ty != asm_ty {
                let msg = "incompatible types for asm inout argument";
                let mut err = self.tcx.sess.struct_span_err(vec![in_expr.span, expr.span], msg);

                let in_expr_ty = (self.get_operand_ty)(in_expr);
                err.span_label(in_expr.span, &format!("type `{in_expr_ty}`"));
                err.span_label(expr.span, &format!("type `{ty}`"));
                err.note(
                    "asm inout arguments must have the same type, \
                    unless they are both pointers or integers of the same size",
                );
                err.emit();
            }

            // All of the later checks have already been done on the input, so
            // let's not emit errors and warnings twice.
            return Some(asm_ty);
        }

        // Check the type against the list of types supported by the selected
        // register class.
        let asm_arch = self.tcx.sess.asm_arch.unwrap();
        let reg_class = reg.reg_class();
        let supported_tys = reg_class.supported_types(asm_arch);
        let Some((_, feature)) = supported_tys.iter().find(|&&(t, _)| t == asm_ty) else {
            let msg = &format!("type `{ty}` cannot be used with this register class");
            let mut err = self.tcx.sess.struct_span_err(expr.span, msg);
            let supported_tys: Vec<_> =
                supported_tys.iter().map(|(t, _)| t.to_string()).collect();
            err.note(&format!(
                "register class `{}` supports these types: {}",
                reg_class.name(),
                supported_tys.join(", "),
            ));
            if let Some(suggest) = reg_class.suggest_class(asm_arch, asm_ty) {
                err.help(&format!(
                    "consider using the `{}` register class instead",
                    suggest.name()
                ));
            }
            err.emit();
            return Some(asm_ty);
        };

        // Check whether the selected type requires a target feature. Note that
        // this is different from the feature check we did earlier. While the
        // previous check checked that this register class is usable at all
        // with the currently enabled features, some types may only be usable
        // with a register class when a certain feature is enabled. We check
        // this here since it depends on the results of typeck.
        //
        // Also note that this check isn't run when the operand type is never
        // (!). In that case we still need the earlier check to verify that the
        // register class is usable at all.
        if let Some(feature) = feature {
            if !target_features.contains(&feature) {
                let msg = &format!("`{}` target feature is not enabled", feature);
                let mut err = self.tcx.sess.struct_span_err(expr.span, msg);
                err.note(&format!(
                    "this is required to use type `{}` with register class `{}`",
                    ty,
                    reg_class.name(),
                ));
                err.emit();
                return Some(asm_ty);
            }
        }

        // Check whether a modifier is suggested for using this type.
        if let Some((suggested_modifier, suggested_result)) =
            reg_class.suggest_modifier(asm_arch, asm_ty)
        {
            // Search for any use of this operand without a modifier and emit
            // the suggestion for them.
            let mut spans = vec![];
            for piece in template {
                if let &InlineAsmTemplatePiece::Placeholder { operand_idx, modifier, span } = piece
                {
                    if operand_idx == idx && modifier.is_none() {
                        spans.push(span);
                    }
                }
            }
            if !spans.is_empty() {
                let (default_modifier, default_result) =
                    reg_class.default_modifier(asm_arch).unwrap();
                self.tcx.struct_span_lint_hir(
                    lint::builtin::ASM_SUB_REGISTER,
                    expr.hir_id,
                    spans,
                    "formatting may not be suitable for sub-register argument",
                    |lint| {
                        lint.span_label(expr.span, "for this argument");
                        lint.help(&format!(
                            "use `{{{idx}:{suggested_modifier}}}` to have the register formatted as `{suggested_result}`",
                        ));
                        lint.help(&format!(
                            "or use `{{{idx}:{default_modifier}}}` to keep the default formatting of `{default_result}`",
                        ));
                        lint
                    },
                );
            }
        }

        Some(asm_ty)
    }

    pub fn check_asm(&self, asm: &hir::InlineAsm<'tcx>, enclosing_id: LocalDefId) {
        let target_features = self.tcx.asm_target_features(enclosing_id.to_def_id());
        let Some(asm_arch) = self.tcx.sess.asm_arch else {
            self.tcx.sess.delay_span_bug(DUMMY_SP, "target architecture does not support asm");
            return;
        };
        for (idx, (op, op_sp)) in asm.operands.iter().enumerate() {
            // Validate register classes against currently enabled target
            // features. We check that at least one type is available for
            // the enabled features.
            //
            // We ignore target feature requirements for clobbers: if the
            // feature is disabled then the compiler doesn't care what we
            // do with the registers.
            //
            // Note that this is only possible for explicit register
            // operands, which cannot be used in the asm string.
            if let Some(reg) = op.reg() {
                // Some explicit registers cannot be used depending on the
                // target. Reject those here.
                if let InlineAsmRegOrRegClass::Reg(reg) = reg {
                    if let InlineAsmReg::Err = reg {
                        // `validate` will panic on `Err`, as an error must
                        // already have been reported.
                        continue;
                    }
                    if let Err(msg) = reg.validate(
                        asm_arch,
                        self.tcx.sess.relocation_model(),
                        &target_features,
                        &self.tcx.sess.target,
                        op.is_clobber(),
                    ) {
                        let msg = format!("cannot use register `{}`: {}", reg.name(), msg);
                        self.tcx.sess.struct_span_err(*op_sp, &msg).emit();
                        continue;
                    }
                }

                if !op.is_clobber() {
                    let mut missing_required_features = vec![];
                    let reg_class = reg.reg_class();
                    if let InlineAsmRegClass::Err = reg_class {
                        continue;
                    }
                    for &(_, feature) in reg_class.supported_types(asm_arch) {
                        match feature {
                            Some(feature) => {
                                if target_features.contains(&feature) {
                                    missing_required_features.clear();
                                    break;
                                } else {
                                    missing_required_features.push(feature);
                                }
                            }
                            None => {
                                missing_required_features.clear();
                                break;
                            }
                        }
                    }

                    // We are sorting primitive strs here and can use unstable sort here
                    missing_required_features.sort_unstable();
                    missing_required_features.dedup();
                    match &missing_required_features[..] {
                        [] => {}
                        [feature] => {
                            let msg = format!(
                                "register class `{}` requires the `{}` target feature",
                                reg_class.name(),
                                feature
                            );
                            self.tcx.sess.struct_span_err(*op_sp, &msg).emit();
                            // register isn't enabled, don't do more checks
                            continue;
                        }
                        features => {
                            let msg = format!(
                                "register class `{}` requires at least one of the following target features: {}",
                                reg_class.name(),
                                features
                                    .iter()
                                    .map(|f| f.as_str())
                                    .intersperse(", ")
                                    .collect::<String>(),
                            );
                            self.tcx.sess.struct_span_err(*op_sp, &msg).emit();
                            // register isn't enabled, don't do more checks
                            continue;
                        }
                    }
                }
            }

            match *op {
                hir::InlineAsmOperand::In { reg, expr } => {
                    self.check_asm_operand_type(
                        idx,
                        reg,
                        expr,
                        asm.template,
                        true,
                        None,
                        &target_features,
                    );
                }
                hir::InlineAsmOperand::Out { reg, late: _, expr } => {
                    if let Some(expr) = expr {
                        self.check_asm_operand_type(
                            idx,
                            reg,
                            expr,
                            asm.template,
                            false,
                            None,
                            &target_features,
                        );
                    }
                }
                hir::InlineAsmOperand::InOut { reg, late: _, expr } => {
                    self.check_asm_operand_type(
                        idx,
                        reg,
                        expr,
                        asm.template,
                        false,
                        None,
                        &target_features,
                    );
                }
                hir::InlineAsmOperand::SplitInOut { reg, late: _, in_expr, out_expr } => {
                    let in_ty = self.check_asm_operand_type(
                        idx,
                        reg,
                        in_expr,
                        asm.template,
                        true,
                        None,
                        &target_features,
                    );
                    if let Some(out_expr) = out_expr {
                        self.check_asm_operand_type(
                            idx,
                            reg,
                            out_expr,
                            asm.template,
                            false,
                            Some((in_expr, in_ty)),
                            &target_features,
                        );
                    }
                }
                // No special checking is needed for these:
                // - Typeck has checked that Const operands are integers.
                // - AST lowering guarantees that SymStatic points to a static.
                hir::InlineAsmOperand::Const { .. } | hir::InlineAsmOperand::SymStatic { .. } => {}
                // Check that sym actually points to a function. Later passes
                // depend on this.
                hir::InlineAsmOperand::SymFn { anon_const } => {
                    let ty = self.tcx.type_of(anon_const.def_id);
                    match ty.kind() {
                        ty::Never | ty::Error(_) => {}
                        ty::FnDef(..) => {}
                        _ => {
                            let mut err =
                                self.tcx.sess.struct_span_err(*op_sp, "invalid `sym` operand");
                            err.span_label(
                                self.tcx.def_span(anon_const.def_id),
                                &format!("is {} `{}`", ty.kind().article(), ty),
                            );
                            err.help("`sym` operands must refer to either a function or a static");
                            err.emit();
                        }
                    };
                }
            }
        }
    }
}
