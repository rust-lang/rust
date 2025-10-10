use rustc_abi::FieldIdx;
use rustc_ast::InlineAsmTemplatePiece;
use rustc_data_structures::fx::FxIndexSet;
use rustc_hir::def_id::DefId;
use rustc_hir::{self as hir, LangItem};
use rustc_middle::bug;
use rustc_middle::ty::{self, Article, FloatTy, IntTy, Ty, TyCtxt, TypeVisitableExt, UintTy};
use rustc_session::lint;
use rustc_span::def_id::LocalDefId;
use rustc_span::{Span, Symbol, sym};
use rustc_target::asm::{
    InlineAsmReg, InlineAsmRegClass, InlineAsmRegOrRegClass, InlineAsmType, ModifierInfo,
};
use rustc_trait_selection::infer::InferCtxtExt;

use crate::FnCtxt;
use crate::errors::RegisterTypeUnstable;

pub(crate) struct InlineAsmCtxt<'a, 'tcx> {
    target_features: &'tcx FxIndexSet<Symbol>,
    fcx: &'a FnCtxt<'a, 'tcx>,
}

enum NonAsmTypeReason<'tcx> {
    UnevaluatedSIMDArrayLength(DefId, ty::Const<'tcx>),
    Invalid(Ty<'tcx>),
    InvalidElement(DefId, Ty<'tcx>),
    NotSizedPtr(Ty<'tcx>),
    EmptySIMDArray(Ty<'tcx>),
}

impl<'a, 'tcx> InlineAsmCtxt<'a, 'tcx> {
    pub(crate) fn new(fcx: &'a FnCtxt<'a, 'tcx>, def_id: LocalDefId) -> Self {
        InlineAsmCtxt { target_features: fcx.tcx.asm_target_features(def_id), fcx }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        self.fcx.tcx
    }

    fn expr_ty(&self, expr: &hir::Expr<'tcx>) -> Ty<'tcx> {
        let ty = self.fcx.typeck_results.borrow().expr_ty_adjusted(expr);
        let ty = self.fcx.try_structurally_resolve_type(expr.span, ty);
        if ty.has_non_region_infer() {
            Ty::new_misc_error(self.tcx())
        } else {
            self.tcx().erase_and_anonymize_regions(ty)
        }
    }

    // FIXME(compiler-errors): This could use `<$ty as Pointee>::Metadata == ()`
    fn is_thin_ptr_ty(&self, span: Span, ty: Ty<'tcx>) -> bool {
        // Type still may have region variables, but `Sized` does not depend
        // on those, so just erase them before querying.
        if self.fcx.type_is_sized_modulo_regions(self.fcx.param_env, ty) {
            return true;
        }
        if let ty::Foreign(..) = self.fcx.try_structurally_resolve_type(span, ty).kind() {
            return true;
        }
        false
    }

    fn get_asm_ty(
        &self,
        span: Span,
        ty: Ty<'tcx>,
    ) -> Result<InlineAsmType, NonAsmTypeReason<'tcx>> {
        let asm_ty_isize = match self.tcx().sess.target.pointer_width {
            16 => InlineAsmType::I16,
            32 => InlineAsmType::I32,
            64 => InlineAsmType::I64,
            width => bug!("unsupported pointer width: {width}"),
        };

        match *ty.kind() {
            ty::Int(IntTy::I8) | ty::Uint(UintTy::U8) => Ok(InlineAsmType::I8),
            ty::Int(IntTy::I16) | ty::Uint(UintTy::U16) => Ok(InlineAsmType::I16),
            ty::Int(IntTy::I32) | ty::Uint(UintTy::U32) => Ok(InlineAsmType::I32),
            ty::Int(IntTy::I64) | ty::Uint(UintTy::U64) => Ok(InlineAsmType::I64),
            ty::Int(IntTy::I128) | ty::Uint(UintTy::U128) => Ok(InlineAsmType::I128),
            ty::Int(IntTy::Isize) | ty::Uint(UintTy::Usize) => Ok(asm_ty_isize),
            ty::Float(FloatTy::F16) => Ok(InlineAsmType::F16),
            ty::Float(FloatTy::F32) => Ok(InlineAsmType::F32),
            ty::Float(FloatTy::F64) => Ok(InlineAsmType::F64),
            ty::Float(FloatTy::F128) => Ok(InlineAsmType::F128),
            ty::FnPtr(..) => Ok(asm_ty_isize),
            ty::RawPtr(elem_ty, _) => {
                if self.is_thin_ptr_ty(span, elem_ty) {
                    Ok(asm_ty_isize)
                } else {
                    Err(NonAsmTypeReason::NotSizedPtr(ty))
                }
            }
            ty::Adt(adt, args) if adt.repr().simd() => {
                let fields = &adt.non_enum_variant().fields;
                if fields.is_empty() {
                    return Err(NonAsmTypeReason::EmptySIMDArray(ty));
                }
                let field = &fields[FieldIdx::ZERO];
                let elem_ty = field.ty(self.tcx(), args);

                let (size, ty) = match *elem_ty.kind() {
                    ty::Array(ty, len) => {
                        // FIXME: `try_structurally_resolve_const` doesn't eval consts
                        // in the old solver.
                        let len = if self.fcx.next_trait_solver() {
                            self.fcx.try_structurally_resolve_const(span, len)
                        } else {
                            self.fcx.tcx.normalize_erasing_regions(
                                self.fcx.typing_env(self.fcx.param_env),
                                len,
                            )
                        };
                        if let Some(len) = len.try_to_target_usize(self.tcx()) {
                            (len, ty)
                        } else {
                            return Err(NonAsmTypeReason::UnevaluatedSIMDArrayLength(
                                field.did, len,
                            ));
                        }
                    }
                    _ => (fields.len() as u64, elem_ty),
                };

                match ty.kind() {
                    ty::Int(IntTy::I8) | ty::Uint(UintTy::U8) => Ok(InlineAsmType::VecI8(size)),
                    ty::Int(IntTy::I16) | ty::Uint(UintTy::U16) => Ok(InlineAsmType::VecI16(size)),
                    ty::Int(IntTy::I32) | ty::Uint(UintTy::U32) => Ok(InlineAsmType::VecI32(size)),
                    ty::Int(IntTy::I64) | ty::Uint(UintTy::U64) => Ok(InlineAsmType::VecI64(size)),
                    ty::Int(IntTy::I128) | ty::Uint(UintTy::U128) => {
                        Ok(InlineAsmType::VecI128(size))
                    }
                    ty::Int(IntTy::Isize) | ty::Uint(UintTy::Usize) => {
                        Ok(match self.tcx().sess.target.pointer_width {
                            16 => InlineAsmType::VecI16(size),
                            32 => InlineAsmType::VecI32(size),
                            64 => InlineAsmType::VecI64(size),
                            width => bug!("unsupported pointer width: {width}"),
                        })
                    }
                    ty::Float(FloatTy::F16) => Ok(InlineAsmType::VecF16(size)),
                    ty::Float(FloatTy::F32) => Ok(InlineAsmType::VecF32(size)),
                    ty::Float(FloatTy::F64) => Ok(InlineAsmType::VecF64(size)),
                    ty::Float(FloatTy::F128) => Ok(InlineAsmType::VecF128(size)),
                    _ => Err(NonAsmTypeReason::InvalidElement(field.did, ty)),
                }
            }
            ty::Infer(_) => bug!("unexpected infer ty in asm operand"),
            _ => Err(NonAsmTypeReason::Invalid(ty)),
        }
    }

    fn check_asm_operand_type(
        &self,
        idx: usize,
        reg: InlineAsmRegOrRegClass,
        expr: &'tcx hir::Expr<'tcx>,
        template: &[InlineAsmTemplatePiece],
        is_input: bool,
        tied_input: Option<(&'tcx hir::Expr<'tcx>, Option<InlineAsmType>)>,
    ) -> Option<InlineAsmType> {
        let ty = self.expr_ty(expr);
        if ty.has_non_region_infer() {
            bug!("inference variable in asm operand ty: {:?} {:?}", expr, ty);
        }

        let asm_ty = match *ty.kind() {
            // `!` is allowed for input but not for output (issue #87802)
            ty::Never if is_input => return None,
            _ if ty.references_error() => return None,
            ty::Adt(adt, args) if self.tcx().is_lang_item(adt.did(), LangItem::MaybeUninit) => {
                let fields = &adt.non_enum_variant().fields;
                let ty = fields[FieldIdx::ONE].ty(self.tcx(), args);
                // FIXME: Are we just trying to map to the `T` in `MaybeUninit<T>`?
                // If so, just get it from the args.
                let ty::Adt(ty, args) = ty.kind() else {
                    unreachable!("expected first field of `MaybeUninit` to be an ADT")
                };
                assert!(
                    ty.is_manually_drop(),
                    "expected first field of `MaybeUninit` to be `ManuallyDrop`"
                );
                let fields = &ty.non_enum_variant().fields;
                let ty = fields[FieldIdx::ZERO].ty(self.tcx(), args);
                self.get_asm_ty(expr.span, ty)
            }
            _ => self.get_asm_ty(expr.span, ty),
        };
        let asm_ty = match asm_ty {
            Ok(asm_ty) => asm_ty,
            Err(reason) => {
                match reason {
                    NonAsmTypeReason::UnevaluatedSIMDArrayLength(did, len) => {
                        let msg = format!("cannot evaluate SIMD vector length `{len}`");
                        self.fcx
                            .dcx()
                            .struct_span_err(self.tcx().def_span(did), msg)
                            .with_span_note(
                                expr.span,
                                "SIMD vector length needs to be known statically for use in `asm!`",
                            )
                            .emit();
                    }
                    NonAsmTypeReason::Invalid(ty) => {
                        let msg = format!("cannot use value of type `{ty}` for inline assembly");
                        self.fcx.dcx().struct_span_err(expr.span, msg).with_note(
                            "only integers, floats, SIMD vectors, pointers and function pointers \
                            can be used as arguments for inline assembly",
                        ).emit();
                    }
                    NonAsmTypeReason::NotSizedPtr(ty) => {
                        let msg = format!(
                            "cannot use value of unsized pointer type `{ty}` for inline assembly"
                        );
                        self.fcx
                            .dcx()
                            .struct_span_err(expr.span, msg)
                            .with_note("only sized pointers can be used in inline assembly")
                            .emit();
                    }
                    NonAsmTypeReason::InvalidElement(did, ty) => {
                        let msg = format!(
                            "cannot use SIMD vector with element type `{ty}` for inline assembly"
                        );
                        self.fcx.dcx()
                        .struct_span_err(self.tcx().def_span(did), msg).with_span_note(
                            expr.span,
                            "only integers, floats, SIMD vectors, pointers and function pointers \
                            can be used as arguments for inline assembly",
                        ).emit();
                    }
                    NonAsmTypeReason::EmptySIMDArray(ty) => {
                        let msg = format!("use of empty SIMD vector `{ty}`");
                        self.fcx.dcx().struct_span_err(expr.span, msg).emit();
                    }
                }
                return None;
            }
        };

        // Check that the type implements Copy. The only case where this can
        // possibly fail is for SIMD types which don't #[derive(Copy)].
        if !self.fcx.type_is_copy_modulo_regions(self.fcx.param_env, ty) {
            let msg = "arguments for inline assembly must be copyable";
            self.fcx
                .dcx()
                .struct_span_err(expr.span, msg)
                .with_note(format!("`{ty}` does not implement the Copy trait"))
                .emit();
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
                let in_expr_ty = self.expr_ty(in_expr);
                self.fcx
                    .dcx()
                    .struct_span_err(vec![in_expr.span, expr.span], msg)
                    .with_span_label(in_expr.span, format!("type `{in_expr_ty}`"))
                    .with_span_label(expr.span, format!("type `{ty}`"))
                    .with_note(
                        "asm inout arguments must have the same type, \
                        unless they are both pointers or integers of the same size",
                    )
                    .emit();
            }

            // All of the later checks have already been done on the input, so
            // let's not emit errors and warnings twice.
            return Some(asm_ty);
        }

        // Check the type against the list of types supported by the selected
        // register class.
        let asm_arch = self.tcx().sess.asm_arch.unwrap();
        let allow_experimental_reg = self.tcx().features().asm_experimental_reg();
        let reg_class = reg.reg_class();
        let supported_tys = reg_class.supported_types(asm_arch, allow_experimental_reg);
        let Some((_, feature)) = supported_tys.iter().find(|&&(t, _)| t == asm_ty) else {
            let mut err = if !allow_experimental_reg
                && reg_class.supported_types(asm_arch, true).iter().any(|&(t, _)| t == asm_ty)
            {
                self.tcx().sess.create_feature_err(
                    RegisterTypeUnstable { span: expr.span, ty },
                    sym::asm_experimental_reg,
                )
            } else {
                let msg = format!("type `{ty}` cannot be used with this register class");
                let mut err = self.fcx.dcx().struct_span_err(expr.span, msg);
                let supported_tys: Vec<_> =
                    supported_tys.iter().map(|(t, _)| t.to_string()).collect();
                err.note(format!(
                    "register class `{}` supports these types: {}",
                    reg_class.name(),
                    supported_tys.join(", "),
                ));
                err
            };
            if let Some(suggest) = reg_class.suggest_class(asm_arch, asm_ty) {
                err.help(format!("consider using the `{}` register class instead", suggest.name()));
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
            if !self.target_features.contains(feature) {
                let msg = format!("`{feature}` target feature is not enabled");
                self.fcx
                    .dcx()
                    .struct_span_err(expr.span, msg)
                    .with_note(format!(
                        "this is required to use type `{}` with register class `{}`",
                        ty,
                        reg_class.name(),
                    ))
                    .emit();
                return Some(asm_ty);
            }
        }

        // Check whether a modifier is suggested for using this type.
        if let Some(ModifierInfo {
            modifier: suggested_modifier,
            result: suggested_result,
            size: suggested_size,
        }) = reg_class.suggest_modifier(asm_arch, asm_ty)
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
                let ModifierInfo {
                    modifier: default_modifier,
                    result: default_result,
                    size: default_size,
                } = reg_class.default_modifier(asm_arch).unwrap();
                self.tcx().node_span_lint(
                    lint::builtin::ASM_SUB_REGISTER,
                    expr.hir_id,
                    spans,
                    |lint| {
                        lint.primary_message("formatting may not be suitable for sub-register argument");
                        lint.span_label(expr.span, "for this argument");
                        lint.help(format!(
                            "use `{{{idx}:{suggested_modifier}}}` to have the register formatted as `{suggested_result}` (for {suggested_size}-bit values)",
                        ));
                        lint.help(format!(
                            "or use `{{{idx}:{default_modifier}}}` to keep the default formatting of `{default_result}` (for {default_size}-bit values)",
                        ));
                    },
                );
            }
        }

        Some(asm_ty)
    }

    pub(crate) fn check_asm(&self, asm: &hir::InlineAsm<'tcx>) {
        let Some(asm_arch) = self.tcx().sess.asm_arch else {
            self.fcx.dcx().delayed_bug("target architecture does not support asm");
            return;
        };
        let allow_experimental_reg = self.tcx().features().asm_experimental_reg();
        for (idx, &(op, op_sp)) in asm.operands.iter().enumerate() {
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
                        self.tcx().sess.relocation_model(),
                        self.target_features,
                        &self.tcx().sess.target,
                        op.is_clobber(),
                    ) {
                        let msg = format!("cannot use register `{}`: {}", reg.name(), msg);
                        self.fcx.dcx().span_err(op_sp, msg);
                        continue;
                    }
                }

                if !op.is_clobber() {
                    let mut missing_required_features = vec![];
                    let reg_class = reg.reg_class();
                    if let InlineAsmRegClass::Err = reg_class {
                        continue;
                    }
                    for &(_, feature) in reg_class.supported_types(asm_arch, allow_experimental_reg)
                    {
                        match feature {
                            Some(feature) => {
                                if self.target_features.contains(&feature) {
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
                            self.fcx.dcx().span_err(op_sp, msg);
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
                            self.fcx.dcx().span_err(op_sp, msg);
                            // register isn't enabled, don't do more checks
                            continue;
                        }
                    }
                }
            }

            match op {
                hir::InlineAsmOperand::In { reg, expr } => {
                    self.check_asm_operand_type(idx, reg, expr, asm.template, true, None);
                }
                hir::InlineAsmOperand::Out { reg, late: _, expr } => {
                    if let Some(expr) = expr {
                        self.check_asm_operand_type(idx, reg, expr, asm.template, false, None);
                    }
                }
                hir::InlineAsmOperand::InOut { reg, late: _, expr } => {
                    self.check_asm_operand_type(idx, reg, expr, asm.template, false, None);
                }
                hir::InlineAsmOperand::SplitInOut { reg, late: _, in_expr, out_expr } => {
                    let in_ty =
                        self.check_asm_operand_type(idx, reg, in_expr, asm.template, true, None);
                    if let Some(out_expr) = out_expr {
                        self.check_asm_operand_type(
                            idx,
                            reg,
                            out_expr,
                            asm.template,
                            false,
                            Some((in_expr, in_ty)),
                        );
                    }
                }
                hir::InlineAsmOperand::Const { anon_const } => {
                    let ty = self.expr_ty(self.tcx().hir_body(anon_const.body).value);
                    match ty.kind() {
                        ty::Error(_) => {}
                        _ if ty.is_integral() => {}
                        _ => {
                            self.fcx
                                .dcx()
                                .struct_span_err(op_sp, "invalid type for `const` operand")
                                .with_span_label(
                                    self.tcx().def_span(anon_const.def_id),
                                    format!("is {} `{}`", ty.kind().article(), ty),
                                )
                                .with_help("`const` operands must be of an integer type")
                                .emit();
                        }
                    }
                }
                // Typeck has checked that SymFn refers to a function.
                hir::InlineAsmOperand::SymFn { expr } => {
                    let ty = self.expr_ty(expr);
                    match ty.kind() {
                        ty::FnDef(..) => {}
                        ty::Error(_) => {}
                        _ => {
                            self.fcx
                                .dcx()
                                .struct_span_err(op_sp, "invalid `sym` operand")
                                .with_span_label(
                                    expr.span,
                                    format!("is {} `{}`", ty.kind().article(), ty),
                                )
                                .with_help(
                                    "`sym` operands must refer to either a function or a static",
                                )
                                .emit();
                        }
                    }
                }
                // AST lowering guarantees that SymStatic points to a static.
                hir::InlineAsmOperand::SymStatic { .. } => {}
                // No special checking is needed for labels.
                hir::InlineAsmOperand::Label { .. } => {}
            }
        }
    }
}
