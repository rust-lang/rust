use rustc_ast::InlineAsmTemplatePiece;
use rustc_errors::struct_span_err;
use rustc_hir as hir;
use rustc_hir::def::{DefKind, Res};
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_index::vec::Idx;
use rustc_middle::ty::layout::{LayoutError, SizeSkeleton};
use rustc_middle::ty::query::Providers;
use rustc_middle::ty::{self, FloatTy, IntTy, Ty, TyCtxt, UintTy};
use rustc_session::lint;
use rustc_span::{sym, Span, Symbol, DUMMY_SP};
use rustc_target::abi::{Pointer, VariantIdx};
use rustc_target::asm::{InlineAsmRegOrRegClass, InlineAsmType};
use rustc_target::spec::abi::Abi::RustIntrinsic;

fn check_mod_intrinsics(tcx: TyCtxt<'_>, module_def_id: LocalDefId) {
    tcx.hir().visit_item_likes_in_module(module_def_id, &mut ItemVisitor { tcx }.as_deep_visitor());
}

pub fn provide(providers: &mut Providers) {
    *providers = Providers { check_mod_intrinsics, ..*providers };
}

struct ItemVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
}

struct ExprVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    typeck_results: &'tcx ty::TypeckResults<'tcx>,
    param_env: ty::ParamEnv<'tcx>,
}

/// If the type is `Option<T>`, it will return `T`, otherwise
/// the type itself. Works on most `Option`-like types.
fn unpack_option_like<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    let (def, substs) = match *ty.kind() {
        ty::Adt(def, substs) => (def, substs),
        _ => return ty,
    };

    if def.variants.len() == 2 && !def.repr.c() && def.repr.int.is_none() {
        let data_idx;

        let one = VariantIdx::new(1);
        let zero = VariantIdx::new(0);

        if def.variants[zero].fields.is_empty() {
            data_idx = one;
        } else if def.variants[one].fields.is_empty() {
            data_idx = zero;
        } else {
            return ty;
        }

        if def.variants[data_idx].fields.len() == 1 {
            return def.variants[data_idx].fields[0].ty(tcx, substs);
        }
    }

    ty
}

impl<'tcx> ExprVisitor<'tcx> {
    fn def_id_is_transmute(&self, def_id: DefId) -> bool {
        self.tcx.fn_sig(def_id).abi() == RustIntrinsic
            && self.tcx.item_name(def_id) == sym::transmute
    }

    fn check_transmute(&self, span: Span, from: Ty<'tcx>, to: Ty<'tcx>) {
        let sk_from = SizeSkeleton::compute(from, self.tcx, self.param_env);
        let sk_to = SizeSkeleton::compute(to, self.tcx, self.param_env);

        // Check for same size using the skeletons.
        if let (Ok(sk_from), Ok(sk_to)) = (sk_from, sk_to) {
            if sk_from.same_size(sk_to) {
                return;
            }

            // Special-case transmuting from `typeof(function)` and
            // `Option<typeof(function)>` to present a clearer error.
            let from = unpack_option_like(self.tcx, from);
            if let (&ty::FnDef(..), SizeSkeleton::Known(size_to)) = (from.kind(), sk_to) {
                if size_to == Pointer.size(&self.tcx) {
                    struct_span_err!(self.tcx.sess, span, E0591, "can't transmute zero-sized type")
                        .note(&format!("source type: {}", from))
                        .note(&format!("target type: {}", to))
                        .help("cast with `as` to a pointer instead")
                        .emit();
                    return;
                }
            }
        }

        // Try to display a sensible error with as much information as possible.
        let skeleton_string = |ty: Ty<'tcx>, sk| match sk {
            Ok(SizeSkeleton::Known(size)) => format!("{} bits", size.bits()),
            Ok(SizeSkeleton::Pointer { tail, .. }) => format!("pointer to `{}`", tail),
            Err(LayoutError::Unknown(bad)) => {
                if bad == ty {
                    "this type does not have a fixed size".to_owned()
                } else {
                    format!("size can vary because of {}", bad)
                }
            }
            Err(err) => err.to_string(),
        };

        let mut err = struct_span_err!(
            self.tcx.sess,
            span,
            E0512,
            "cannot transmute between types of different sizes, \
                                        or dependently-sized types"
        );
        if from == to {
            err.note(&format!("`{}` does not have a fixed size", from));
        } else {
            err.note(&format!("source type: `{}` ({})", from, skeleton_string(from, sk_from)))
                .note(&format!("target type: `{}` ({})", to, skeleton_string(to, sk_to)));
        }
        err.emit()
    }

    fn is_thin_ptr_ty(&self, ty: Ty<'tcx>) -> bool {
        if ty.is_sized(self.tcx.at(DUMMY_SP), self.param_env) {
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
        expr: &hir::Expr<'tcx>,
        template: &[InlineAsmTemplatePiece],
        is_input: bool,
        tied_input: Option<(&hir::Expr<'tcx>, Option<InlineAsmType>)>,
        target_features: &[Symbol],
    ) -> Option<InlineAsmType> {
        // Check the type against the allowed types for inline asm.
        let ty = self.typeck_results.expr_ty_adjusted(expr);
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
            ty::Adt(adt, substs) if adt.repr.simd() => {
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
            _ => None,
        };
        let asm_ty = match asm_ty {
            Some(asm_ty) => asm_ty,
            None => {
                let msg = &format!("cannot use value of type `{}` for inline assembly", ty);
                let mut err = self.tcx.sess.struct_span_err(expr.span, msg);
                err.note(
                    "only integers, floats, SIMD vectors, pointers and function pointers \
                     can be used as arguments for inline assembly",
                );
                err.emit();
                return None;
            }
        };

        // Check that the type implements Copy. The only case where this can
        // possibly fail is for SIMD types which don't #[derive(Copy)].
        if !ty.is_copy_modulo_regions(self.tcx.at(DUMMY_SP), self.param_env) {
            let msg = "arguments for inline assembly must be copyable";
            let mut err = self.tcx.sess.struct_span_err(expr.span, msg);
            err.note(&format!("`{}` does not implement the Copy trait", ty));
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
                err.span_label(
                    in_expr.span,
                    &format!("type `{}`", self.typeck_results.expr_ty_adjusted(in_expr)),
                );
                err.span_label(expr.span, &format!("type `{}`", ty));
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
        let feature = match supported_tys.iter().find(|&&(t, _)| t == asm_ty) {
            Some((_, feature)) => feature,
            None => {
                let msg = &format!("type `{}` cannot be used with this register class", ty);
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
            }
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
            if !self.tcx.sess.target_features.contains(&feature)
                && !target_features.contains(&feature)
            {
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
                    |lint| {
                        let msg = "formatting may not be suitable for sub-register argument";
                        let mut err = lint.build(msg);
                        err.span_label(expr.span, "for this argument");
                        err.help(&format!(
                            "use the `{}` modifier to have the register formatted as `{}`",
                            suggested_modifier, suggested_result,
                        ));
                        err.help(&format!(
                            "or use the `{}` modifier to keep the default formatting of `{}`",
                            default_modifier, default_result,
                        ));
                        err.emit();
                    },
                );
            }
        }

        Some(asm_ty)
    }

    fn check_asm(&self, asm: &hir::InlineAsm<'tcx>, hir_id: hir::HirId) {
        let hir = self.tcx.hir();
        let enclosing_id = hir.enclosing_body_owner(hir_id);
        let enclosing_def_id = hir.local_def_id(enclosing_id).to_def_id();
        let attrs = self.tcx.codegen_fn_attrs(enclosing_def_id);
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
                if !op.is_clobber() {
                    let mut missing_required_features = vec![];
                    let reg_class = reg.reg_class();
                    for &(_, feature) in reg_class.supported_types(self.tcx.sess.asm_arch.unwrap())
                    {
                        match feature {
                            Some(feature) => {
                                if self.tcx.sess.target_features.contains(&feature)
                                    || attrs.target_features.contains(&feature)
                                {
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
                hir::InlineAsmOperand::In { reg, ref expr } => {
                    self.check_asm_operand_type(
                        idx,
                        reg,
                        expr,
                        asm.template,
                        true,
                        None,
                        &attrs.target_features,
                    );
                }
                hir::InlineAsmOperand::Out { reg, late: _, ref expr } => {
                    if let Some(expr) = expr {
                        self.check_asm_operand_type(
                            idx,
                            reg,
                            expr,
                            asm.template,
                            false,
                            None,
                            &attrs.target_features,
                        );
                    }
                }
                hir::InlineAsmOperand::InOut { reg, late: _, ref expr } => {
                    self.check_asm_operand_type(
                        idx,
                        reg,
                        expr,
                        asm.template,
                        false,
                        None,
                        &attrs.target_features,
                    );
                }
                hir::InlineAsmOperand::SplitInOut { reg, late: _, ref in_expr, ref out_expr } => {
                    let in_ty = self.check_asm_operand_type(
                        idx,
                        reg,
                        in_expr,
                        asm.template,
                        true,
                        None,
                        &attrs.target_features,
                    );
                    if let Some(out_expr) = out_expr {
                        self.check_asm_operand_type(
                            idx,
                            reg,
                            out_expr,
                            asm.template,
                            false,
                            Some((in_expr, in_ty)),
                            &attrs.target_features,
                        );
                    }
                }
                hir::InlineAsmOperand::Const { .. } | hir::InlineAsmOperand::Sym { .. } => {}
            }
        }
    }
}

impl<'tcx> Visitor<'tcx> for ItemVisitor<'tcx> {
    fn visit_nested_body(&mut self, body_id: hir::BodyId) {
        let owner_def_id = self.tcx.hir().body_owner_def_id(body_id);
        let body = self.tcx.hir().body(body_id);
        let param_env = self.tcx.param_env(owner_def_id.to_def_id());
        let typeck_results = self.tcx.typeck(owner_def_id);
        ExprVisitor { tcx: self.tcx, param_env, typeck_results }.visit_body(body);
        self.visit_body(body);
    }
}

impl<'tcx> Visitor<'tcx> for ExprVisitor<'tcx> {
    fn visit_expr(&mut self, expr: &'tcx hir::Expr<'tcx>) {
        match expr.kind {
            hir::ExprKind::Path(ref qpath) => {
                let res = self.typeck_results.qpath_res(qpath, expr.hir_id);
                if let Res::Def(DefKind::Fn, did) = res {
                    if self.def_id_is_transmute(did) {
                        let typ = self.typeck_results.node_type(expr.hir_id);
                        let sig = typ.fn_sig(self.tcx);
                        let from = sig.inputs().skip_binder()[0];
                        let to = sig.output().skip_binder();
                        self.check_transmute(expr.span, from, to);
                    }
                }
            }

            hir::ExprKind::InlineAsm(asm) => self.check_asm(asm, expr.hir_id),

            _ => {}
        }

        intravisit::walk_expr(self, expr);
    }
}
