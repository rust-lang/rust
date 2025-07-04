use rustc_abi::{self as abi, FIRST_VARIANT};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::layout::{HasTyCtxt, HasTypingEnv, LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt};
use rustc_middle::{bug, mir};
use rustc_session::config::OptLevel;
use rustc_span::{DUMMY_SP, Span};
use tracing::{debug, instrument};

use super::operand::{OperandRef, OperandValue};
use super::place::PlaceRef;
use super::{FunctionCx, LocalRef};
use crate::common::{IntPredicate, TypeKind};
use crate::traits::*;
use crate::{MemFlags, base};

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    #[instrument(level = "trace", skip(self, bx))]
    pub(crate) fn codegen_rvalue(
        &mut self,
        bx: &mut Bx,
        dest: PlaceRef<'tcx, Bx::Value>,
        rvalue: &mir::Rvalue<'tcx>,
    ) {
        match *rvalue {
            mir::Rvalue::Use(ref operand) => {
                let cg_operand = self.codegen_operand(bx, operand);
                // FIXME: consider not copying constants through stack. (Fixable by codegen'ing
                // constants into `OperandValue::Ref`; why don’t we do that yet if we don’t?)
                cg_operand.val.store(bx, dest);
            }

            mir::Rvalue::Cast(
                mir::CastKind::PointerCoercion(PointerCoercion::Unsize, _),
                ref source,
                _,
            ) => {
                // The destination necessarily contains a wide pointer, so if
                // it's a scalar pair, it's a wide pointer or newtype thereof.
                if bx.cx().is_backend_scalar_pair(dest.layout) {
                    // Into-coerce of a thin pointer to a wide pointer -- just
                    // use the operand path.
                    let temp = self.codegen_rvalue_operand(bx, rvalue);
                    temp.val.store(bx, dest);
                    return;
                }

                // Unsize of a nontrivial struct. I would prefer for
                // this to be eliminated by MIR building, but
                // `CoerceUnsized` can be passed by a where-clause,
                // so the (generic) MIR may not be able to expand it.
                let operand = self.codegen_operand(bx, source);
                match operand.val {
                    OperandValue::Pair(..) | OperandValue::Immediate(_) => {
                        // Unsize from an immediate structure. We don't
                        // really need a temporary alloca here, but
                        // avoiding it would require us to have
                        // `coerce_unsized_into` use `extractvalue` to
                        // index into the struct, and this case isn't
                        // important enough for it.
                        debug!("codegen_rvalue: creating ugly alloca");
                        let scratch = PlaceRef::alloca(bx, operand.layout);
                        scratch.storage_live(bx);
                        operand.val.store(bx, scratch);
                        base::coerce_unsized_into(bx, scratch, dest);
                        scratch.storage_dead(bx);
                    }
                    OperandValue::Ref(val) => {
                        if val.llextra.is_some() {
                            bug!("unsized coercion on an unsized rvalue");
                        }
                        base::coerce_unsized_into(bx, val.with_type(operand.layout), dest);
                    }
                    OperandValue::ZeroSized => {
                        bug!("unsized coercion on a ZST rvalue");
                    }
                }
            }

            mir::Rvalue::Cast(mir::CastKind::Transmute, ref operand, _ty) => {
                let src = self.codegen_operand(bx, operand);
                self.codegen_transmute(bx, src, dest);
            }

            mir::Rvalue::Repeat(ref elem, count) => {
                // Do not generate the loop for zero-sized elements or empty arrays.
                if dest.layout.is_zst() {
                    return;
                }

                // When the element is a const with all bytes uninit, emit a single memset that
                // writes undef to the entire destination.
                if let mir::Operand::Constant(const_op) = elem {
                    let val = self.eval_mir_constant(const_op);
                    if val.all_bytes_uninit(self.cx.tcx()) {
                        let size = bx.const_usize(dest.layout.size.bytes());
                        bx.memset(
                            dest.val.llval,
                            bx.const_undef(bx.type_i8()),
                            size,
                            dest.val.align,
                            MemFlags::empty(),
                        );
                        return;
                    }
                }

                let cg_elem = self.codegen_operand(bx, elem);

                let try_init_all_same = |bx: &mut Bx, v| {
                    let start = dest.val.llval;
                    let size = bx.const_usize(dest.layout.size.bytes());

                    // Use llvm.memset.p0i8.* to initialize all same byte arrays
                    if let Some(int) = bx.cx().const_to_opt_u128(v, false) {
                        let bytes = &int.to_le_bytes()[..cg_elem.layout.size.bytes_usize()];
                        let first = bytes[0];
                        if bytes[1..].iter().all(|&b| b == first) {
                            let fill = bx.cx().const_u8(first);
                            bx.memset(start, fill, size, dest.val.align, MemFlags::empty());
                            return true;
                        }
                    }

                    // Use llvm.memset.p0i8.* to initialize byte arrays
                    let v = bx.from_immediate(v);
                    if bx.cx().val_ty(v) == bx.cx().type_i8() {
                        bx.memset(start, v, size, dest.val.align, MemFlags::empty());
                        return true;
                    }
                    false
                };

                match cg_elem.val {
                    OperandValue::Immediate(v) => {
                        if try_init_all_same(bx, v) {
                            return;
                        }
                    }
                    _ => (),
                }

                let count = self
                    .monomorphize(count)
                    .try_to_target_usize(bx.tcx())
                    .expect("expected monomorphic const in codegen");

                bx.write_operand_repeatedly(cg_elem, count, dest);
            }

            // This implementation does field projection, so never use it for `RawPtr`,
            // which will always be fine with the `codegen_rvalue_operand` path below.
            mir::Rvalue::Aggregate(ref kind, ref operands)
                if !matches!(**kind, mir::AggregateKind::RawPtr(..)) =>
            {
                let (variant_index, variant_dest, active_field_index) = match **kind {
                    mir::AggregateKind::Adt(_, variant_index, _, _, active_field_index) => {
                        let variant_dest = dest.project_downcast(bx, variant_index);
                        (variant_index, variant_dest, active_field_index)
                    }
                    _ => (FIRST_VARIANT, dest, None),
                };
                if active_field_index.is_some() {
                    assert_eq!(operands.len(), 1);
                }
                for (i, operand) in operands.iter_enumerated() {
                    let op = self.codegen_operand(bx, operand);
                    // Do not generate stores and GEPis for zero-sized fields.
                    if !op.layout.is_zst() {
                        let field_index = active_field_index.unwrap_or(i);
                        let field = if let mir::AggregateKind::Array(_) = **kind {
                            let llindex = bx.cx().const_usize(field_index.as_u32().into());
                            variant_dest.project_index(bx, llindex)
                        } else {
                            variant_dest.project_field(bx, field_index.as_usize())
                        };
                        op.val.store(bx, field);
                    }
                }
                dest.codegen_set_discr(bx, variant_index);
            }

            _ => {
                assert!(self.rvalue_creates_operand(rvalue, DUMMY_SP));
                let temp = self.codegen_rvalue_operand(bx, rvalue);
                temp.val.store(bx, dest);
            }
        }
    }

    fn codegen_transmute(
        &mut self,
        bx: &mut Bx,
        src: OperandRef<'tcx, Bx::Value>,
        dst: PlaceRef<'tcx, Bx::Value>,
    ) {
        // The MIR validator enforces no unsized transmutes.
        assert!(src.layout.is_sized());
        assert!(dst.layout.is_sized());

        if src.layout.size == dst.layout.size {
            // Since in this path we have a place anyway, we can store or copy to it,
            // making sure we use the destination place's alignment even if the
            // source would normally have a higher one.
            src.val.store(bx, dst.val.with_type(src.layout));
        } else if src.layout.is_uninhabited() {
            bx.unreachable()
        } else {
            // Since this is known statically and the input could have existed
            // without already having hit UB, might as well trap for it, even
            // though it's UB so we *could* also unreachable it.
            bx.abort();
        }
    }

    /// Attempts to transmute an `OperandValue` to another `OperandValue`.
    ///
    /// Returns `None` for cases that can't work in that framework, such as for
    /// `Immediate`->`Ref` that needs an `alloca` to get the location.
    pub(crate) fn codegen_transmute_operand(
        &mut self,
        bx: &mut Bx,
        operand: OperandRef<'tcx, Bx::Value>,
        cast: TyAndLayout<'tcx>,
    ) -> Option<OperandValue<Bx::Value>> {
        // Check for transmutes that are always UB.
        if operand.layout.size != cast.size
            || operand.layout.is_uninhabited()
            || cast.is_uninhabited()
        {
            if !operand.layout.is_uninhabited() {
                // Since this is known statically and the input could have existed
                // without already having hit UB, might as well trap for it.
                bx.abort();
            }

            // Because this transmute is UB, return something easy to generate,
            // since it's fine that later uses of the value are probably UB.
            return Some(OperandValue::poison(bx, cast));
        }

        Some(match (operand.val, operand.layout.backend_repr, cast.backend_repr) {
            _ if cast.is_zst() => OperandValue::ZeroSized,
            (OperandValue::ZeroSized, _, _) => bug!(),
            (
                OperandValue::Ref(source_place_val),
                abi::BackendRepr::Memory { .. },
                abi::BackendRepr::Scalar(_) | abi::BackendRepr::ScalarPair(_, _),
            ) => {
                assert_eq!(source_place_val.llextra, None);
                // The existing alignment is part of `source_place_val`,
                // so that alignment will be used, not `cast`'s.
                bx.load_operand(source_place_val.with_type(cast)).val
            }
            (
                OperandValue::Immediate(imm),
                abi::BackendRepr::Scalar(from_scalar),
                abi::BackendRepr::Scalar(to_scalar),
            ) => OperandValue::Immediate(transmute_immediate(bx, imm, from_scalar, to_scalar)),
            (
                OperandValue::Pair(imm_a, imm_b),
                abi::BackendRepr::ScalarPair(in_a, in_b),
                abi::BackendRepr::ScalarPair(out_a, out_b),
            ) => OperandValue::Pair(
                transmute_immediate(bx, imm_a, in_a, out_a),
                transmute_immediate(bx, imm_b, in_b, out_b),
            ),
            _ => return None,
        })
    }

    /// Cast one of the immediates from an [`OperandValue::Immediate`]
    /// or an [`OperandValue::Pair`] to an immediate of the target type.
    ///
    /// Returns `None` if the cast is not possible.
    fn cast_immediate(
        &self,
        bx: &mut Bx,
        mut imm: Bx::Value,
        from_scalar: abi::Scalar,
        from_backend_ty: Bx::Type,
        to_scalar: abi::Scalar,
        to_backend_ty: Bx::Type,
    ) -> Option<Bx::Value> {
        use abi::Primitive::*;

        // When scalars are passed by value, there's no metadata recording their
        // valid ranges. For example, `char`s are passed as just `i32`, with no
        // way for LLVM to know that they're 0x10FFFF at most. Thus we assume
        // the range of the input value too, not just the output range.
        assume_scalar_range(bx, imm, from_scalar, from_backend_ty);

        imm = match (from_scalar.primitive(), to_scalar.primitive()) {
            (Int(_, is_signed), Int(..)) => bx.intcast(imm, to_backend_ty, is_signed),
            (Float(_), Float(_)) => {
                let srcsz = bx.cx().float_width(from_backend_ty);
                let dstsz = bx.cx().float_width(to_backend_ty);
                if dstsz > srcsz {
                    bx.fpext(imm, to_backend_ty)
                } else if srcsz > dstsz {
                    bx.fptrunc(imm, to_backend_ty)
                } else {
                    imm
                }
            }
            (Int(_, is_signed), Float(_)) => {
                if is_signed {
                    bx.sitofp(imm, to_backend_ty)
                } else {
                    bx.uitofp(imm, to_backend_ty)
                }
            }
            (Pointer(..), Pointer(..)) => bx.pointercast(imm, to_backend_ty),
            (Int(_, is_signed), Pointer(..)) => {
                let usize_imm = bx.intcast(imm, bx.cx().type_isize(), is_signed);
                bx.inttoptr(usize_imm, to_backend_ty)
            }
            (Float(_), Int(_, is_signed)) => bx.cast_float_to_int(is_signed, imm, to_backend_ty),
            _ => return None,
        };
        Some(imm)
    }

    pub(crate) fn codegen_rvalue_unsized(
        &mut self,
        bx: &mut Bx,
        indirect_dest: PlaceRef<'tcx, Bx::Value>,
        rvalue: &mir::Rvalue<'tcx>,
    ) {
        debug!(
            "codegen_rvalue_unsized(indirect_dest.llval={:?}, rvalue={:?})",
            indirect_dest.val.llval, rvalue
        );

        match *rvalue {
            mir::Rvalue::Use(ref operand) => {
                let cg_operand = self.codegen_operand(bx, operand);
                cg_operand.val.store_unsized(bx, indirect_dest);
            }

            _ => bug!("unsized assignment other than `Rvalue::Use`"),
        }
    }

    pub(crate) fn codegen_rvalue_operand(
        &mut self,
        bx: &mut Bx,
        rvalue: &mir::Rvalue<'tcx>,
    ) -> OperandRef<'tcx, Bx::Value> {
        assert!(
            self.rvalue_creates_operand(rvalue, DUMMY_SP),
            "cannot codegen {rvalue:?} to operand",
        );

        match *rvalue {
            mir::Rvalue::Cast(ref kind, ref source, mir_cast_ty) => {
                let operand = self.codegen_operand(bx, source);
                debug!("cast operand is {:?}", operand);
                let cast = bx.cx().layout_of(self.monomorphize(mir_cast_ty));

                let val = match *kind {
                    mir::CastKind::PointerExposeProvenance => {
                        assert!(bx.cx().is_backend_immediate(cast));
                        let llptr = operand.immediate();
                        let llcast_ty = bx.cx().immediate_backend_type(cast);
                        let lladdr = bx.ptrtoint(llptr, llcast_ty);
                        OperandValue::Immediate(lladdr)
                    }
                    mir::CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer, _) => {
                        match *operand.layout.ty.kind() {
                            ty::FnDef(def_id, args) => {
                                let instance = ty::Instance::resolve_for_fn_ptr(
                                    bx.tcx(),
                                    bx.typing_env(),
                                    def_id,
                                    args,
                                )
                                .unwrap();
                                OperandValue::Immediate(bx.get_fn_addr(instance))
                            }
                            _ => bug!("{} cannot be reified to a fn ptr", operand.layout.ty),
                        }
                    }
                    mir::CastKind::PointerCoercion(PointerCoercion::ClosureFnPointer(_), _) => {
                        match *operand.layout.ty.kind() {
                            ty::Closure(def_id, args) => {
                                let instance = Instance::resolve_closure(
                                    bx.cx().tcx(),
                                    def_id,
                                    args,
                                    ty::ClosureKind::FnOnce,
                                );
                                OperandValue::Immediate(bx.cx().get_fn_addr(instance))
                            }
                            _ => bug!("{} cannot be cast to a fn ptr", operand.layout.ty),
                        }
                    }
                    mir::CastKind::PointerCoercion(PointerCoercion::UnsafeFnPointer, _) => {
                        // This is a no-op at the LLVM level.
                        operand.val
                    }
                    mir::CastKind::PointerCoercion(PointerCoercion::Unsize, _) => {
                        assert!(bx.cx().is_backend_scalar_pair(cast));
                        let (lldata, llextra) = operand.val.pointer_parts();
                        let (lldata, llextra) =
                            base::unsize_ptr(bx, lldata, operand.layout.ty, cast.ty, llextra);
                        OperandValue::Pair(lldata, llextra)
                    }
                    mir::CastKind::PointerCoercion(
                        PointerCoercion::MutToConstPointer | PointerCoercion::ArrayToPointer, _
                    ) => {
                        bug!("{kind:?} is for borrowck, and should never appear in codegen");
                    }
                    mir::CastKind::PtrToPtr
                        if bx.cx().is_backend_scalar_pair(operand.layout) =>
                    {
                        if let OperandValue::Pair(data_ptr, meta) = operand.val {
                            if bx.cx().is_backend_scalar_pair(cast) {
                                OperandValue::Pair(data_ptr, meta)
                            } else {
                                // Cast of wide-ptr to thin-ptr is an extraction of data-ptr.
                                OperandValue::Immediate(data_ptr)
                            }
                        } else {
                            bug!("unexpected non-pair operand");
                        }
                    }
                    | mir::CastKind::IntToInt
                    | mir::CastKind::FloatToInt
                    | mir::CastKind::FloatToFloat
                    | mir::CastKind::IntToFloat
                    | mir::CastKind::PtrToPtr
                    | mir::CastKind::FnPtrToPtr
                    // Since int2ptr can have arbitrary integer types as input (so we have to do
                    // sign extension and all that), it is currently best handled in the same code
                    // path as the other integer-to-X casts.
                    | mir::CastKind::PointerWithExposedProvenance => {
                        let imm = operand.immediate();
                        let abi::BackendRepr::Scalar(from_scalar) = operand.layout.backend_repr else {
                            bug!("Found non-scalar for operand {operand:?}");
                        };
                        let from_backend_ty = bx.cx().immediate_backend_type(operand.layout);

                        assert!(bx.cx().is_backend_immediate(cast));
                        let to_backend_ty = bx.cx().immediate_backend_type(cast);
                        if operand.layout.is_uninhabited() {
                            let val = OperandValue::Immediate(bx.cx().const_poison(to_backend_ty));
                            return OperandRef { val, layout: cast };
                        }
                        let abi::BackendRepr::Scalar(to_scalar) = cast.layout.backend_repr else {
                            bug!("Found non-scalar for cast {cast:?}");
                        };

                        self.cast_immediate(bx, imm, from_scalar, from_backend_ty, to_scalar, to_backend_ty)
                            .map(OperandValue::Immediate)
                            .unwrap_or_else(|| {
                                bug!("Unsupported cast of {operand:?} to {cast:?}");
                            })
                    }
                    mir::CastKind::Transmute => {
                        self.codegen_transmute_operand(bx, operand, cast).unwrap_or_else(|| {
                            bug!("Unsupported transmute-as-operand of {operand:?} to {cast:?}");
                        })
                    }
                };
                OperandRef { val, layout: cast }
            }

            mir::Rvalue::Ref(_, bk, place) => {
                let mk_ref = move |tcx: TyCtxt<'tcx>, ty: Ty<'tcx>| {
                    Ty::new_ref(tcx, tcx.lifetimes.re_erased, ty, bk.to_mutbl_lossy())
                };
                self.codegen_place_to_pointer(bx, place, mk_ref)
            }

            mir::Rvalue::CopyForDeref(place) => {
                self.codegen_operand(bx, &mir::Operand::Copy(place))
            }
            mir::Rvalue::RawPtr(kind, place) => {
                let mk_ptr = move |tcx: TyCtxt<'tcx>, ty: Ty<'tcx>| {
                    Ty::new_ptr(tcx, ty, kind.to_mutbl_lossy())
                };
                self.codegen_place_to_pointer(bx, place, mk_ptr)
            }

            mir::Rvalue::Len(place) => {
                let size = self.evaluate_array_len(bx, place);
                OperandRef {
                    val: OperandValue::Immediate(size),
                    layout: bx.cx().layout_of(bx.tcx().types.usize),
                }
            }

            mir::Rvalue::BinaryOp(op_with_overflow, box (ref lhs, ref rhs))
                if let Some(op) = op_with_overflow.overflowing_to_wrapping() =>
            {
                let lhs = self.codegen_operand(bx, lhs);
                let rhs = self.codegen_operand(bx, rhs);
                let result = self.codegen_scalar_checked_binop(
                    bx,
                    op,
                    lhs.immediate(),
                    rhs.immediate(),
                    lhs.layout.ty,
                );
                let val_ty = op.ty(bx.tcx(), lhs.layout.ty, rhs.layout.ty);
                let operand_ty = Ty::new_tup(bx.tcx(), &[val_ty, bx.tcx().types.bool]);
                OperandRef { val: result, layout: bx.cx().layout_of(operand_ty) }
            }
            mir::Rvalue::BinaryOp(op, box (ref lhs, ref rhs)) => {
                let lhs = self.codegen_operand(bx, lhs);
                let rhs = self.codegen_operand(bx, rhs);
                let llresult = match (lhs.val, rhs.val) {
                    (
                        OperandValue::Pair(lhs_addr, lhs_extra),
                        OperandValue::Pair(rhs_addr, rhs_extra),
                    ) => self.codegen_wide_ptr_binop(
                        bx,
                        op,
                        lhs_addr,
                        lhs_extra,
                        rhs_addr,
                        rhs_extra,
                        lhs.layout.ty,
                    ),

                    (OperandValue::Immediate(lhs_val), OperandValue::Immediate(rhs_val)) => self
                        .codegen_scalar_binop(
                            bx,
                            op,
                            lhs_val,
                            rhs_val,
                            lhs.layout.ty,
                            rhs.layout.ty,
                        ),

                    _ => bug!(),
                };
                OperandRef {
                    val: OperandValue::Immediate(llresult),
                    layout: bx.cx().layout_of(op.ty(bx.tcx(), lhs.layout.ty, rhs.layout.ty)),
                }
            }

            mir::Rvalue::UnaryOp(op, ref operand) => {
                let operand = self.codegen_operand(bx, operand);
                let is_float = operand.layout.ty.is_floating_point();
                let (val, layout) = match op {
                    mir::UnOp::Not => {
                        let llval = bx.not(operand.immediate());
                        (OperandValue::Immediate(llval), operand.layout)
                    }
                    mir::UnOp::Neg => {
                        let llval = if is_float {
                            bx.fneg(operand.immediate())
                        } else {
                            bx.neg(operand.immediate())
                        };
                        (OperandValue::Immediate(llval), operand.layout)
                    }
                    mir::UnOp::PtrMetadata => {
                        assert!(operand.layout.ty.is_raw_ptr() || operand.layout.ty.is_ref(),);
                        let (_, meta) = operand.val.pointer_parts();
                        assert_eq!(operand.layout.fields.count() > 1, meta.is_some());
                        if let Some(meta) = meta {
                            (OperandValue::Immediate(meta), operand.layout.field(self.cx, 1))
                        } else {
                            (OperandValue::ZeroSized, bx.cx().layout_of(bx.tcx().types.unit))
                        }
                    }
                };
                assert!(
                    val.is_expected_variant_for_type(self.cx, layout),
                    "Made wrong variant {val:?} for type {layout:?}",
                );
                OperandRef { val, layout }
            }

            mir::Rvalue::Discriminant(ref place) => {
                let discr_ty = rvalue.ty(self.mir, bx.tcx());
                let discr_ty = self.monomorphize(discr_ty);
                let operand = self.codegen_consume(bx, place.as_ref());
                let discr = operand.codegen_get_discr(self, bx, discr_ty);
                OperandRef {
                    val: OperandValue::Immediate(discr),
                    layout: self.cx.layout_of(discr_ty),
                }
            }

            mir::Rvalue::NullaryOp(ref null_op, ty) => {
                let ty = self.monomorphize(ty);
                let layout = bx.cx().layout_of(ty);
                let val = match null_op {
                    mir::NullOp::SizeOf => {
                        assert!(bx.cx().type_is_sized(ty));
                        let val = layout.size.bytes();
                        bx.cx().const_usize(val)
                    }
                    mir::NullOp::AlignOf => {
                        assert!(bx.cx().type_is_sized(ty));
                        let val = layout.align.abi.bytes();
                        bx.cx().const_usize(val)
                    }
                    mir::NullOp::OffsetOf(fields) => {
                        let val = bx
                            .tcx()
                            .offset_of_subfield(bx.typing_env(), layout, fields.iter())
                            .bytes();
                        bx.cx().const_usize(val)
                    }
                    mir::NullOp::UbChecks => {
                        let val = bx.tcx().sess.ub_checks();
                        bx.cx().const_bool(val)
                    }
                    mir::NullOp::ContractChecks => {
                        let val = bx.tcx().sess.contract_checks();
                        bx.cx().const_bool(val)
                    }
                };
                let tcx = self.cx.tcx();
                OperandRef {
                    val: OperandValue::Immediate(val),
                    layout: self.cx.layout_of(null_op.ty(tcx)),
                }
            }

            mir::Rvalue::ThreadLocalRef(def_id) => {
                assert!(bx.cx().tcx().is_static(def_id));
                let layout = bx.layout_of(bx.cx().tcx().static_ptr_ty(def_id, bx.typing_env()));
                let static_ = if !def_id.is_local() && bx.cx().tcx().needs_thread_local_shim(def_id)
                {
                    let instance = ty::Instance {
                        def: ty::InstanceKind::ThreadLocalShim(def_id),
                        args: ty::GenericArgs::empty(),
                    };
                    let fn_ptr = bx.get_fn_addr(instance);
                    let fn_abi = bx.fn_abi_of_instance(instance, ty::List::empty());
                    let fn_ty = bx.fn_decl_backend_type(fn_abi);
                    let fn_attrs = if bx.tcx().def_kind(instance.def_id()).has_codegen_attrs() {
                        Some(bx.tcx().codegen_fn_attrs(instance.def_id()))
                    } else {
                        None
                    };
                    bx.call(fn_ty, fn_attrs, Some(fn_abi), fn_ptr, &[], None, Some(instance))
                } else {
                    bx.get_static(def_id)
                };
                OperandRef { val: OperandValue::Immediate(static_), layout }
            }
            mir::Rvalue::Use(ref operand) => self.codegen_operand(bx, operand),
            mir::Rvalue::Repeat(..) => bug!("{rvalue:?} in codegen_rvalue_operand"),
            mir::Rvalue::Aggregate(_, ref fields) => {
                let ty = rvalue.ty(self.mir, self.cx.tcx());
                let ty = self.monomorphize(ty);
                let layout = self.cx.layout_of(ty);

                // `rvalue_creates_operand` has arranged that we only get here if
                // we can build the aggregate immediate from the field immediates.
                let Some(mut builder) = OperandRef::builder(layout) else {
                    bug!("Cannot use type in operand builder: {layout:?}")
                };
                for (field_idx, field) in fields.iter_enumerated() {
                    let op = self.codegen_operand(bx, field);
                    builder.insert_field(bx, FIRST_VARIANT, field_idx, op);
                }

                builder.build()
            }
            mir::Rvalue::ShallowInitBox(ref operand, content_ty) => {
                let operand = self.codegen_operand(bx, operand);
                let val = operand.immediate();

                let content_ty = self.monomorphize(content_ty);
                let box_layout = bx.cx().layout_of(Ty::new_box(bx.tcx(), content_ty));

                OperandRef { val: OperandValue::Immediate(val), layout: box_layout }
            }
            mir::Rvalue::WrapUnsafeBinder(ref operand, binder_ty) => {
                let operand = self.codegen_operand(bx, operand);
                let binder_ty = self.monomorphize(binder_ty);
                let layout = bx.cx().layout_of(binder_ty);
                OperandRef { val: operand.val, layout }
            }
        }
    }

    fn evaluate_array_len(&mut self, bx: &mut Bx, place: mir::Place<'tcx>) -> Bx::Value {
        // ZST are passed as operands and require special handling
        // because codegen_place() panics if Local is operand.
        if let Some(index) = place.as_local()
            && let LocalRef::Operand(op) = self.locals[index]
            && let ty::Array(_, n) = op.layout.ty.kind()
        {
            let n = n.try_to_target_usize(bx.tcx()).expect("expected monomorphic const in codegen");
            return bx.cx().const_usize(n);
        }
        // use common size calculation for non zero-sized types
        let cg_value = self.codegen_place(bx, place.as_ref());
        cg_value.len(bx.cx())
    }

    /// Codegen an `Rvalue::RawPtr` or `Rvalue::Ref`
    fn codegen_place_to_pointer(
        &mut self,
        bx: &mut Bx,
        place: mir::Place<'tcx>,
        mk_ptr_ty: impl FnOnce(TyCtxt<'tcx>, Ty<'tcx>) -> Ty<'tcx>,
    ) -> OperandRef<'tcx, Bx::Value> {
        let cg_place = self.codegen_place(bx, place.as_ref());
        let val = cg_place.val.address();

        let ty = cg_place.layout.ty;
        assert!(
            if bx.cx().tcx().type_has_metadata(ty, bx.cx().typing_env()) {
                matches!(val, OperandValue::Pair(..))
            } else {
                matches!(val, OperandValue::Immediate(..))
            },
            "Address of place was unexpectedly {val:?} for pointee type {ty:?}",
        );

        OperandRef { val, layout: self.cx.layout_of(mk_ptr_ty(self.cx.tcx(), ty)) }
    }

    fn codegen_scalar_binop(
        &mut self,
        bx: &mut Bx,
        op: mir::BinOp,
        lhs: Bx::Value,
        rhs: Bx::Value,
        lhs_ty: Ty<'tcx>,
        rhs_ty: Ty<'tcx>,
    ) -> Bx::Value {
        let is_float = lhs_ty.is_floating_point();
        let is_signed = lhs_ty.is_signed();
        match op {
            mir::BinOp::Add => {
                if is_float {
                    bx.fadd(lhs, rhs)
                } else {
                    bx.add(lhs, rhs)
                }
            }
            mir::BinOp::AddUnchecked => {
                if is_signed {
                    bx.unchecked_sadd(lhs, rhs)
                } else {
                    bx.unchecked_uadd(lhs, rhs)
                }
            }
            mir::BinOp::Sub => {
                if is_float {
                    bx.fsub(lhs, rhs)
                } else {
                    bx.sub(lhs, rhs)
                }
            }
            mir::BinOp::SubUnchecked => {
                if is_signed {
                    bx.unchecked_ssub(lhs, rhs)
                } else {
                    bx.unchecked_usub(lhs, rhs)
                }
            }
            mir::BinOp::Mul => {
                if is_float {
                    bx.fmul(lhs, rhs)
                } else {
                    bx.mul(lhs, rhs)
                }
            }
            mir::BinOp::MulUnchecked => {
                if is_signed {
                    bx.unchecked_smul(lhs, rhs)
                } else {
                    bx.unchecked_umul(lhs, rhs)
                }
            }
            mir::BinOp::Div => {
                if is_float {
                    bx.fdiv(lhs, rhs)
                } else if is_signed {
                    bx.sdiv(lhs, rhs)
                } else {
                    bx.udiv(lhs, rhs)
                }
            }
            mir::BinOp::Rem => {
                if is_float {
                    bx.frem(lhs, rhs)
                } else if is_signed {
                    bx.srem(lhs, rhs)
                } else {
                    bx.urem(lhs, rhs)
                }
            }
            mir::BinOp::BitOr => bx.or(lhs, rhs),
            mir::BinOp::BitAnd => bx.and(lhs, rhs),
            mir::BinOp::BitXor => bx.xor(lhs, rhs),
            mir::BinOp::Offset => {
                let pointee_type = lhs_ty
                    .builtin_deref(true)
                    .unwrap_or_else(|| bug!("deref of non-pointer {:?}", lhs_ty));
                let pointee_layout = bx.cx().layout_of(pointee_type);
                if pointee_layout.is_zst() {
                    // `Offset` works in terms of the size of pointee,
                    // so offsetting a pointer to ZST is a noop.
                    lhs
                } else {
                    let llty = bx.cx().backend_type(pointee_layout);
                    if !rhs_ty.is_signed() {
                        bx.inbounds_nuw_gep(llty, lhs, &[rhs])
                    } else {
                        bx.inbounds_gep(llty, lhs, &[rhs])
                    }
                }
            }
            mir::BinOp::Shl | mir::BinOp::ShlUnchecked => {
                let rhs = base::build_shift_expr_rhs(bx, lhs, rhs, op == mir::BinOp::ShlUnchecked);
                bx.shl(lhs, rhs)
            }
            mir::BinOp::Shr | mir::BinOp::ShrUnchecked => {
                let rhs = base::build_shift_expr_rhs(bx, lhs, rhs, op == mir::BinOp::ShrUnchecked);
                if is_signed { bx.ashr(lhs, rhs) } else { bx.lshr(lhs, rhs) }
            }
            mir::BinOp::Ne
            | mir::BinOp::Lt
            | mir::BinOp::Gt
            | mir::BinOp::Eq
            | mir::BinOp::Le
            | mir::BinOp::Ge => {
                if is_float {
                    bx.fcmp(base::bin_op_to_fcmp_predicate(op), lhs, rhs)
                } else {
                    bx.icmp(base::bin_op_to_icmp_predicate(op, is_signed), lhs, rhs)
                }
            }
            mir::BinOp::Cmp => {
                use std::cmp::Ordering;
                assert!(!is_float);
                if let Some(value) = bx.three_way_compare(lhs_ty, lhs, rhs) {
                    return value;
                }
                let pred = |op| base::bin_op_to_icmp_predicate(op, is_signed);
                if bx.cx().tcx().sess.opts.optimize == OptLevel::No {
                    // FIXME: This actually generates tighter assembly, and is a classic trick
                    // <https://graphics.stanford.edu/~seander/bithacks.html#CopyIntegerSign>
                    // However, as of 2023-11 it optimizes worse in things like derived
                    // `PartialOrd`, so only use it in debug for now. Once LLVM can handle it
                    // better (see <https://github.com/llvm/llvm-project/issues/73417>), it'll
                    // be worth trying it in optimized builds as well.
                    let is_gt = bx.icmp(pred(mir::BinOp::Gt), lhs, rhs);
                    let gtext = bx.zext(is_gt, bx.type_i8());
                    let is_lt = bx.icmp(pred(mir::BinOp::Lt), lhs, rhs);
                    let ltext = bx.zext(is_lt, bx.type_i8());
                    bx.unchecked_ssub(gtext, ltext)
                } else {
                    // These operations are those expected by `tests/codegen/integer-cmp.rs`,
                    // from <https://github.com/rust-lang/rust/pull/63767>.
                    let is_lt = bx.icmp(pred(mir::BinOp::Lt), lhs, rhs);
                    let is_ne = bx.icmp(pred(mir::BinOp::Ne), lhs, rhs);
                    let ge = bx.select(
                        is_ne,
                        bx.cx().const_i8(Ordering::Greater as i8),
                        bx.cx().const_i8(Ordering::Equal as i8),
                    );
                    bx.select(is_lt, bx.cx().const_i8(Ordering::Less as i8), ge)
                }
            }
            mir::BinOp::AddWithOverflow
            | mir::BinOp::SubWithOverflow
            | mir::BinOp::MulWithOverflow => {
                bug!("{op:?} needs to return a pair, so call codegen_scalar_checked_binop instead")
            }
        }
    }

    fn codegen_wide_ptr_binop(
        &mut self,
        bx: &mut Bx,
        op: mir::BinOp,
        lhs_addr: Bx::Value,
        lhs_extra: Bx::Value,
        rhs_addr: Bx::Value,
        rhs_extra: Bx::Value,
        _input_ty: Ty<'tcx>,
    ) -> Bx::Value {
        match op {
            mir::BinOp::Eq => {
                let lhs = bx.icmp(IntPredicate::IntEQ, lhs_addr, rhs_addr);
                let rhs = bx.icmp(IntPredicate::IntEQ, lhs_extra, rhs_extra);
                bx.and(lhs, rhs)
            }
            mir::BinOp::Ne => {
                let lhs = bx.icmp(IntPredicate::IntNE, lhs_addr, rhs_addr);
                let rhs = bx.icmp(IntPredicate::IntNE, lhs_extra, rhs_extra);
                bx.or(lhs, rhs)
            }
            mir::BinOp::Le | mir::BinOp::Lt | mir::BinOp::Ge | mir::BinOp::Gt => {
                // a OP b ~ a.0 STRICT(OP) b.0 | (a.0 == b.0 && a.1 OP a.1)
                let (op, strict_op) = match op {
                    mir::BinOp::Lt => (IntPredicate::IntULT, IntPredicate::IntULT),
                    mir::BinOp::Le => (IntPredicate::IntULE, IntPredicate::IntULT),
                    mir::BinOp::Gt => (IntPredicate::IntUGT, IntPredicate::IntUGT),
                    mir::BinOp::Ge => (IntPredicate::IntUGE, IntPredicate::IntUGT),
                    _ => bug!(),
                };
                let lhs = bx.icmp(strict_op, lhs_addr, rhs_addr);
                let and_lhs = bx.icmp(IntPredicate::IntEQ, lhs_addr, rhs_addr);
                let and_rhs = bx.icmp(op, lhs_extra, rhs_extra);
                let rhs = bx.and(and_lhs, and_rhs);
                bx.or(lhs, rhs)
            }
            _ => {
                bug!("unexpected wide ptr binop");
            }
        }
    }

    fn codegen_scalar_checked_binop(
        &mut self,
        bx: &mut Bx,
        op: mir::BinOp,
        lhs: Bx::Value,
        rhs: Bx::Value,
        input_ty: Ty<'tcx>,
    ) -> OperandValue<Bx::Value> {
        let (val, of) = match op {
            // These are checked using intrinsics
            mir::BinOp::Add | mir::BinOp::Sub | mir::BinOp::Mul => {
                let oop = match op {
                    mir::BinOp::Add => OverflowOp::Add,
                    mir::BinOp::Sub => OverflowOp::Sub,
                    mir::BinOp::Mul => OverflowOp::Mul,
                    _ => unreachable!(),
                };
                bx.checked_binop(oop, input_ty, lhs, rhs)
            }
            _ => bug!("Operator `{:?}` is not a checkable operator", op),
        };

        OperandValue::Pair(val, of)
    }

    pub(crate) fn rvalue_creates_operand(&self, rvalue: &mir::Rvalue<'tcx>, span: Span) -> bool {
        match *rvalue {
            mir::Rvalue::Cast(mir::CastKind::Transmute, ref operand, cast_ty) => {
                let operand_ty = operand.ty(self.mir, self.cx.tcx());
                let cast_layout = self.cx.layout_of(self.monomorphize(cast_ty));
                let operand_layout = self.cx.layout_of(self.monomorphize(operand_ty));
                match (operand_layout.backend_repr, cast_layout.backend_repr) {
                    // If the input is in a place we can load immediates from there.
                    (abi::BackendRepr::Memory { .. }, abi::BackendRepr::Scalar(_) | abi::BackendRepr::ScalarPair(_, _)) => true,

                    // When we have scalar immediates, we can only convert things
                    // where the sizes match, to avoid endianness questions.
                    (abi::BackendRepr::Scalar(a), abi::BackendRepr::Scalar(b)) =>
                        a.size(self.cx) == b.size(self.cx),
                    (abi::BackendRepr::ScalarPair(a0, a1), abi::BackendRepr::ScalarPair(b0, b1)) =>
                        a0.size(self.cx) == b0.size(self.cx) && a1.size(self.cx) == b1.size(self.cx),

                    // SIMD vectors don't work like normal immediates,
                    // so always send them through memory.
                    (abi::BackendRepr::SimdVector { .. }, _) | (_, abi::BackendRepr::SimdVector { .. }) => false,

                    // When the output will be in memory anyway, just use its place
                    // (instead of the operand path) unless it's the trivial ZST case.
                    (_, abi::BackendRepr::Memory { .. }) => cast_layout.is_zst(),

                    // Mixing Scalars and ScalarPairs can get quite complicated when
                    // padding and undef get involved, so leave that to the memory path.
                    (abi::BackendRepr::Scalar(_), abi::BackendRepr::ScalarPair(_, _)) |
                    (abi::BackendRepr::ScalarPair(_, _), abi::BackendRepr::Scalar(_)) => false,
                }
            }
            mir::Rvalue::Ref(..) |
            mir::Rvalue::CopyForDeref(..) |
            mir::Rvalue::RawPtr(..) |
            mir::Rvalue::Len(..) |
            mir::Rvalue::Cast(..) | // (*)
            mir::Rvalue::ShallowInitBox(..) | // (*)
            mir::Rvalue::BinaryOp(..) |
            mir::Rvalue::UnaryOp(..) |
            mir::Rvalue::Discriminant(..) |
            mir::Rvalue::NullaryOp(..) |
            mir::Rvalue::ThreadLocalRef(_) |
            mir::Rvalue::Use(..) |
            mir::Rvalue::WrapUnsafeBinder(..) => // (*)
                true,
            // Arrays are always aggregates, so it's not worth checking anything here.
            // (If it's really `[(); N]` or `[T; 0]` and we use the place path, fine.)
            mir::Rvalue::Repeat(..) => false,
            mir::Rvalue::Aggregate(ref kind, _) => {
                let allowed_kind = match **kind {
                    // This always produces a `ty::RawPtr`, so will be Immediate or Pair
                    mir::AggregateKind::RawPtr(..) => true,
                    mir::AggregateKind::Array(..) => false,
                    mir::AggregateKind::Tuple => true,
                    mir::AggregateKind::Adt(def_id, ..) => {
                        let adt_def = self.cx.tcx().adt_def(def_id);
                        adt_def.is_struct() && !adt_def.repr().simd()
                    }
                    mir::AggregateKind::Closure(..) => true,
                    // FIXME: Can we do this for simple coroutines too?
                    mir::AggregateKind::Coroutine(..) | mir::AggregateKind::CoroutineClosure(..) => false,
                };
                allowed_kind && {
                    let ty = rvalue.ty(self.mir, self.cx.tcx());
                    let ty = self.monomorphize(ty);
                    let layout = self.cx.spanned_layout_of(ty, span);
                    OperandRef::<Bx::Value>::builder(layout).is_some()
                }
            }
        }

        // (*) this is only true if the type is suitable
    }
}

/// Transmutes one of the immediates from an [`OperandValue::Immediate`]
/// or an [`OperandValue::Pair`] to an immediate of the target type.
///
/// `to_backend_ty` must be the *non*-immediate backend type (so it will be
/// `i8`, not `i1`, for `bool`-like types.)
pub(super) fn transmute_immediate<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    mut imm: Bx::Value,
    from_scalar: abi::Scalar,
    to_scalar: abi::Scalar,
) -> Bx::Value {
    assert_eq!(from_scalar.size(bx.cx()), to_scalar.size(bx.cx()));
    let imm_ty = bx.cx().val_ty(imm);
    assert_ne!(
        bx.cx().type_kind(imm_ty),
        TypeKind::Vector,
        "Vector type {imm_ty:?} not allowed in transmute_immediate {from_scalar:?} -> {to_scalar:?}"
    );

    // While optimizations will remove no-op transmutes, they might still be
    // there in debug or things that aren't no-op in MIR because they change
    // the Rust type but not the underlying layout/niche.
    if from_scalar == to_scalar {
        return imm;
    }

    use abi::Primitive::*;
    imm = bx.from_immediate(imm);

    let from_backend_ty = bx.cx().type_from_scalar(from_scalar);
    debug_assert_eq!(bx.cx().val_ty(imm), from_backend_ty);
    let to_backend_ty = bx.cx().type_from_scalar(to_scalar);

    // If we have a scalar, we must already know its range. Either
    //
    // 1) It's a parameter with `range` parameter metadata,
    // 2) It's something we `load`ed with `!range` metadata, or
    // 3) After a transmute we `assume`d the range (see below).
    //
    // That said, last time we tried removing this, it didn't actually help
    // the rustc-perf results, so might as well keep doing it
    // <https://github.com/rust-lang/rust/pull/135610#issuecomment-2599275182>
    assume_scalar_range(bx, imm, from_scalar, from_backend_ty);

    imm = match (from_scalar.primitive(), to_scalar.primitive()) {
        (Int(..) | Float(_), Int(..) | Float(_)) => bx.bitcast(imm, to_backend_ty),
        (Pointer(..), Pointer(..)) => bx.pointercast(imm, to_backend_ty),
        (Int(..), Pointer(..)) => bx.ptradd(bx.const_null(bx.type_ptr()), imm),
        (Pointer(..), Int(..)) => {
            // FIXME: this exposes the provenance, which shouldn't be necessary.
            bx.ptrtoint(imm, to_backend_ty)
        }
        (Float(_), Pointer(..)) => {
            let int_imm = bx.bitcast(imm, bx.cx().type_isize());
            bx.ptradd(bx.const_null(bx.type_ptr()), int_imm)
        }
        (Pointer(..), Float(_)) => {
            // FIXME: this exposes the provenance, which shouldn't be necessary.
            let int_imm = bx.ptrtoint(imm, bx.cx().type_isize());
            bx.bitcast(int_imm, to_backend_ty)
        }
    };

    debug_assert_eq!(bx.cx().val_ty(imm), to_backend_ty);

    // This `assume` remains important for cases like (a conceptual)
    //    transmute::<u32, NonZeroU32>(x) == 0
    // since it's never passed to something with parameter metadata (especially
    // after MIR inlining) so the only way to tell the backend about the
    // constraint that the `transmute` introduced is to `assume` it.
    assume_scalar_range(bx, imm, to_scalar, to_backend_ty);

    imm = bx.to_immediate_scalar(imm, to_scalar);
    imm
}

fn assume_scalar_range<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    imm: Bx::Value,
    scalar: abi::Scalar,
    backend_ty: Bx::Type,
) {
    if matches!(bx.cx().sess().opts.optimize, OptLevel::No) || scalar.is_always_valid(bx.cx()) {
        return;
    }

    match scalar.primitive() {
        abi::Primitive::Int(..) => {
            let range = scalar.valid_range(bx.cx());
            bx.assume_integer_range(imm, backend_ty, range);
        }
        abi::Primitive::Pointer(abi::AddressSpace::DATA)
            if !scalar.valid_range(bx.cx()).contains(0) =>
        {
            bx.assume_nonnull(imm);
        }
        abi::Primitive::Pointer(..) | abi::Primitive::Float(..) => {}
    }
}
