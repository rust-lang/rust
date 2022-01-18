use super::operand::{OperandRef, OperandValue};
use super::place::PlaceRef;
use super::{FunctionCx, LocalRef};

use crate::base;
use crate::common::{self, IntPredicate};
use crate::traits::*;
use crate::MemFlags;

use rustc_middle::mir;
use rustc_middle::ty::cast::{CastTy, IntTy};
use rustc_middle::ty::layout::{HasTyCtxt, LayoutOf};
use rustc_middle::ty::{self, adjustment::PointerCast, Instance, Ty, TyCtxt};
use rustc_span::source_map::{Span, DUMMY_SP};
use rustc_target::abi::{Abi, Int, Variants};

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn codegen_rvalue(
        &mut self,
        mut bx: Bx,
        dest: PlaceRef<'tcx, Bx::Value>,
        rvalue: &mir::Rvalue<'tcx>,
    ) -> Bx {
        debug!("codegen_rvalue(dest.llval={:?}, rvalue={:?})", dest.llval, rvalue);

        match *rvalue {
            mir::Rvalue::Use(ref operand) => {
                let cg_operand = self.codegen_operand(&mut bx, operand);
                // FIXME: consider not copying constants through stack. (Fixable by codegen'ing
                // constants into `OperandValue::Ref`; why don’t we do that yet if we don’t?)
                cg_operand.val.store(&mut bx, dest);
                bx
            }

            mir::Rvalue::Cast(mir::CastKind::Pointer(PointerCast::Unsize), ref source, _) => {
                // The destination necessarily contains a fat pointer, so if
                // it's a scalar pair, it's a fat pointer or newtype thereof.
                if bx.cx().is_backend_scalar_pair(dest.layout) {
                    // Into-coerce of a thin pointer to a fat pointer -- just
                    // use the operand path.
                    let (mut bx, temp) = self.codegen_rvalue_operand(bx, rvalue);
                    temp.val.store(&mut bx, dest);
                    return bx;
                }

                // Unsize of a nontrivial struct. I would prefer for
                // this to be eliminated by MIR building, but
                // `CoerceUnsized` can be passed by a where-clause,
                // so the (generic) MIR may not be able to expand it.
                let operand = self.codegen_operand(&mut bx, source);
                match operand.val {
                    OperandValue::Pair(..) | OperandValue::Immediate(_) => {
                        // Unsize from an immediate structure. We don't
                        // really need a temporary alloca here, but
                        // avoiding it would require us to have
                        // `coerce_unsized_into` use `extractvalue` to
                        // index into the struct, and this case isn't
                        // important enough for it.
                        debug!("codegen_rvalue: creating ugly alloca");
                        let scratch = PlaceRef::alloca(&mut bx, operand.layout);
                        scratch.storage_live(&mut bx);
                        operand.val.store(&mut bx, scratch);
                        base::coerce_unsized_into(&mut bx, scratch, dest);
                        scratch.storage_dead(&mut bx);
                    }
                    OperandValue::Ref(llref, None, align) => {
                        let source = PlaceRef::new_sized_aligned(llref, operand.layout, align);
                        base::coerce_unsized_into(&mut bx, source, dest);
                    }
                    OperandValue::Ref(_, Some(_), _) => {
                        bug!("unsized coercion on an unsized rvalue");
                    }
                }
                bx
            }

            mir::Rvalue::Repeat(ref elem, count) => {
                let cg_elem = self.codegen_operand(&mut bx, elem);

                // Do not generate the loop for zero-sized elements or empty arrays.
                if dest.layout.is_zst() {
                    return bx;
                }

                if let OperandValue::Immediate(v) = cg_elem.val {
                    let zero = bx.const_usize(0);
                    let start = dest.project_index(&mut bx, zero).llval;
                    let size = bx.const_usize(dest.layout.size.bytes());

                    // Use llvm.memset.p0i8.* to initialize all zero arrays
                    if bx.cx().const_to_opt_uint(v) == Some(0) {
                        let fill = bx.cx().const_u8(0);
                        bx.memset(start, fill, size, dest.align, MemFlags::empty());
                        return bx;
                    }

                    // Use llvm.memset.p0i8.* to initialize byte arrays
                    let v = bx.from_immediate(v);
                    if bx.cx().val_ty(v) == bx.cx().type_i8() {
                        bx.memset(start, v, size, dest.align, MemFlags::empty());
                        return bx;
                    }
                }

                let count =
                    self.monomorphize(count).eval_usize(bx.cx().tcx(), ty::ParamEnv::reveal_all());

                bx.write_operand_repeatedly(cg_elem, count, dest)
            }

            mir::Rvalue::Aggregate(ref kind, ref operands) => {
                let (dest, active_field_index) = match **kind {
                    mir::AggregateKind::Adt(adt_did, variant_index, _, _, active_field_index) => {
                        dest.codegen_set_discr(&mut bx, variant_index);
                        if bx.tcx().adt_def(adt_did).is_enum() {
                            (dest.project_downcast(&mut bx, variant_index), active_field_index)
                        } else {
                            (dest, active_field_index)
                        }
                    }
                    _ => (dest, None),
                };
                for (i, operand) in operands.iter().enumerate() {
                    let op = self.codegen_operand(&mut bx, operand);
                    // Do not generate stores and GEPis for zero-sized fields.
                    if !op.layout.is_zst() {
                        let field_index = active_field_index.unwrap_or(i);
                        let field = dest.project_field(&mut bx, field_index);
                        op.val.store(&mut bx, field);
                    }
                }
                bx
            }

            _ => {
                assert!(self.rvalue_creates_operand(rvalue, DUMMY_SP));
                let (mut bx, temp) = self.codegen_rvalue_operand(bx, rvalue);
                temp.val.store(&mut bx, dest);
                bx
            }
        }
    }

    pub fn codegen_rvalue_unsized(
        &mut self,
        mut bx: Bx,
        indirect_dest: PlaceRef<'tcx, Bx::Value>,
        rvalue: &mir::Rvalue<'tcx>,
    ) -> Bx {
        debug!(
            "codegen_rvalue_unsized(indirect_dest.llval={:?}, rvalue={:?})",
            indirect_dest.llval, rvalue
        );

        match *rvalue {
            mir::Rvalue::Use(ref operand) => {
                let cg_operand = self.codegen_operand(&mut bx, operand);
                cg_operand.val.store_unsized(&mut bx, indirect_dest);
                bx
            }

            _ => bug!("unsized assignment other than `Rvalue::Use`"),
        }
    }

    pub fn codegen_rvalue_operand(
        &mut self,
        mut bx: Bx,
        rvalue: &mir::Rvalue<'tcx>,
    ) -> (Bx, OperandRef<'tcx, Bx::Value>) {
        assert!(
            self.rvalue_creates_operand(rvalue, DUMMY_SP),
            "cannot codegen {:?} to operand",
            rvalue,
        );

        match *rvalue {
            mir::Rvalue::Cast(ref kind, ref source, mir_cast_ty) => {
                let operand = self.codegen_operand(&mut bx, source);
                debug!("cast operand is {:?}", operand);
                let cast = bx.cx().layout_of(self.monomorphize(mir_cast_ty));

                let val = match *kind {
                    mir::CastKind::Pointer(PointerCast::ReifyFnPointer) => {
                        match *operand.layout.ty.kind() {
                            ty::FnDef(def_id, substs) => {
                                let instance = ty::Instance::resolve_for_fn_ptr(
                                    bx.tcx(),
                                    ty::ParamEnv::reveal_all(),
                                    def_id,
                                    substs,
                                )
                                .unwrap()
                                .polymorphize(bx.cx().tcx());
                                OperandValue::Immediate(bx.get_fn_addr(instance))
                            }
                            _ => bug!("{} cannot be reified to a fn ptr", operand.layout.ty),
                        }
                    }
                    mir::CastKind::Pointer(PointerCast::ClosureFnPointer(_)) => {
                        match *operand.layout.ty.kind() {
                            ty::Closure(def_id, substs) => {
                                let instance = Instance::resolve_closure(
                                    bx.cx().tcx(),
                                    def_id,
                                    substs,
                                    ty::ClosureKind::FnOnce,
                                )
                                .polymorphize(bx.cx().tcx());
                                OperandValue::Immediate(bx.cx().get_fn_addr(instance))
                            }
                            _ => bug!("{} cannot be cast to a fn ptr", operand.layout.ty),
                        }
                    }
                    mir::CastKind::Pointer(PointerCast::UnsafeFnPointer) => {
                        // This is a no-op at the LLVM level.
                        operand.val
                    }
                    mir::CastKind::Pointer(PointerCast::Unsize) => {
                        assert!(bx.cx().is_backend_scalar_pair(cast));
                        let (lldata, llextra) = match operand.val {
                            OperandValue::Pair(lldata, llextra) => {
                                // unsize from a fat pointer -- this is a
                                // "trait-object-to-supertrait" coercion.
                                (lldata, Some(llextra))
                            }
                            OperandValue::Immediate(lldata) => {
                                // "standard" unsize
                                (lldata, None)
                            }
                            OperandValue::Ref(..) => {
                                bug!("by-ref operand {:?} in `codegen_rvalue_operand`", operand);
                            }
                        };
                        let (lldata, llextra) =
                            base::unsize_ptr(&mut bx, lldata, operand.layout.ty, cast.ty, llextra);
                        OperandValue::Pair(lldata, llextra)
                    }
                    mir::CastKind::Pointer(PointerCast::MutToConstPointer)
                    | mir::CastKind::Misc
                        if bx.cx().is_backend_scalar_pair(operand.layout) =>
                    {
                        if let OperandValue::Pair(data_ptr, meta) = operand.val {
                            if bx.cx().is_backend_scalar_pair(cast) {
                                let data_cast = bx.pointercast(
                                    data_ptr,
                                    bx.cx().scalar_pair_element_backend_type(cast, 0, true),
                                );
                                OperandValue::Pair(data_cast, meta)
                            } else {
                                // cast to thin-ptr
                                // Cast of fat-ptr to thin-ptr is an extraction of data-ptr and
                                // pointer-cast of that pointer to desired pointer type.
                                let llcast_ty = bx.cx().immediate_backend_type(cast);
                                let llval = bx.pointercast(data_ptr, llcast_ty);
                                OperandValue::Immediate(llval)
                            }
                        } else {
                            bug!("unexpected non-pair operand");
                        }
                    }
                    mir::CastKind::Pointer(
                        PointerCast::MutToConstPointer | PointerCast::ArrayToPointer,
                    )
                    | mir::CastKind::Misc => {
                        assert!(bx.cx().is_backend_immediate(cast));
                        let ll_t_out = bx.cx().immediate_backend_type(cast);
                        if operand.layout.abi.is_uninhabited() {
                            let val = OperandValue::Immediate(bx.cx().const_undef(ll_t_out));
                            return (bx, OperandRef { val, layout: cast });
                        }
                        let r_t_in =
                            CastTy::from_ty(operand.layout.ty).expect("bad input type for cast");
                        let r_t_out = CastTy::from_ty(cast.ty).expect("bad output type for cast");
                        let ll_t_in = bx.cx().immediate_backend_type(operand.layout);
                        match operand.layout.variants {
                            Variants::Single { index } => {
                                if let Some(discr) =
                                    operand.layout.ty.discriminant_for_variant(bx.tcx(), index)
                                {
                                    let discr_layout = bx.cx().layout_of(discr.ty);
                                    let discr_t = bx.cx().immediate_backend_type(discr_layout);
                                    let discr_val = bx.cx().const_uint_big(discr_t, discr.val);
                                    let discr_val =
                                        bx.intcast(discr_val, ll_t_out, discr.ty.is_signed());

                                    return (
                                        bx,
                                        OperandRef {
                                            val: OperandValue::Immediate(discr_val),
                                            layout: cast,
                                        },
                                    );
                                }
                            }
                            Variants::Multiple { .. } => {}
                        }
                        let llval = operand.immediate();

                        let mut signed = false;
                        if let Abi::Scalar(scalar) = operand.layout.abi {
                            if let Int(_, s) = scalar.value {
                                // We use `i1` for bytes that are always `0` or `1`,
                                // e.g., `#[repr(i8)] enum E { A, B }`, but we can't
                                // let LLVM interpret the `i1` as signed, because
                                // then `i1 1` (i.e., E::B) is effectively `i8 -1`.
                                signed = !scalar.is_bool() && s;

                                if !scalar.is_always_valid(bx.cx())
                                    && scalar.valid_range.end >= scalar.valid_range.start
                                {
                                    // We want `table[e as usize ± k]` to not
                                    // have bound checks, and this is the most
                                    // convenient place to put the `assume`s.
                                    if scalar.valid_range.start > 0 {
                                        let enum_value_lower_bound = bx
                                            .cx()
                                            .const_uint_big(ll_t_in, scalar.valid_range.start);
                                        let cmp_start = bx.icmp(
                                            IntPredicate::IntUGE,
                                            llval,
                                            enum_value_lower_bound,
                                        );
                                        bx.assume(cmp_start);
                                    }

                                    let enum_value_upper_bound =
                                        bx.cx().const_uint_big(ll_t_in, scalar.valid_range.end);
                                    let cmp_end = bx.icmp(
                                        IntPredicate::IntULE,
                                        llval,
                                        enum_value_upper_bound,
                                    );
                                    bx.assume(cmp_end);
                                }
                            }
                        }

                        let newval = match (r_t_in, r_t_out) {
                            (CastTy::Int(_), CastTy::Int(_)) => bx.intcast(llval, ll_t_out, signed),
                            (CastTy::Float, CastTy::Float) => {
                                let srcsz = bx.cx().float_width(ll_t_in);
                                let dstsz = bx.cx().float_width(ll_t_out);
                                if dstsz > srcsz {
                                    bx.fpext(llval, ll_t_out)
                                } else if srcsz > dstsz {
                                    bx.fptrunc(llval, ll_t_out)
                                } else {
                                    llval
                                }
                            }
                            (CastTy::Int(_), CastTy::Float) => {
                                if signed {
                                    bx.sitofp(llval, ll_t_out)
                                } else {
                                    bx.uitofp(llval, ll_t_out)
                                }
                            }
                            (CastTy::Ptr(_) | CastTy::FnPtr, CastTy::Ptr(_)) => {
                                bx.pointercast(llval, ll_t_out)
                            }
                            (CastTy::Ptr(_) | CastTy::FnPtr, CastTy::Int(_)) => {
                                bx.ptrtoint(llval, ll_t_out)
                            }
                            (CastTy::Int(_), CastTy::Ptr(_)) => {
                                let usize_llval = bx.intcast(llval, bx.cx().type_isize(), signed);
                                bx.inttoptr(usize_llval, ll_t_out)
                            }
                            (CastTy::Float, CastTy::Int(IntTy::I)) => {
                                bx.cast_float_to_int(true, llval, ll_t_out)
                            }
                            (CastTy::Float, CastTy::Int(_)) => {
                                bx.cast_float_to_int(false, llval, ll_t_out)
                            }
                            _ => bug!("unsupported cast: {:?} to {:?}", operand.layout.ty, cast.ty),
                        };
                        OperandValue::Immediate(newval)
                    }
                };
                (bx, OperandRef { val, layout: cast })
            }

            mir::Rvalue::Ref(_, bk, place) => {
                let mk_ref = move |tcx: TyCtxt<'tcx>, ty: Ty<'tcx>| {
                    tcx.mk_ref(
                        tcx.lifetimes.re_erased,
                        ty::TypeAndMut { ty, mutbl: bk.to_mutbl_lossy() },
                    )
                };
                self.codegen_place_to_pointer(bx, place, mk_ref)
            }

            mir::Rvalue::AddressOf(mutability, place) => {
                let mk_ptr = move |tcx: TyCtxt<'tcx>, ty: Ty<'tcx>| {
                    tcx.mk_ptr(ty::TypeAndMut { ty, mutbl: mutability })
                };
                self.codegen_place_to_pointer(bx, place, mk_ptr)
            }

            mir::Rvalue::Len(place) => {
                let size = self.evaluate_array_len(&mut bx, place);
                let operand = OperandRef {
                    val: OperandValue::Immediate(size),
                    layout: bx.cx().layout_of(bx.tcx().types.usize),
                };
                (bx, operand)
            }

            mir::Rvalue::BinaryOp(op, box (ref lhs, ref rhs)) => {
                let lhs = self.codegen_operand(&mut bx, lhs);
                let rhs = self.codegen_operand(&mut bx, rhs);
                let llresult = match (lhs.val, rhs.val) {
                    (
                        OperandValue::Pair(lhs_addr, lhs_extra),
                        OperandValue::Pair(rhs_addr, rhs_extra),
                    ) => self.codegen_fat_ptr_binop(
                        &mut bx,
                        op,
                        lhs_addr,
                        lhs_extra,
                        rhs_addr,
                        rhs_extra,
                        lhs.layout.ty,
                    ),

                    (OperandValue::Immediate(lhs_val), OperandValue::Immediate(rhs_val)) => {
                        self.codegen_scalar_binop(&mut bx, op, lhs_val, rhs_val, lhs.layout.ty)
                    }

                    _ => bug!(),
                };
                let operand = OperandRef {
                    val: OperandValue::Immediate(llresult),
                    layout: bx.cx().layout_of(op.ty(bx.tcx(), lhs.layout.ty, rhs.layout.ty)),
                };
                (bx, operand)
            }
            mir::Rvalue::CheckedBinaryOp(op, box (ref lhs, ref rhs)) => {
                let lhs = self.codegen_operand(&mut bx, lhs);
                let rhs = self.codegen_operand(&mut bx, rhs);
                let result = self.codegen_scalar_checked_binop(
                    &mut bx,
                    op,
                    lhs.immediate(),
                    rhs.immediate(),
                    lhs.layout.ty,
                );
                let val_ty = op.ty(bx.tcx(), lhs.layout.ty, rhs.layout.ty);
                let operand_ty = bx.tcx().intern_tup(&[val_ty, bx.tcx().types.bool]);
                let operand = OperandRef { val: result, layout: bx.cx().layout_of(operand_ty) };

                (bx, operand)
            }

            mir::Rvalue::UnaryOp(op, ref operand) => {
                let operand = self.codegen_operand(&mut bx, operand);
                let lloperand = operand.immediate();
                let is_float = operand.layout.ty.is_floating_point();
                let llval = match op {
                    mir::UnOp::Not => bx.not(lloperand),
                    mir::UnOp::Neg => {
                        if is_float {
                            bx.fneg(lloperand)
                        } else {
                            bx.neg(lloperand)
                        }
                    }
                };
                (bx, OperandRef { val: OperandValue::Immediate(llval), layout: operand.layout })
            }

            mir::Rvalue::Discriminant(ref place) => {
                let discr_ty = rvalue.ty(self.mir, bx.tcx());
                let discr_ty = self.monomorphize(discr_ty);
                let discr = self
                    .codegen_place(&mut bx, place.as_ref())
                    .codegen_get_discr(&mut bx, discr_ty);
                (
                    bx,
                    OperandRef {
                        val: OperandValue::Immediate(discr),
                        layout: self.cx.layout_of(discr_ty),
                    },
                )
            }

            mir::Rvalue::NullaryOp(null_op, ty) => {
                let ty = self.monomorphize(ty);
                assert!(bx.cx().type_is_sized(ty));
                let layout = bx.cx().layout_of(ty);
                let val = match null_op {
                    mir::NullOp::SizeOf => layout.size.bytes(),
                    mir::NullOp::AlignOf => layout.align.abi.bytes(),
                };
                let val = bx.cx().const_usize(val);
                let tcx = self.cx.tcx();
                (
                    bx,
                    OperandRef {
                        val: OperandValue::Immediate(val),
                        layout: self.cx.layout_of(tcx.types.usize),
                    },
                )
            }

            mir::Rvalue::ThreadLocalRef(def_id) => {
                assert!(bx.cx().tcx().is_static(def_id));
                let static_ = bx.get_static(def_id);
                let layout = bx.layout_of(bx.cx().tcx().static_ptr_ty(def_id));
                let operand = OperandRef::from_immediate_or_packed_pair(&mut bx, static_, layout);
                (bx, operand)
            }
            mir::Rvalue::Use(ref operand) => {
                let operand = self.codegen_operand(&mut bx, operand);
                (bx, operand)
            }
            mir::Rvalue::Repeat(..) | mir::Rvalue::Aggregate(..) => {
                // According to `rvalue_creates_operand`, only ZST
                // aggregate rvalues are allowed to be operands.
                let ty = rvalue.ty(self.mir, self.cx.tcx());
                let operand =
                    OperandRef::new_zst(&mut bx, self.cx.layout_of(self.monomorphize(ty)));
                (bx, operand)
            }
            mir::Rvalue::ShallowInitBox(ref operand, content_ty) => {
                let operand = self.codegen_operand(&mut bx, operand);
                let lloperand = operand.immediate();

                let content_ty = self.monomorphize(content_ty);
                let box_layout = bx.cx().layout_of(bx.tcx().mk_box(content_ty));
                let llty_ptr = bx.cx().backend_type(box_layout);

                let val = bx.pointercast(lloperand, llty_ptr);
                let operand = OperandRef { val: OperandValue::Immediate(val), layout: box_layout };
                (bx, operand)
            }
        }
    }

    fn evaluate_array_len(&mut self, bx: &mut Bx, place: mir::Place<'tcx>) -> Bx::Value {
        // ZST are passed as operands and require special handling
        // because codegen_place() panics if Local is operand.
        if let Some(index) = place.as_local() {
            if let LocalRef::Operand(Some(op)) = self.locals[index] {
                if let ty::Array(_, n) = op.layout.ty.kind() {
                    let n = n.eval_usize(bx.cx().tcx(), ty::ParamEnv::reveal_all());
                    return bx.cx().const_usize(n);
                }
            }
        }
        // use common size calculation for non zero-sized types
        let cg_value = self.codegen_place(bx, place.as_ref());
        cg_value.len(bx.cx())
    }

    /// Codegen an `Rvalue::AddressOf` or `Rvalue::Ref`
    fn codegen_place_to_pointer(
        &mut self,
        mut bx: Bx,
        place: mir::Place<'tcx>,
        mk_ptr_ty: impl FnOnce(TyCtxt<'tcx>, Ty<'tcx>) -> Ty<'tcx>,
    ) -> (Bx, OperandRef<'tcx, Bx::Value>) {
        let cg_place = self.codegen_place(&mut bx, place.as_ref());

        let ty = cg_place.layout.ty;

        // Note: places are indirect, so storing the `llval` into the
        // destination effectively creates a reference.
        let val = if !bx.cx().type_has_metadata(ty) {
            OperandValue::Immediate(cg_place.llval)
        } else {
            OperandValue::Pair(cg_place.llval, cg_place.llextra.unwrap())
        };
        (bx, OperandRef { val, layout: self.cx.layout_of(mk_ptr_ty(self.cx.tcx(), ty)) })
    }

    pub fn codegen_scalar_binop(
        &mut self,
        bx: &mut Bx,
        op: mir::BinOp,
        lhs: Bx::Value,
        rhs: Bx::Value,
        input_ty: Ty<'tcx>,
    ) -> Bx::Value {
        let is_float = input_ty.is_floating_point();
        let is_signed = input_ty.is_signed();
        match op {
            mir::BinOp::Add => {
                if is_float {
                    bx.fadd(lhs, rhs)
                } else {
                    bx.add(lhs, rhs)
                }
            }
            mir::BinOp::Sub => {
                if is_float {
                    bx.fsub(lhs, rhs)
                } else {
                    bx.sub(lhs, rhs)
                }
            }
            mir::BinOp::Mul => {
                if is_float {
                    bx.fmul(lhs, rhs)
                } else {
                    bx.mul(lhs, rhs)
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
                let pointee_type = input_ty
                    .builtin_deref(true)
                    .unwrap_or_else(|| bug!("deref of non-pointer {:?}", input_ty))
                    .ty;
                let llty = bx.cx().backend_type(bx.cx().layout_of(pointee_type));
                bx.inbounds_gep(llty, lhs, &[rhs])
            }
            mir::BinOp::Shl => common::build_unchecked_lshift(bx, lhs, rhs),
            mir::BinOp::Shr => common::build_unchecked_rshift(bx, input_ty, lhs, rhs),
            mir::BinOp::Ne
            | mir::BinOp::Lt
            | mir::BinOp::Gt
            | mir::BinOp::Eq
            | mir::BinOp::Le
            | mir::BinOp::Ge => {
                if is_float {
                    bx.fcmp(base::bin_op_to_fcmp_predicate(op.to_hir_binop()), lhs, rhs)
                } else {
                    bx.icmp(base::bin_op_to_icmp_predicate(op.to_hir_binop(), is_signed), lhs, rhs)
                }
            }
        }
    }

    pub fn codegen_fat_ptr_binop(
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
                bug!("unexpected fat ptr binop");
            }
        }
    }

    pub fn codegen_scalar_checked_binop(
        &mut self,
        bx: &mut Bx,
        op: mir::BinOp,
        lhs: Bx::Value,
        rhs: Bx::Value,
        input_ty: Ty<'tcx>,
    ) -> OperandValue<Bx::Value> {
        // This case can currently arise only from functions marked
        // with #[rustc_inherit_overflow_checks] and inlined from
        // another crate (mostly core::num generic/#[inline] fns),
        // while the current crate doesn't use overflow checks.
        if !bx.cx().check_overflow() {
            let val = self.codegen_scalar_binop(bx, op, lhs, rhs, input_ty);
            return OperandValue::Pair(val, bx.cx().const_bool(false));
        }

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
            mir::BinOp::Shl | mir::BinOp::Shr => {
                let lhs_llty = bx.cx().val_ty(lhs);
                let rhs_llty = bx.cx().val_ty(rhs);
                let invert_mask = common::shift_mask_val(bx, lhs_llty, rhs_llty, true);
                let outer_bits = bx.and(rhs, invert_mask);

                let of = bx.icmp(IntPredicate::IntNE, outer_bits, bx.cx().const_null(rhs_llty));
                let val = self.codegen_scalar_binop(bx, op, lhs, rhs, input_ty);

                (val, of)
            }
            _ => bug!("Operator `{:?}` is not a checkable operator", op),
        };

        OperandValue::Pair(val, of)
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn rvalue_creates_operand(&self, rvalue: &mir::Rvalue<'tcx>, span: Span) -> bool {
        match *rvalue {
            mir::Rvalue::Ref(..) |
            mir::Rvalue::AddressOf(..) |
            mir::Rvalue::Len(..) |
            mir::Rvalue::Cast(..) | // (*)
            mir::Rvalue::ShallowInitBox(..) | // (*)
            mir::Rvalue::BinaryOp(..) |
            mir::Rvalue::CheckedBinaryOp(..) |
            mir::Rvalue::UnaryOp(..) |
            mir::Rvalue::Discriminant(..) |
            mir::Rvalue::NullaryOp(..) |
            mir::Rvalue::ThreadLocalRef(_) |
            mir::Rvalue::Use(..) => // (*)
                true,
            mir::Rvalue::Repeat(..) |
            mir::Rvalue::Aggregate(..) => {
                let ty = rvalue.ty(self.mir, self.cx.tcx());
                let ty = self.monomorphize(ty);
                self.cx.spanned_layout_of(ty, span).is_zst()
            }
        }

        // (*) this is only true if the type is suitable
    }
}
