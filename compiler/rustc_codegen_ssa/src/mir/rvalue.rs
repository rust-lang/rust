use super::operand::{OperandRef, OperandValue};
use super::place::PlaceRef;
use super::{FunctionCx, LocalRef};

use crate::base;
use crate::common::IntPredicate;
use crate::traits::*;
use crate::MemFlags;

use rustc_middle::mir;
use rustc_middle::ty::cast::{CastTy, IntTy};
use rustc_middle::ty::layout::{HasTyCtxt, LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, adjustment::PointerCoercion, Instance, Ty, TyCtxt};
use rustc_middle::{bug, span_bug};
use rustc_session::config::OptLevel;
use rustc_span::{Span, DUMMY_SP};
use rustc_target::abi::{self, FieldIdx, FIRST_VARIANT};

use arrayvec::ArrayVec;
use tracing::{debug, instrument};

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    #[instrument(level = "trace", skip(self, bx))]
    pub fn codegen_rvalue(
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
                mir::CastKind::PointerCoercion(PointerCoercion::Unsize),
                ref source,
                _,
            ) => {
                // The destination necessarily contains a fat pointer, so if
                // it's a scalar pair, it's a fat pointer or newtype thereof.
                if bx.cx().is_backend_scalar_pair(dest.layout) {
                    // Into-coerce of a thin pointer to a fat pointer -- just
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
                let cg_elem = self.codegen_operand(bx, elem);

                // Do not generate the loop for zero-sized elements or empty arrays.
                if dest.layout.is_zst() {
                    return;
                }

                if let OperandValue::Immediate(v) = cg_elem.val {
                    let start = dest.val.llval;
                    let size = bx.const_usize(dest.layout.size.bytes());

                    // Use llvm.memset.p0i8.* to initialize all zero arrays
                    if bx.cx().const_to_opt_u128(v, false) == Some(0) {
                        let fill = bx.cx().const_u8(0);
                        bx.memset(start, fill, size, dest.val.align, MemFlags::empty());
                        return;
                    }

                    // Use llvm.memset.p0i8.* to initialize byte arrays
                    let v = bx.from_immediate(v);
                    if bx.cx().val_ty(v) == bx.cx().type_i8() {
                        bx.memset(start, v, size, dest.val.align, MemFlags::empty());
                        return;
                    }
                }

                let count = self
                    .monomorphize(count)
                    .eval_target_usize(bx.cx().tcx(), ty::ParamEnv::reveal_all());

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

        if let Some(val) = self.codegen_transmute_operand(bx, src, dst.layout) {
            val.store(bx, dst);
            return;
        }

        match src.val {
            OperandValue::Ref(..) | OperandValue::ZeroSized => {
                span_bug!(
                    self.mir.span,
                    "Operand path should have handled transmute \
                    from {src:?} to place {dst:?}"
                );
            }
            OperandValue::Immediate(..) | OperandValue::Pair(..) => {
                // When we have immediate(s), the alignment of the source is irrelevant,
                // so we can store them using the destination's alignment.
                src.val.store(bx, dst.val.with_type(src.layout));
            }
        }
    }

    /// Attempts to transmute an `OperandValue` to another `OperandValue`.
    ///
    /// Returns `None` for cases that can't work in that framework, such as for
    /// `Immediate`->`Ref` that needs an `alloc` to get the location.
    fn codegen_transmute_operand(
        &mut self,
        bx: &mut Bx,
        operand: OperandRef<'tcx, Bx::Value>,
        cast: TyAndLayout<'tcx>,
    ) -> Option<OperandValue<Bx::Value>> {
        // Check for transmutes that are always UB.
        if operand.layout.size != cast.size
            || operand.layout.abi.is_uninhabited()
            || cast.abi.is_uninhabited()
        {
            if !operand.layout.abi.is_uninhabited() {
                // Since this is known statically and the input could have existed
                // without already having hit UB, might as well trap for it.
                bx.abort();
            }

            // Because this transmute is UB, return something easy to generate,
            // since it's fine that later uses of the value are probably UB.
            return Some(OperandValue::poison(bx, cast));
        }

        let operand_kind = self.value_kind(operand.layout);
        let cast_kind = self.value_kind(cast);

        match operand.val {
            OperandValue::Ref(source_place_val) => {
                assert_eq!(source_place_val.llextra, None);
                assert!(matches!(operand_kind, OperandValueKind::Ref));
                Some(bx.load_operand(source_place_val.with_type(cast)).val)
            }
            OperandValue::ZeroSized => {
                let OperandValueKind::ZeroSized = operand_kind else {
                    bug!("Found {operand_kind:?} for operand {operand:?}");
                };
                if let OperandValueKind::ZeroSized = cast_kind {
                    Some(OperandValue::ZeroSized)
                } else {
                    None
                }
            }
            OperandValue::Immediate(imm) => {
                let OperandValueKind::Immediate(in_scalar) = operand_kind else {
                    bug!("Found {operand_kind:?} for operand {operand:?}");
                };
                if let OperandValueKind::Immediate(out_scalar) = cast_kind
                    && in_scalar.size(self.cx) == out_scalar.size(self.cx)
                {
                    let operand_bty = bx.backend_type(operand.layout);
                    let cast_bty = bx.backend_type(cast);
                    Some(OperandValue::Immediate(self.transmute_immediate(
                        bx,
                        imm,
                        in_scalar,
                        operand_bty,
                        out_scalar,
                        cast_bty,
                    )))
                } else {
                    None
                }
            }
            OperandValue::Pair(imm_a, imm_b) => {
                let OperandValueKind::Pair(in_a, in_b) = operand_kind else {
                    bug!("Found {operand_kind:?} for operand {operand:?}");
                };
                if let OperandValueKind::Pair(out_a, out_b) = cast_kind
                    && in_a.size(self.cx) == out_a.size(self.cx)
                    && in_b.size(self.cx) == out_b.size(self.cx)
                {
                    let in_a_ibty = bx.scalar_pair_element_backend_type(operand.layout, 0, false);
                    let in_b_ibty = bx.scalar_pair_element_backend_type(operand.layout, 1, false);
                    let out_a_ibty = bx.scalar_pair_element_backend_type(cast, 0, false);
                    let out_b_ibty = bx.scalar_pair_element_backend_type(cast, 1, false);
                    Some(OperandValue::Pair(
                        self.transmute_immediate(bx, imm_a, in_a, in_a_ibty, out_a, out_a_ibty),
                        self.transmute_immediate(bx, imm_b, in_b, in_b_ibty, out_b, out_b_ibty),
                    ))
                } else {
                    None
                }
            }
        }
    }

    /// Transmutes one of the immediates from an [`OperandValue::Immediate`]
    /// or an [`OperandValue::Pair`] to an immediate of the target type.
    ///
    /// `to_backend_ty` must be the *non*-immediate backend type (so it will be
    /// `i8`, not `i1`, for `bool`-like types.)
    fn transmute_immediate(
        &self,
        bx: &mut Bx,
        mut imm: Bx::Value,
        from_scalar: abi::Scalar,
        from_backend_ty: Bx::Type,
        to_scalar: abi::Scalar,
        to_backend_ty: Bx::Type,
    ) -> Bx::Value {
        assert_eq!(from_scalar.size(self.cx), to_scalar.size(self.cx));

        use abi::Primitive::*;
        imm = bx.from_immediate(imm);

        // When scalars are passed by value, there's no metadata recording their
        // valid ranges. For example, `char`s are passed as just `i32`, with no
        // way for LLVM to know that they're 0x10FFFF at most. Thus we assume
        // the range of the input value too, not just the output range.
        self.assume_scalar_range(bx, imm, from_scalar, from_backend_ty);

        imm = match (from_scalar.primitive(), to_scalar.primitive()) {
            (Int(..) | Float(_), Int(..) | Float(_)) => bx.bitcast(imm, to_backend_ty),
            (Pointer(..), Pointer(..)) => bx.pointercast(imm, to_backend_ty),
            (Int(..), Pointer(..)) => bx.ptradd(bx.const_null(bx.type_ptr()), imm),
            (Pointer(..), Int(..)) => bx.ptrtoint(imm, to_backend_ty),
            (Float(_), Pointer(..)) => {
                let int_imm = bx.bitcast(imm, bx.cx().type_isize());
                bx.ptradd(bx.const_null(bx.type_ptr()), int_imm)
            }
            (Pointer(..), Float(_)) => {
                let int_imm = bx.ptrtoint(imm, bx.cx().type_isize());
                bx.bitcast(int_imm, to_backend_ty)
            }
        };
        self.assume_scalar_range(bx, imm, to_scalar, to_backend_ty);
        imm = bx.to_immediate_scalar(imm, to_scalar);
        imm
    }

    fn assume_scalar_range(
        &self,
        bx: &mut Bx,
        imm: Bx::Value,
        scalar: abi::Scalar,
        backend_ty: Bx::Type,
    ) {
        if matches!(self.cx.sess().opts.optimize, OptLevel::No | OptLevel::Less)
            // For now, the critical niches are all over `Int`eger values.
            // Should floating-point values or pointers ever get more complex
            // niches, then this code will probably want to handle them too.
            || !matches!(scalar.primitive(), abi::Primitive::Int(..))
            || scalar.is_always_valid(self.cx)
        {
            return;
        }

        let abi::WrappingRange { start, end } = scalar.valid_range(self.cx);

        if start <= end {
            if start > 0 {
                let low = bx.const_uint_big(backend_ty, start);
                let cmp = bx.icmp(IntPredicate::IntUGE, imm, low);
                bx.assume(cmp);
            }

            let type_max = scalar.size(self.cx).unsigned_int_max();
            if end < type_max {
                let high = bx.const_uint_big(backend_ty, end);
                let cmp = bx.icmp(IntPredicate::IntULE, imm, high);
                bx.assume(cmp);
            }
        } else {
            let low = bx.const_uint_big(backend_ty, start);
            let cmp_low = bx.icmp(IntPredicate::IntUGE, imm, low);

            let high = bx.const_uint_big(backend_ty, end);
            let cmp_high = bx.icmp(IntPredicate::IntULE, imm, high);

            let or = bx.or(cmp_low, cmp_high);
            bx.assume(or);
        }
    }

    pub fn codegen_rvalue_unsized(
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

    pub fn codegen_rvalue_operand(
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
                    mir::CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer) => {
                        match *operand.layout.ty.kind() {
                            ty::FnDef(def_id, args) => {
                                let instance = ty::Instance::resolve_for_fn_ptr(
                                    bx.tcx(),
                                    ty::ParamEnv::reveal_all(),
                                    def_id,
                                    args,
                                )
                                .unwrap()
                                .polymorphize(bx.cx().tcx());
                                OperandValue::Immediate(bx.get_fn_addr(instance))
                            }
                            _ => bug!("{} cannot be reified to a fn ptr", operand.layout.ty),
                        }
                    }
                    mir::CastKind::PointerCoercion(PointerCoercion::ClosureFnPointer(_)) => {
                        match *operand.layout.ty.kind() {
                            ty::Closure(def_id, args) => {
                                let instance = Instance::resolve_closure(
                                    bx.cx().tcx(),
                                    def_id,
                                    args,
                                    ty::ClosureKind::FnOnce,
                                )
                                .polymorphize(bx.cx().tcx());
                                OperandValue::Immediate(bx.cx().get_fn_addr(instance))
                            }
                            _ => bug!("{} cannot be cast to a fn ptr", operand.layout.ty),
                        }
                    }
                    mir::CastKind::PointerCoercion(PointerCoercion::UnsafeFnPointer) => {
                        // This is a no-op at the LLVM level.
                        operand.val
                    }
                    mir::CastKind::PointerCoercion(PointerCoercion::Unsize) => {
                        assert!(bx.cx().is_backend_scalar_pair(cast));
                        let (lldata, llextra) = operand.val.pointer_parts();
                        let (lldata, llextra) =
                            base::unsize_ptr(bx, lldata, operand.layout.ty, cast.ty, llextra);
                        OperandValue::Pair(lldata, llextra)
                    }
                    mir::CastKind::PointerCoercion(
                        PointerCoercion::MutToConstPointer | PointerCoercion::ArrayToPointer,
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
                                // Cast of fat-ptr to thin-ptr is an extraction of data-ptr.
                                OperandValue::Immediate(data_ptr)
                            }
                        } else {
                            bug!("unexpected non-pair operand");
                        }
                    }
                    mir::CastKind::DynStar => {
                        let (lldata, llextra) = operand.val.pointer_parts();
                        let (lldata, llextra) =
                            base::cast_to_dyn_star(bx, lldata, operand.layout, cast.ty, llextra);
                        OperandValue::Pair(lldata, llextra)
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
                        assert!(bx.cx().is_backend_immediate(cast));
                        let ll_t_out = bx.cx().immediate_backend_type(cast);
                        if operand.layout.abi.is_uninhabited() {
                            let val = OperandValue::Immediate(bx.cx().const_poison(ll_t_out));
                            return OperandRef { val, layout: cast };
                        }
                        let r_t_in =
                            CastTy::from_ty(operand.layout.ty).expect("bad input type for cast");
                        let r_t_out = CastTy::from_ty(cast.ty).expect("bad output type for cast");
                        let ll_t_in = bx.cx().immediate_backend_type(operand.layout);
                        let llval = operand.immediate();

                        let newval = match (r_t_in, r_t_out) {
                            (CastTy::Int(i), CastTy::Int(_)) => {
                                bx.intcast(llval, ll_t_out, i.is_signed())
                            }
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
                            (CastTy::Int(i), CastTy::Float) => {
                                if i.is_signed() {
                                    bx.sitofp(llval, ll_t_out)
                                } else {
                                    bx.uitofp(llval, ll_t_out)
                                }
                            }
                            (CastTy::Ptr(_) | CastTy::FnPtr, CastTy::Ptr(_)) => {
                                bx.pointercast(llval, ll_t_out)
                            }
                            (CastTy::Int(i), CastTy::Ptr(_)) => {
                                let usize_llval =
                                    bx.intcast(llval, bx.cx().type_isize(), i.is_signed());
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
            mir::Rvalue::AddressOf(mutability, place) => {
                let mk_ptr =
                    move |tcx: TyCtxt<'tcx>, ty: Ty<'tcx>| Ty::new_ptr(tcx, ty, mutability);
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
                    ) => self.codegen_fat_ptr_binop(
                        bx,
                        op,
                        lhs_addr,
                        lhs_extra,
                        rhs_addr,
                        rhs_extra,
                        lhs.layout.ty,
                    ),

                    (OperandValue::Immediate(lhs_val), OperandValue::Immediate(rhs_val)) => {
                        self.codegen_scalar_binop(bx, op, lhs_val, rhs_val, lhs.layout.ty)
                    }

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
                        assert!(operand.layout.ty.is_unsafe_ptr() || operand.layout.ty.is_ref(),);
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
                let discr = self.codegen_place(bx, place.as_ref()).codegen_get_discr(bx, discr_ty);
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
                            .offset_of_subfield(bx.param_env(), layout, fields.iter())
                            .bytes();
                        bx.cx().const_usize(val)
                    }
                    mir::NullOp::UbChecks => {
                        let val = bx.tcx().sess.ub_checks();
                        bx.cx().const_bool(val)
                    }
                };
                let tcx = self.cx.tcx();
                OperandRef {
                    val: OperandValue::Immediate(val),
                    layout: self.cx.layout_of(tcx.types.usize),
                }
            }

            mir::Rvalue::ThreadLocalRef(def_id) => {
                assert!(bx.cx().tcx().is_static(def_id));
                let layout = bx.layout_of(bx.cx().tcx().static_ptr_ty(def_id));
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
                let mut inputs = ArrayVec::<Bx::Value, 2>::new();
                let mut input_scalars = ArrayVec::<abi::Scalar, 2>::new();
                for field_idx in layout.fields.index_by_increasing_offset() {
                    let field_idx = FieldIdx::from_usize(field_idx);
                    let op = self.codegen_operand(bx, &fields[field_idx]);
                    let values = op.val.immediates_or_place().left_or_else(|p| {
                        bug!("Field {field_idx:?} is {p:?} making {layout:?}");
                    });
                    let scalars = self.value_kind(op.layout).scalars().unwrap();
                    assert_eq!(values.len(), scalars.len());
                    inputs.extend(values);
                    input_scalars.extend(scalars);
                }

                let output_scalars = self.value_kind(layout).scalars().unwrap();
                itertools::izip!(&mut inputs, input_scalars, output_scalars).for_each(
                    |(v, in_s, out_s)| {
                        if in_s != out_s {
                            // We have to be really careful about bool here, because
                            // `(bool,)` stays i1 but `Cell<bool>` becomes i8.
                            *v = bx.from_immediate(*v);
                            *v = bx.to_immediate_scalar(*v, out_s);
                        }
                    },
                );

                let val = OperandValue::from_immediates(inputs);
                assert!(
                    val.is_expected_variant_for_type(self.cx, layout),
                    "Made wrong variant {val:?} for type {layout:?}",
                );
                OperandRef { val, layout }
            }
            mir::Rvalue::ShallowInitBox(ref operand, content_ty) => {
                let operand = self.codegen_operand(bx, operand);
                let val = operand.immediate();

                let content_ty = self.monomorphize(content_ty);
                let box_layout = bx.cx().layout_of(Ty::new_box(bx.tcx(), content_ty));

                OperandRef { val: OperandValue::Immediate(val), layout: box_layout }
            }
        }
    }

    fn evaluate_array_len(&mut self, bx: &mut Bx, place: mir::Place<'tcx>) -> Bx::Value {
        // ZST are passed as operands and require special handling
        // because codegen_place() panics if Local is operand.
        if let Some(index) = place.as_local() {
            if let LocalRef::Operand(op) = self.locals[index] {
                if let ty::Array(_, n) = op.layout.ty.kind() {
                    let n = n.eval_target_usize(bx.cx().tcx(), ty::ParamEnv::reveal_all());
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
        bx: &mut Bx,
        place: mir::Place<'tcx>,
        mk_ptr_ty: impl FnOnce(TyCtxt<'tcx>, Ty<'tcx>) -> Ty<'tcx>,
    ) -> OperandRef<'tcx, Bx::Value> {
        let cg_place = self.codegen_place(bx, place.as_ref());
        let val = cg_place.val.address();

        let ty = cg_place.layout.ty;
        assert!(
            if bx.cx().type_has_metadata(ty) {
                matches!(val, OperandValue::Pair(..))
            } else {
                matches!(val, OperandValue::Immediate(..))
            },
            "Address of place was unexpectedly {val:?} for pointee type {ty:?}",
        );

        OperandRef { val, layout: self.cx.layout_of(mk_ptr_ty(self.cx.tcx(), ty)) }
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
                let pointee_type = input_ty
                    .builtin_deref(true)
                    .unwrap_or_else(|| bug!("deref of non-pointer {:?}", input_ty));
                let pointee_layout = bx.cx().layout_of(pointee_type);
                if pointee_layout.is_zst() {
                    // `Offset` works in terms of the size of pointee,
                    // so offsetting a pointer to ZST is a noop.
                    lhs
                } else {
                    let llty = bx.cx().backend_type(pointee_layout);
                    bx.inbounds_gep(llty, lhs, &[rhs])
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
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn rvalue_creates_operand(&self, rvalue: &mir::Rvalue<'tcx>, span: Span) -> bool {
        match *rvalue {
            mir::Rvalue::Cast(mir::CastKind::Transmute, ref operand, cast_ty) => {
                let operand_ty = operand.ty(self.mir, self.cx.tcx());
                let cast_layout = self.cx.layout_of(self.monomorphize(cast_ty));
                let operand_layout = self.cx.layout_of(self.monomorphize(operand_ty));

                match (self.value_kind(operand_layout), self.value_kind(cast_layout)) {
                    // Can always load from a pointer as needed
                    (OperandValueKind::Ref, _) => true,

                    // ZST-to-ZST is the easiest thing ever
                    (OperandValueKind::ZeroSized, OperandValueKind::ZeroSized) => true,

                    // But if only one of them is a ZST the sizes can't match
                    (OperandValueKind::ZeroSized, _) | (_, OperandValueKind::ZeroSized) => false,

                    // Need to generate an `alloc` to get a pointer from an immediate
                    (OperandValueKind::Immediate(..) | OperandValueKind::Pair(..), OperandValueKind::Ref) => false,

                    // When we have scalar immediates, we can only convert things
                    // where the sizes match, to avoid endianness questions.
                    (OperandValueKind::Immediate(a), OperandValueKind::Immediate(b)) =>
                        a.size(self.cx) == b.size(self.cx),
                    (OperandValueKind::Pair(a0, a1), OperandValueKind::Pair(b0, b1)) =>
                        a0.size(self.cx) == b0.size(self.cx) && a1.size(self.cx) == b1.size(self.cx),

                    // Send mixings between scalars and pairs through the memory route
                    // FIXME: Maybe this could use insertvalue/extractvalue instead?
                    (OperandValueKind::Immediate(..), OperandValueKind::Pair(..)) |
                    (OperandValueKind::Pair(..), OperandValueKind::Immediate(..)) => false,
                }
            }
            mir::Rvalue::Ref(..) |
            mir::Rvalue::CopyForDeref(..) |
            mir::Rvalue::AddressOf(..) |
            mir::Rvalue::Len(..) |
            mir::Rvalue::Cast(..) | // (*)
            mir::Rvalue::ShallowInitBox(..) | // (*)
            mir::Rvalue::BinaryOp(..) |
            mir::Rvalue::UnaryOp(..) |
            mir::Rvalue::Discriminant(..) |
            mir::Rvalue::NullaryOp(..) |
            mir::Rvalue::ThreadLocalRef(_) |
            mir::Rvalue::Use(..) => // (*)
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
                    !self.cx.is_backend_ref(layout)
                }
            }
        }

        // (*) this is only true if the type is suitable
    }

    /// Gets which variant of [`OperandValue`] is expected for a particular type.
    fn value_kind(&self, layout: TyAndLayout<'tcx>) -> OperandValueKind {
        if layout.is_zst() {
            OperandValueKind::ZeroSized
        } else if self.cx.is_backend_immediate(layout) {
            assert!(!self.cx.is_backend_scalar_pair(layout));
            OperandValueKind::Immediate(match layout.abi {
                abi::Abi::Scalar(s) => s,
                abi::Abi::Vector { element, .. } => element,
                x => span_bug!(self.mir.span, "Couldn't translate {x:?} as backend immediate"),
            })
        } else if self.cx.is_backend_scalar_pair(layout) {
            let abi::Abi::ScalarPair(s1, s2) = layout.abi else {
                span_bug!(
                    self.mir.span,
                    "Couldn't translate {:?} as backend scalar pair",
                    layout.abi,
                );
            };
            OperandValueKind::Pair(s1, s2)
        } else {
            OperandValueKind::Ref
        }
    }
}

/// The variants of this match [`OperandValue`], giving details about the
/// backend values that will be held in that other type.
#[derive(Debug, Copy, Clone)]
enum OperandValueKind {
    Ref,
    Immediate(abi::Scalar),
    Pair(abi::Scalar, abi::Scalar),
    ZeroSized,
}

impl OperandValueKind {
    fn scalars(self) -> Option<ArrayVec<abi::Scalar, 2>> {
        Some(match self {
            OperandValueKind::ZeroSized => ArrayVec::new(),
            OperandValueKind::Immediate(a) => ArrayVec::from_iter([a]),
            OperandValueKind::Pair(a, b) => [a, b].into(),
            OperandValueKind::Ref => return None,
        })
    }
}
