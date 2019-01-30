//! Functions concerning immediate values and operands, and reading from operands.
//! All high-level functions to read from memory work on operands as sources.

use std::convert::TryInto;

use rustc::{mir, ty};
use rustc::ty::layout::{self, Size, LayoutOf, TyLayout, HasDataLayout, IntegerExt, VariantIdx};

use rustc::mir::interpret::{
    GlobalId, AllocId, InboundsCheck,
    ConstValue, Pointer, Scalar,
    EvalResult, EvalErrorKind,
};
use super::{EvalContext, Machine, MemPlace, MPlaceTy, MemoryKind};
pub use rustc::mir::interpret::ScalarMaybeUndef;

/// A `Value` represents a single immediate self-contained Rust value.
///
/// For optimization of a few very common cases, there is also a representation for a pair of
/// primitive values (`ScalarPair`). It allows Miri to avoid making allocations for checked binary
/// operations and fat pointers. This idea was taken from rustc's codegen.
/// In particular, thanks to `ScalarPair`, arithmetic operations and casts can be entirely
/// defined on `Immediate`, and do not have to work with a `Place`.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Immediate<Tag=(), Id=AllocId> {
    Scalar(ScalarMaybeUndef<Tag, Id>),
    ScalarPair(ScalarMaybeUndef<Tag, Id>, ScalarMaybeUndef<Tag, Id>),
}

impl Immediate {
    #[inline]
    pub fn with_default_tag<Tag>(self) -> Immediate<Tag>
        where Tag: Default
    {
        match self {
            Immediate::Scalar(x) => Immediate::Scalar(x.with_default_tag()),
            Immediate::ScalarPair(x, y) =>
                Immediate::ScalarPair(x.with_default_tag(), y.with_default_tag()),
        }
    }
}

impl<'tcx, Tag> Immediate<Tag> {
    #[inline]
    pub fn erase_tag(self) -> Immediate
    {
        match self {
            Immediate::Scalar(x) => Immediate::Scalar(x.erase_tag()),
            Immediate::ScalarPair(x, y) =>
                Immediate::ScalarPair(x.erase_tag(), y.erase_tag()),
        }
    }

    pub fn new_slice(
        val: Scalar<Tag>,
        len: u64,
        cx: &impl HasDataLayout
    ) -> Self {
        Immediate::ScalarPair(
            val.into(),
            Scalar::from_uint(len, cx.data_layout().pointer_size).into(),
        )
    }

    pub fn new_dyn_trait(val: Scalar<Tag>, vtable: Pointer<Tag>) -> Self {
        Immediate::ScalarPair(val.into(), Scalar::Ptr(vtable).into())
    }

    #[inline]
    pub fn to_scalar_or_undef(self) -> ScalarMaybeUndef<Tag> {
        match self {
            Immediate::Scalar(val) => val,
            Immediate::ScalarPair(..) => bug!("Got a fat pointer where a scalar was expected"),
        }
    }

    #[inline]
    pub fn to_scalar(self) -> EvalResult<'tcx, Scalar<Tag>> {
        self.to_scalar_or_undef().not_undef()
    }

    #[inline]
    pub fn to_scalar_pair(self) -> EvalResult<'tcx, (Scalar<Tag>, Scalar<Tag>)> {
        match self {
            Immediate::Scalar(..) => bug!("Got a thin pointer where a scalar pair was expected"),
            Immediate::ScalarPair(a, b) => Ok((a.not_undef()?, b.not_undef()?))
        }
    }

    /// Convert the immediate into a pointer (or a pointer-sized integer).
    /// Throws away the second half of a ScalarPair!
    #[inline]
    pub fn to_scalar_ptr(self) -> EvalResult<'tcx, Scalar<Tag>> {
        match self {
            Immediate::Scalar(ptr) |
            Immediate::ScalarPair(ptr, _) => ptr.not_undef(),
        }
    }

    /// Convert the value into its metadata.
    /// Throws away the first half of a ScalarPair!
    #[inline]
    pub fn to_meta(self) -> EvalResult<'tcx, Option<Scalar<Tag>>> {
        Ok(match self {
            Immediate::Scalar(_) => None,
            Immediate::ScalarPair(_, meta) => Some(meta.not_undef()?),
        })
    }
}

// ScalarPair needs a type to interpret, so we often have an immediate and a type together
// as input for binary and cast operations.
#[derive(Copy, Clone, Debug)]
pub struct ImmTy<'tcx, Tag=()> {
    immediate: Immediate<Tag>,
    pub layout: TyLayout<'tcx>,
}

impl<'tcx, Tag> ::std::ops::Deref for ImmTy<'tcx, Tag> {
    type Target = Immediate<Tag>;
    #[inline(always)]
    fn deref(&self) -> &Immediate<Tag> {
        &self.immediate
    }
}

/// An `Operand` is the result of computing a `mir::Operand`. It can be immediate,
/// or still in memory.  The latter is an optimization, to delay reading that chunk of
/// memory and to avoid having to store arbitrary-sized data here.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Operand<Tag=(), Id=AllocId> {
    Immediate(Immediate<Tag, Id>),
    Indirect(MemPlace<Tag, Id>),
}

impl Operand {
    #[inline]
    pub fn with_default_tag<Tag>(self) -> Operand<Tag>
        where Tag: Default
    {
        match self {
            Operand::Immediate(x) => Operand::Immediate(x.with_default_tag()),
            Operand::Indirect(x) => Operand::Indirect(x.with_default_tag()),
        }
    }
}

impl<Tag> Operand<Tag> {
    #[inline]
    pub fn erase_tag(self) -> Operand
    {
        match self {
            Operand::Immediate(x) => Operand::Immediate(x.erase_tag()),
            Operand::Indirect(x) => Operand::Indirect(x.erase_tag()),
        }
    }

    #[inline]
    pub fn to_mem_place(self) -> MemPlace<Tag>
        where Tag: ::std::fmt::Debug
    {
        match self {
            Operand::Indirect(mplace) => mplace,
            _ => bug!("to_mem_place: expected Operand::Indirect, got {:?}", self),

        }
    }

    #[inline]
    pub fn to_immediate(self) -> Immediate<Tag>
        where Tag: ::std::fmt::Debug
    {
        match self {
            Operand::Immediate(imm) => imm,
            _ => bug!("to_immediate: expected Operand::Immediate, got {:?}", self),

        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct OpTy<'tcx, Tag=()> {
    crate op: Operand<Tag>, // ideally we'd make this private, but const_prop needs this
    pub layout: TyLayout<'tcx>,
}

impl<'tcx, Tag> ::std::ops::Deref for OpTy<'tcx, Tag> {
    type Target = Operand<Tag>;
    #[inline(always)]
    fn deref(&self) -> &Operand<Tag> {
        &self.op
    }
}

impl<'tcx, Tag: Copy> From<MPlaceTy<'tcx, Tag>> for OpTy<'tcx, Tag> {
    #[inline(always)]
    fn from(mplace: MPlaceTy<'tcx, Tag>) -> Self {
        OpTy {
            op: Operand::Indirect(*mplace),
            layout: mplace.layout
        }
    }
}

impl<'tcx, Tag> From<ImmTy<'tcx, Tag>> for OpTy<'tcx, Tag> {
    #[inline(always)]
    fn from(val: ImmTy<'tcx, Tag>) -> Self {
        OpTy {
            op: Operand::Immediate(val.immediate),
            layout: val.layout
        }
    }
}

impl<'tcx, Tag> OpTy<'tcx, Tag>
{
    #[inline]
    pub fn erase_tag(self) -> OpTy<'tcx>
    {
        OpTy {
            op: self.op.erase_tag(),
            layout: self.layout,
        }
    }
}

// Use the existing layout if given (but sanity check in debug mode),
// or compute the layout.
#[inline(always)]
pub(super) fn from_known_layout<'tcx>(
    layout: Option<TyLayout<'tcx>>,
    compute: impl FnOnce() -> EvalResult<'tcx, TyLayout<'tcx>>
) -> EvalResult<'tcx, TyLayout<'tcx>> {
    match layout {
        None => compute(),
        Some(layout) => {
            if cfg!(debug_assertions) {
                let layout2 = compute()?;
                assert_eq!(layout.details, layout2.details,
                    "Mismatch in layout of supposedly equal-layout types {:?} and {:?}",
                    layout.ty, layout2.ty);
            }
            Ok(layout)
        }
    }
}

impl<'a, 'mir, 'tcx, M: Machine<'a, 'mir, 'tcx>> EvalContext<'a, 'mir, 'tcx, M> {
    /// Try reading an immediate in memory; this is interesting particularly for ScalarPair.
    /// Return None if the layout does not permit loading this as a value.
    pub(super) fn try_read_immediate_from_mplace(
        &self,
        mplace: MPlaceTy<'tcx, M::PointerTag>,
    ) -> EvalResult<'tcx, Option<Immediate<M::PointerTag>>> {
        if mplace.layout.is_unsized() {
            // Don't touch unsized
            return Ok(None);
        }
        let (ptr, ptr_align) = mplace.to_scalar_ptr_align();

        if mplace.layout.is_zst() {
            // Not all ZSTs have a layout we would handle below, so just short-circuit them
            // all here.
            self.memory.check_align(ptr, ptr_align)?;
            return Ok(Some(Immediate::Scalar(Scalar::zst().into())));
        }

        // check for integer pointers before alignment to report better errors
        let ptr = ptr.to_ptr()?;
        self.memory.check_align(ptr.into(), ptr_align)?;
        match mplace.layout.abi {
            layout::Abi::Scalar(..) => {
                let scalar = self.memory
                    .get(ptr.alloc_id)?
                    .read_scalar(self, ptr, mplace.layout.size)?;
                Ok(Some(Immediate::Scalar(scalar)))
            }
            layout::Abi::ScalarPair(ref a, ref b) => {
                let (a, b) = (&a.value, &b.value);
                let (a_size, b_size) = (a.size(self), b.size(self));
                let a_ptr = ptr;
                let b_offset = a_size.align_to(b.align(self).abi);
                assert!(b_offset.bytes() > 0); // we later use the offset to test which field to use
                let b_ptr = ptr.offset(b_offset, self)?;
                let a_val = self.memory
                    .get(ptr.alloc_id)?
                    .read_scalar(self, a_ptr, a_size)?;
                let b_align = ptr_align.restrict_for_offset(b_offset);
                self.memory.check_align(b_ptr.into(), b_align)?;
                let b_val = self.memory
                    .get(ptr.alloc_id)?
                    .read_scalar(self, b_ptr, b_size)?;
                Ok(Some(Immediate::ScalarPair(a_val, b_val)))
            }
            _ => Ok(None),
        }
    }

    /// Try returning an immediate for the operand.
    /// If the layout does not permit loading this as an immediate, return where in memory
    /// we can find the data.
    /// Note that for a given layout, this operation will either always fail or always
    /// succeed!  Whether it succeeds depends on whether the layout can be represented
    /// in a `Immediate`, not on which data is stored there currently.
    pub(crate) fn try_read_immediate(
        &self,
        src: OpTy<'tcx, M::PointerTag>,
    ) -> EvalResult<'tcx, Result<Immediate<M::PointerTag>, MemPlace<M::PointerTag>>> {
        Ok(match src.try_as_mplace() {
            Ok(mplace) => {
                if let Some(val) = self.try_read_immediate_from_mplace(mplace)? {
                    Ok(val)
                } else {
                    Err(*mplace)
                }
            },
            Err(val) => Ok(val),
        })
    }

    /// Read an immediate from a place, asserting that that is possible with the given layout.
    #[inline(always)]
    pub fn read_immediate(
        &self,
        op: OpTy<'tcx, M::PointerTag>
    ) -> EvalResult<'tcx, ImmTy<'tcx, M::PointerTag>> {
        if let Ok(immediate) = self.try_read_immediate(op)? {
            Ok(ImmTy { immediate, layout: op.layout })
        } else {
            bug!("primitive read failed for type: {:?}", op.layout.ty);
        }
    }

    /// Read a scalar from a place
    pub fn read_scalar(
        &self,
        op: OpTy<'tcx, M::PointerTag>
    ) -> EvalResult<'tcx, ScalarMaybeUndef<M::PointerTag>> {
        Ok(self.read_immediate(op)?.to_scalar_or_undef())
    }

    // Turn the MPlace into a string (must already be dereferenced!)
    pub fn read_str(
        &self,
        mplace: MPlaceTy<'tcx, M::PointerTag>,
    ) -> EvalResult<'tcx, &str> {
        let len = mplace.len(self)?;
        let bytes = self.memory.read_bytes(mplace.ptr, Size::from_bytes(len as u64))?;
        let str = ::std::str::from_utf8(bytes)
            .map_err(|err| EvalErrorKind::ValidationFailure(err.to_string()))?;
        Ok(str)
    }

    pub fn uninit_operand(
        &mut self,
        layout: TyLayout<'tcx>
    ) -> EvalResult<'tcx, Operand<M::PointerTag>> {
        // This decides which types we will use the Immediate optimization for, and hence should
        // match what `try_read_immediate` and `eval_place_to_op` support.
        if layout.is_zst() {
            return Ok(Operand::Immediate(Immediate::Scalar(Scalar::zst().into())));
        }

        Ok(match layout.abi {
            layout::Abi::Scalar(..) =>
                Operand::Immediate(Immediate::Scalar(ScalarMaybeUndef::Undef)),
            layout::Abi::ScalarPair(..) =>
                Operand::Immediate(Immediate::ScalarPair(
                    ScalarMaybeUndef::Undef,
                    ScalarMaybeUndef::Undef,
                )),
            _ => {
                trace!("Forcing allocation for local of type {:?}", layout.ty);
                Operand::Indirect(
                    *self.allocate(layout, MemoryKind::Stack)
                )
            }
        })
    }

    /// Projection functions
    pub fn operand_field(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
        field: u64,
    ) -> EvalResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        let base = match op.try_as_mplace() {
            Ok(mplace) => {
                // The easy case
                let field = self.mplace_field(mplace, field)?;
                return Ok(field.into());
            },
            Err(value) => value
        };

        let field = field.try_into().unwrap();
        let field_layout = op.layout.field(self, field)?;
        if field_layout.is_zst() {
            let immediate = Immediate::Scalar(Scalar::zst().into());
            return Ok(OpTy { op: Operand::Immediate(immediate), layout: field_layout });
        }
        let offset = op.layout.fields.offset(field);
        let immediate = match base {
            // the field covers the entire type
            _ if offset.bytes() == 0 && field_layout.size == op.layout.size => base,
            // extract fields from types with `ScalarPair` ABI
            Immediate::ScalarPair(a, b) => {
                let val = if offset.bytes() == 0 { a } else { b };
                Immediate::Scalar(val)
            },
            Immediate::Scalar(val) =>
                bug!("field access on non aggregate {:#?}, {:#?}", val, op.layout),
        };
        Ok(OpTy { op: Operand::Immediate(immediate), layout: field_layout })
    }

    pub fn operand_downcast(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
        variant: VariantIdx,
    ) -> EvalResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        // Downcasts only change the layout
        Ok(match op.try_as_mplace() {
            Ok(mplace) => {
                self.mplace_downcast(mplace, variant)?.into()
            },
            Err(..) => {
                let layout = op.layout.for_variant(self, variant);
                OpTy { layout, ..op }
            }
        })
    }

    pub fn operand_projection(
        &self,
        base: OpTy<'tcx, M::PointerTag>,
        proj_elem: &mir::PlaceElem<'tcx>,
    ) -> EvalResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        use rustc::mir::ProjectionElem::*;
        Ok(match *proj_elem {
            Field(field, _) => self.operand_field(base, field.index() as u64)?,
            Downcast(_, variant) => self.operand_downcast(base, variant)?,
            Deref => self.deref_operand(base)?.into(),
            Subslice { .. } | ConstantIndex { .. } | Index(_) => if base.layout.is_zst() {
                OpTy {
                    op: Operand::Immediate(Immediate::Scalar(Scalar::zst().into())),
                    // the actual index doesn't matter, so we just pick a convenient one like 0
                    layout: base.layout.field(self, 0)?,
                }
            } else {
                // The rest should only occur as mplace, we do not use Immediates for types
                // allowing such operations.  This matches place_projection forcing an allocation.
                let mplace = base.to_mem_place();
                self.mplace_projection(mplace, proj_elem)?.into()
            }
        })
    }

    /// This is used by [priroda](https://github.com/oli-obk/priroda) to get an OpTy from a local
    pub fn access_local(
        &self,
        frame: &super::Frame<'mir, 'tcx, M::PointerTag, M::FrameExtra>,
        local: mir::Local,
        layout: Option<TyLayout<'tcx>>,
    ) -> EvalResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        assert_ne!(local, mir::RETURN_PLACE);
        let op = *frame.locals[local].access()?;
        let layout = self.layout_of_local(frame, local, layout)?;
        Ok(OpTy { op, layout })
    }

    // Evaluate a place with the goal of reading from it.  This lets us sometimes
    // avoid allocations.
    fn eval_place_to_op(
        &self,
        mir_place: &mir::Place<'tcx>,
        layout: Option<TyLayout<'tcx>>,
    ) -> EvalResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        use rustc::mir::Place::*;
        let op = match *mir_place {
            Local(mir::RETURN_PLACE) => return err!(ReadFromReturnPointer),
            Local(local) => self.access_local(self.frame(), local, layout)?,

            Projection(ref proj) => {
                let op = self.eval_place_to_op(&proj.base, None)?;
                self.operand_projection(op, &proj.elem)?
            }

            _ => self.eval_place_to_mplace(mir_place)?.into(),
        };

        trace!("eval_place_to_op: got {:?}", *op);
        Ok(op)
    }

    /// Evaluate the operand, returning a place where you can then find the data.
    /// if you already know the layout, you can save two some table lookups
    /// by passing it in here.
    pub fn eval_operand(
        &self,
        mir_op: &mir::Operand<'tcx>,
        layout: Option<TyLayout<'tcx>>,
    ) -> EvalResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        use rustc::mir::Operand::*;
        let op = match *mir_op {
            // FIXME: do some more logic on `move` to invalidate the old location
            Copy(ref place) |
            Move(ref place) =>
                self.eval_place_to_op(place, layout)?,

            Constant(ref constant) => {
                let layout = from_known_layout(layout, || {
                    let ty = self.monomorphize(mir_op.ty(self.mir(), *self.tcx))?;
                    self.layout_of(ty)
                })?;
                let op = self.const_value_to_op(*constant.literal)?;
                OpTy { op, layout }
            }
        };
        trace!("{:?}: {:?}", mir_op, *op);
        Ok(op)
    }

    /// Evaluate a bunch of operands at once
    pub(super) fn eval_operands(
        &self,
        ops: &[mir::Operand<'tcx>],
    ) -> EvalResult<'tcx, Vec<OpTy<'tcx, M::PointerTag>>> {
        ops.into_iter()
            .map(|op| self.eval_operand(op, None))
            .collect()
    }

    // Used when miri runs into a constant, and by CTFE.
    // FIXME: CTFE should use allocations, then we can make this private (embed it into
    // `eval_operand`, ideally).
    pub(crate) fn const_value_to_op(
        &self,
        val: ty::LazyConst<'tcx>,
    ) -> EvalResult<'tcx, Operand<M::PointerTag>> {
        trace!("const_value_to_op: {:?}", val);
        let val = match val {
            ty::LazyConst::Unevaluated(def_id, substs) => {
                let instance = self.resolve(def_id, substs)?;
                return Ok(*OpTy::from(self.const_eval_raw(GlobalId {
                    instance,
                    promoted: None,
                })?));
            },
            ty::LazyConst::Evaluated(c) => c,
        };
        match val.val {
            ConstValue::ByRef(id, alloc, offset) => {
                // We rely on mutability being set correctly in that allocation to prevent writes
                // where none should happen -- and for `static mut`, we copy on demand anyway.
                Ok(Operand::Indirect(
                    MemPlace::from_ptr(Pointer::new(id, offset), alloc.align)
                ).with_default_tag())
            },
            ConstValue::Slice(a, b) =>
                Ok(Operand::Immediate(Immediate::ScalarPair(
                    a.into(),
                    Scalar::from_uint(b, self.tcx.data_layout.pointer_size).into(),
                )).with_default_tag()),
            ConstValue::Scalar(x) =>
                Ok(Operand::Immediate(Immediate::Scalar(x.into())).with_default_tag()),
        }
    }

    /// Read discriminant, return the runtime value as well as the variant index.
    pub fn read_discriminant(
        &self,
        rval: OpTy<'tcx, M::PointerTag>,
    ) -> EvalResult<'tcx, (u128, VariantIdx)> {
        trace!("read_discriminant_value {:#?}", rval.layout);

        match rval.layout.variants {
            layout::Variants::Single { index } => {
                let discr_val = rval.layout.ty.ty_adt_def().map_or(
                    index.as_u32() as u128,
                    |def| def.discriminant_for_variant(*self.tcx, index).val);
                return Ok((discr_val, index));
            }
            layout::Variants::Tagged { .. } |
            layout::Variants::NicheFilling { .. } => {},
        }
        // read raw discriminant value
        let discr_op = self.operand_field(rval, 0)?;
        let discr_val = self.read_immediate(discr_op)?;
        let raw_discr = discr_val.to_scalar_or_undef();
        trace!("discr value: {:?}", raw_discr);
        // post-process
        Ok(match rval.layout.variants {
            layout::Variants::Single { .. } => bug!(),
            layout::Variants::Tagged { .. } => {
                let bits_discr = match raw_discr.to_bits(discr_val.layout.size) {
                    Ok(raw_discr) => raw_discr,
                    Err(_) => return err!(InvalidDiscriminant(raw_discr.erase_tag())),
                };
                let real_discr = if discr_val.layout.ty.is_signed() {
                    let i = bits_discr as i128;
                    // going from layout tag type to typeck discriminant type
                    // requires first sign extending with the layout discriminant
                    let shift = 128 - discr_val.layout.size.bits();
                    let sexted = (i << shift) >> shift;
                    // and then zeroing with the typeck discriminant type
                    let discr_ty = rval.layout.ty
                        .ty_adt_def().expect("tagged layout corresponds to adt")
                        .repr
                        .discr_type();
                    let discr_ty = layout::Integer::from_attr(self, discr_ty);
                    let shift = 128 - discr_ty.size().bits();
                    let truncatee = sexted as u128;
                    (truncatee << shift) >> shift
                } else {
                    bits_discr
                };
                // Make sure we catch invalid discriminants
                let index = rval.layout.ty
                    .ty_adt_def()
                    .expect("tagged layout for non adt")
                    .discriminants(self.tcx.tcx)
                    .find(|(_, var)| var.val == real_discr)
                    .ok_or_else(|| EvalErrorKind::InvalidDiscriminant(raw_discr.erase_tag()))?;
                (real_discr, index.0)
            },
            layout::Variants::NicheFilling {
                dataful_variant,
                ref niche_variants,
                niche_start,
                ..
            } => {
                let variants_start = niche_variants.start().as_u32() as u128;
                let variants_end = niche_variants.end().as_u32() as u128;
                match raw_discr {
                    ScalarMaybeUndef::Scalar(Scalar::Ptr(ptr)) => {
                        // The niche must be just 0 (which an inbounds pointer value never is)
                        let ptr_valid = niche_start == 0 && variants_start == variants_end &&
                            self.memory.check_bounds_ptr(ptr, InboundsCheck::MaybeDead).is_ok();
                        if !ptr_valid {
                            return err!(InvalidDiscriminant(raw_discr.erase_tag()));
                        }
                        (dataful_variant.as_u32() as u128, dataful_variant)
                    },
                    ScalarMaybeUndef::Scalar(Scalar::Bits { bits: raw_discr, size }) => {
                        assert_eq!(size as u64, discr_val.layout.size.bytes());
                        let adjusted_discr = raw_discr.wrapping_sub(niche_start)
                            .wrapping_add(variants_start);
                        if variants_start <= adjusted_discr && adjusted_discr <= variants_end {
                            let index = adjusted_discr as usize;
                            assert_eq!(index as u128, adjusted_discr);
                            assert!(index < rval.layout.ty
                                .ty_adt_def()
                                .expect("tagged layout for non adt")
                                .variants.len());
                            (adjusted_discr, VariantIdx::from_usize(index))
                        } else {
                            (dataful_variant.as_u32() as u128, dataful_variant)
                        }
                    },
                    ScalarMaybeUndef::Undef =>
                        return err!(InvalidDiscriminant(ScalarMaybeUndef::Undef)),
                }
            }
        })
    }

}
