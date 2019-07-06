//! Functions concerning immediate values and operands, and reading from operands.
//! All high-level functions to read from memory work on operands as sources.

use std::convert::TryInto;

use rustc::{mir, ty};
use rustc::ty::layout::{
    self, Size, LayoutOf, TyLayout, HasDataLayout, IntegerExt, VariantIdx,
};

use rustc::mir::interpret::{
    GlobalId, AllocId,
    ConstValue, Pointer, Scalar,
    InterpResult, InterpError,
    sign_extend, truncate,
};
use super::{
    InterpCx, Machine,
    MemPlace, MPlaceTy, PlaceTy, Place,
};
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

impl<'tcx, Tag> Immediate<Tag> {
    #[inline]
    pub fn from_scalar(val: Scalar<Tag>) -> Self {
        Immediate::Scalar(ScalarMaybeUndef::Scalar(val))
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
    pub fn to_scalar(self) -> InterpResult<'tcx, Scalar<Tag>> {
        self.to_scalar_or_undef().not_undef()
    }

    #[inline]
    pub fn to_scalar_pair(self) -> InterpResult<'tcx, (Scalar<Tag>, Scalar<Tag>)> {
        match self {
            Immediate::Scalar(..) => bug!("Got a thin pointer where a scalar pair was expected"),
            Immediate::ScalarPair(a, b) => Ok((a.not_undef()?, b.not_undef()?))
        }
    }

    /// Converts the immediate into a pointer (or a pointer-sized integer).
    /// Throws away the second half of a ScalarPair!
    #[inline]
    pub fn to_scalar_ptr(self) -> InterpResult<'tcx, Scalar<Tag>> {
        match self {
            Immediate::Scalar(ptr) |
            Immediate::ScalarPair(ptr, _) => ptr.not_undef(),
        }
    }

    /// Converts the value into its metadata.
    /// Throws away the first half of a ScalarPair!
    #[inline]
    pub fn to_meta(self) -> InterpResult<'tcx, Option<Scalar<Tag>>> {
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
    pub imm: Immediate<Tag>,
    pub layout: TyLayout<'tcx>,
}

impl<'tcx, Tag> ::std::ops::Deref for ImmTy<'tcx, Tag> {
    type Target = Immediate<Tag>;
    #[inline(always)]
    fn deref(&self) -> &Immediate<Tag> {
        &self.imm
    }
}

/// An `Operand` is the result of computing a `mir::Operand`. It can be immediate,
/// or still in memory. The latter is an optimization, to delay reading that chunk of
/// memory and to avoid having to store arbitrary-sized data here.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub enum Operand<Tag=(), Id=AllocId> {
    Immediate(Immediate<Tag, Id>),
    Indirect(MemPlace<Tag, Id>),
}

impl<Tag> Operand<Tag> {
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
    op: Operand<Tag>,
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
            op: Operand::Immediate(val.imm),
            layout: val.layout
        }
    }
}

impl<'tcx, Tag: Copy> ImmTy<'tcx, Tag>
{
    #[inline]
    pub fn from_scalar(val: Scalar<Tag>, layout: TyLayout<'tcx>) -> Self {
        ImmTy { imm: Immediate::from_scalar(val), layout }
    }

    #[inline]
    pub fn to_bits(self) -> InterpResult<'tcx, u128> {
        self.to_scalar()?.to_bits(self.layout.size)
    }
}

// Use the existing layout if given (but sanity check in debug mode),
// or compute the layout.
#[inline(always)]
pub(super) fn from_known_layout<'tcx>(
    layout: Option<TyLayout<'tcx>>,
    compute: impl FnOnce() -> InterpResult<'tcx, TyLayout<'tcx>>
) -> InterpResult<'tcx, TyLayout<'tcx>> {
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

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Try reading an immediate in memory; this is interesting particularly for `ScalarPair`.
    /// Returns `None` if the layout does not permit loading this as a value.
    fn try_read_immediate_from_mplace(
        &self,
        mplace: MPlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, Option<ImmTy<'tcx, M::PointerTag>>> {
        if mplace.layout.is_unsized() {
            // Don't touch unsized
            return Ok(None);
        }
        let (ptr, ptr_align) = mplace.to_scalar_ptr_align();

        let ptr = match self.memory.check_ptr_access(ptr, mplace.layout.size, ptr_align)? {
            Some(ptr) => ptr,
            None => return Ok(Some(ImmTy { // zero-sized type
                imm: Immediate::Scalar(Scalar::zst().into()),
                layout: mplace.layout,
            })),
        };

        match mplace.layout.abi {
            layout::Abi::Scalar(..) => {
                let scalar = self.memory
                    .get(ptr.alloc_id)?
                    .read_scalar(self, ptr, mplace.layout.size)?;
                Ok(Some(ImmTy {
                    imm: Immediate::Scalar(scalar),
                    layout: mplace.layout,
                }))
            }
            layout::Abi::ScalarPair(ref a, ref b) => {
                // We checked `ptr_align` above, so all fields will have the alignment they need.
                // We would anyway check against `ptr_align.restrict_for_offset(b_offset)`,
                // which `ptr.offset(b_offset)` cannot possibly fail to satisfy.
                let (a, b) = (&a.value, &b.value);
                let (a_size, b_size) = (a.size(self), b.size(self));
                let a_ptr = ptr;
                let b_offset = a_size.align_to(b.align(self).abi);
                assert!(b_offset.bytes() > 0); // we later use the offset to tell apart the fields
                let b_ptr = ptr.offset(b_offset, self)?;
                let a_val = self.memory
                    .get(ptr.alloc_id)?
                    .read_scalar(self, a_ptr, a_size)?;
                let b_val = self.memory
                    .get(ptr.alloc_id)?
                    .read_scalar(self, b_ptr, b_size)?;
                Ok(Some(ImmTy {
                    imm: Immediate::ScalarPair(a_val, b_val),
                    layout: mplace.layout,
                }))
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
    ) -> InterpResult<'tcx, Result<ImmTy<'tcx, M::PointerTag>, MPlaceTy<'tcx, M::PointerTag>>> {
        Ok(match src.try_as_mplace() {
            Ok(mplace) => {
                if let Some(val) = self.try_read_immediate_from_mplace(mplace)? {
                    Ok(val)
                } else {
                    Err(mplace)
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
    ) -> InterpResult<'tcx, ImmTy<'tcx, M::PointerTag>> {
        if let Ok(imm) = self.try_read_immediate(op)? {
            Ok(imm)
        } else {
            bug!("primitive read failed for type: {:?}", op.layout.ty);
        }
    }

    /// Read a scalar from a place
    pub fn read_scalar(
        &self,
        op: OpTy<'tcx, M::PointerTag>
    ) -> InterpResult<'tcx, ScalarMaybeUndef<M::PointerTag>> {
        Ok(self.read_immediate(op)?.to_scalar_or_undef())
    }

    // Turn the MPlace into a string (must already be dereferenced!)
    pub fn read_str(
        &self,
        mplace: MPlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, &str> {
        let len = mplace.len(self)?;
        let bytes = self.memory.read_bytes(mplace.ptr, Size::from_bytes(len as u64))?;
        let str = ::std::str::from_utf8(bytes)
            .map_err(|err| InterpError::ValidationFailure(err.to_string()))?;
        Ok(str)
    }

    /// Projection functions
    pub fn operand_field(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
        field: u64,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
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
        let immediate = match *base {
            // the field covers the entire type
            _ if offset.bytes() == 0 && field_layout.size == op.layout.size => *base,
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
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
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
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
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
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        assert_ne!(local, mir::RETURN_PLACE);
        let layout = self.layout_of_local(frame, local, layout)?;
        let op = if layout.is_zst() {
            // Do not read from ZST, they might not be initialized
            Operand::Immediate(Immediate::Scalar(Scalar::zst().into()))
        } else {
            frame.locals[local].access()?
        };
        Ok(OpTy { op, layout })
    }

    /// Every place can be read from, so we can turn them into an operand
    #[inline(always)]
    pub fn place_to_op(
        &self,
        place: PlaceTy<'tcx, M::PointerTag>
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        let op = match *place {
            Place::Ptr(mplace) => {
                Operand::Indirect(mplace)
            }
            Place::Local { frame, local } =>
                *self.access_local(&self.stack[frame], local, None)?
        };
        Ok(OpTy { op, layout: place.layout })
    }

    // Evaluate a place with the goal of reading from it.  This lets us sometimes
    // avoid allocations.
    pub(super) fn eval_place_to_op(
        &self,
        mir_place: &mir::Place<'tcx>,
        layout: Option<TyLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        use rustc::mir::Place;
        use rustc::mir::PlaceBase;

        mir_place.iterate(|place_base, place_projection| {
            let mut op = match place_base {
                PlaceBase::Local(mir::RETURN_PLACE) => return err!(ReadFromReturnPointer),
                PlaceBase::Local(local) => {
                    // FIXME use place_projection.is_empty() when is available
                    // Do not use the layout passed in as argument if the base we are looking at
                    // here is not the entire place.
                    let layout = if let Place::Base(_) = mir_place {
                        layout
                    } else {
                        None
                    };

                    self.access_local(self.frame(), *local, layout)?
                }
                PlaceBase::Static(place_static) => {
                    self.eval_static_to_mplace(place_static)?.into()
                }
            };

            for proj in place_projection {
                op = self.operand_projection(op, &proj.elem)?
            }

            trace!("eval_place_to_op: got {:?}", *op);
            Ok(op)
        })
    }

    /// Evaluate the operand, returning a place where you can then find the data.
    /// If you already know the layout, you can save two table lookups
    /// by passing it in here.
    pub fn eval_operand(
        &self,
        mir_op: &mir::Operand<'tcx>,
        layout: Option<TyLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        use rustc::mir::Operand::*;
        let op = match *mir_op {
            // FIXME: do some more logic on `move` to invalidate the old location
            Copy(ref place) |
            Move(ref place) =>
                self.eval_place_to_op(place, layout)?,

            Constant(ref constant) => self.eval_const_to_op(constant.literal, layout)?,
        };
        trace!("{:?}: {:?}", mir_op, *op);
        Ok(op)
    }

    /// Evaluate a bunch of operands at once
    pub(super) fn eval_operands(
        &self,
        ops: &[mir::Operand<'tcx>],
    ) -> InterpResult<'tcx, Vec<OpTy<'tcx, M::PointerTag>>> {
        ops.into_iter()
            .map(|op| self.eval_operand(op, None))
            .collect()
    }

    // Used when the miri-engine runs into a constant and for extracting information from constants
    // in patterns via the `const_eval` module
    crate fn eval_const_to_op(
        &self,
        val: &'tcx ty::Const<'tcx>,
        layout: Option<TyLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        let tag_scalar = |scalar| match scalar {
            Scalar::Ptr(ptr) => Scalar::Ptr(self.tag_static_base_pointer(ptr)),
            Scalar::Raw { data, size } => Scalar::Raw { data, size },
        };
        // Early-return cases.
        match val.val {
            ConstValue::Param(_) => return err!(TooGeneric), // FIXME(oli-obk): try to monomorphize
            ConstValue::Unevaluated(def_id, substs) => {
                let instance = self.resolve(def_id, substs)?;
                return Ok(OpTy::from(self.const_eval_raw(GlobalId {
                    instance,
                    promoted: None,
                })?));
            }
            _ => {}
        }
        // Other cases need layout.
        let layout = from_known_layout(layout, || {
            self.layout_of(self.monomorphize(val.ty)?)
        })?;
        let op = match val.val {
            ConstValue::ByRef { offset, align, alloc } => {
                let id = self.tcx.alloc_map.lock().create_memory_alloc(alloc);
                // We rely on mutability being set correctly in that allocation to prevent writes
                // where none should happen.
                let ptr = self.tag_static_base_pointer(Pointer::new(id, offset));
                Operand::Indirect(MemPlace::from_ptr(ptr, align))
            },
            ConstValue::Scalar(x) =>
                Operand::Immediate(Immediate::Scalar(tag_scalar(x).into())),
            ConstValue::Slice { data, start, end } => {
                // We rely on mutability being set correctly in `data` to prevent writes
                // where none should happen.
                let ptr = Pointer::new(
                    self.tcx.alloc_map.lock().create_memory_alloc(data),
                    Size::from_bytes(start as u64), // offset: `start`
                );
                Operand::Immediate(Immediate::new_slice(
                    self.tag_static_base_pointer(ptr).into(),
                    (end - start) as u64, // len: `end - start`
                    self,
                ))
            }
            ConstValue::Param(..) |
            ConstValue::Infer(..) |
            ConstValue::Placeholder(..) |
            ConstValue::Unevaluated(..) =>
                bug!("eval_const_to_op: Unexpected ConstValue {:?}", val),
        };
        Ok(OpTy { op, layout })
    }

    /// Read discriminant, return the runtime value as well as the variant index.
    pub fn read_discriminant(
        &self,
        rval: OpTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, (u128, VariantIdx)> {
        trace!("read_discriminant_value {:#?}", rval.layout);

        let (discr_kind, discr_index) = match rval.layout.variants {
            layout::Variants::Single { index } => {
                let discr_val = rval.layout.ty.discriminant_for_variant(*self.tcx, index).map_or(
                    index.as_u32() as u128,
                    |discr| discr.val);
                return Ok((discr_val, index));
            }
            layout::Variants::Multiple { ref discr_kind, discr_index, .. } =>
                (discr_kind, discr_index),
        };

        // read raw discriminant value
        let discr_op = self.operand_field(rval, discr_index as u64)?;
        let discr_val = self.read_immediate(discr_op)?;
        let raw_discr = discr_val.to_scalar_or_undef();
        trace!("discr value: {:?}", raw_discr);
        // post-process
        Ok(match *discr_kind {
            layout::DiscriminantKind::Tag => {
                let bits_discr = match raw_discr.to_bits(discr_val.layout.size) {
                    Ok(raw_discr) => raw_discr,
                    Err(_) => return err!(InvalidDiscriminant(raw_discr.erase_tag())),
                };
                let real_discr = if discr_val.layout.ty.is_signed() {
                    // going from layout tag type to typeck discriminant type
                    // requires first sign extending with the layout discriminant
                    let sexted = sign_extend(bits_discr, discr_val.layout.size) as i128;
                    // and then zeroing with the typeck discriminant type
                    let discr_ty = rval.layout.ty
                        .ty_adt_def().expect("tagged layout corresponds to adt")
                        .repr
                        .discr_type();
                    let size = layout::Integer::from_attr(self, discr_ty).size();
                    let truncatee = sexted as u128;
                    truncate(truncatee, size)
                } else {
                    bits_discr
                };
                // Make sure we catch invalid discriminants
                let index = match &rval.layout.ty.sty {
                    ty::Adt(adt, _) => adt
                        .discriminants(self.tcx.tcx)
                        .find(|(_, var)| var.val == real_discr),
                    ty::Generator(def_id, substs, _) => substs
                        .discriminants(*def_id, self.tcx.tcx)
                        .find(|(_, var)| var.val == real_discr),
                    _ => bug!("tagged layout for non-adt non-generator"),
                }.ok_or_else(|| InterpError::InvalidDiscriminant(raw_discr.erase_tag()))?;
                (real_discr, index.0)
            },
            layout::DiscriminantKind::Niche {
                dataful_variant,
                ref niche_variants,
                niche_start,
            } => {
                let variants_start = niche_variants.start().as_u32() as u128;
                let variants_end = niche_variants.end().as_u32() as u128;
                let raw_discr = raw_discr.not_undef()
                    .map_err(|_| InterpError::InvalidDiscriminant(ScalarMaybeUndef::Undef))?;
                match raw_discr.to_bits_or_ptr(discr_val.layout.size, self) {
                    Err(ptr) => {
                        // The niche must be just 0 (which an inbounds pointer value never is)
                        let ptr_valid = niche_start == 0 && variants_start == variants_end &&
                            !self.memory.ptr_may_be_null(ptr);
                        if !ptr_valid {
                            return err!(InvalidDiscriminant(raw_discr.erase_tag().into()));
                        }
                        (dataful_variant.as_u32() as u128, dataful_variant)
                    },
                    Ok(raw_discr) => {
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
                }
            }
        })
    }
}
