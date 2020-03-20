//! Functions concerning immediate values and operands, and reading from operands.
//! All high-level functions to read from memory work on operands as sources.

use std::convert::{TryFrom, TryInto};

use super::{InterpCx, MPlaceTy, Machine, MemPlace, Place, PlaceTy};
pub use rustc::mir::interpret::ScalarMaybeUndef;
use rustc::mir::interpret::{
    sign_extend, truncate, AllocId, ConstValue, GlobalId, InterpResult, Pointer, Scalar,
};
use rustc::ty::layout::{
    self, HasDataLayout, IntegerExt, LayoutOf, PrimitiveExt, Size, TyLayout, VariantIdx,
};
use rustc::ty::print::{FmtPrinter, PrettyPrinter, Printer};
use rustc::ty::Ty;
use rustc::{mir, ty};
use rustc_hir::def::Namespace;
use rustc_macros::HashStable;
use std::fmt::Write;

/// An `Immediate` represents a single immediate self-contained Rust value.
///
/// For optimization of a few very common cases, there is also a representation for a pair of
/// primitive values (`ScalarPair`). It allows Miri to avoid making allocations for checked binary
/// operations and wide pointers. This idea was taken from rustc's codegen.
/// In particular, thanks to `ScalarPair`, arithmetic operations and casts can be entirely
/// defined on `Immediate`, and do not have to work with a `Place`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, HashStable, Hash)]
pub enum Immediate<Tag = (), Id = AllocId> {
    Scalar(ScalarMaybeUndef<Tag, Id>),
    ScalarPair(ScalarMaybeUndef<Tag, Id>, ScalarMaybeUndef<Tag, Id>),
}

impl<Tag> From<ScalarMaybeUndef<Tag>> for Immediate<Tag> {
    #[inline(always)]
    fn from(val: ScalarMaybeUndef<Tag>) -> Self {
        Immediate::Scalar(val)
    }
}

impl<Tag> From<Scalar<Tag>> for Immediate<Tag> {
    #[inline(always)]
    fn from(val: Scalar<Tag>) -> Self {
        Immediate::Scalar(val.into())
    }
}

impl<Tag> From<Pointer<Tag>> for Immediate<Tag> {
    #[inline(always)]
    fn from(val: Pointer<Tag>) -> Self {
        Immediate::Scalar(Scalar::from(val).into())
    }
}

impl<'tcx, Tag> Immediate<Tag> {
    pub fn new_slice(val: Scalar<Tag>, len: u64, cx: &impl HasDataLayout) -> Self {
        Immediate::ScalarPair(
            val.into(),
            Scalar::from_uint(len, cx.data_layout().pointer_size).into(),
        )
    }

    pub fn new_dyn_trait(val: Scalar<Tag>, vtable: Pointer<Tag>) -> Self {
        Immediate::ScalarPair(val.into(), vtable.into())
    }

    #[inline]
    pub fn to_scalar_or_undef(self) -> ScalarMaybeUndef<Tag> {
        match self {
            Immediate::Scalar(val) => val,
            Immediate::ScalarPair(..) => bug!("Got a wide pointer where a scalar was expected"),
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
            Immediate::ScalarPair(a, b) => Ok((a.not_undef()?, b.not_undef()?)),
        }
    }
}

// ScalarPair needs a type to interpret, so we often have an immediate and a type together
// as input for binary and cast operations.
#[derive(Copy, Clone, Debug)]
pub struct ImmTy<'tcx, Tag = ()> {
    pub(crate) imm: Immediate<Tag>,
    pub layout: TyLayout<'tcx>,
}

impl<Tag: Copy> std::fmt::Display for ImmTy<'tcx, Tag> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// Helper function for printing a scalar to a FmtPrinter
        fn p<'a, 'tcx, F: std::fmt::Write, Tag>(
            cx: FmtPrinter<'a, 'tcx, F>,
            s: ScalarMaybeUndef<Tag>,
            ty: Ty<'tcx>,
        ) -> Result<FmtPrinter<'a, 'tcx, F>, std::fmt::Error> {
            match s {
                ScalarMaybeUndef::Scalar(s) => {
                    cx.pretty_print_const_scalar(s.erase_tag(), ty, true)
                }
                ScalarMaybeUndef::Undef => cx.typed_value(
                    |mut this| {
                        this.write_str("{undef ")?;
                        Ok(this)
                    },
                    |this| this.print_type(ty),
                    " ",
                ),
            }
        }
        ty::tls::with(|tcx| {
            match self.imm {
                Immediate::Scalar(s) => {
                    if let Some(ty) = tcx.lift(&self.layout.ty) {
                        let cx = FmtPrinter::new(tcx, f, Namespace::ValueNS);
                        p(cx, s, ty)?;
                        return Ok(());
                    }
                    write!(f, "{:?}: {}", s.erase_tag(), self.layout.ty)
                }
                Immediate::ScalarPair(a, b) => {
                    // FIXME(oli-obk): at least print tuples and slices nicely
                    write!(f, "({:?}, {:?}): {}", a.erase_tag(), b.erase_tag(), self.layout.ty,)
                }
            }
        })
    }
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
#[derive(Copy, Clone, Debug, PartialEq, Eq, HashStable, Hash)]
pub enum Operand<Tag = (), Id = AllocId> {
    Immediate(Immediate<Tag, Id>),
    Indirect(MemPlace<Tag, Id>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpTy<'tcx, Tag = ()> {
    op: Operand<Tag>, // Keep this private; it helps enforce invariants.
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
        OpTy { op: Operand::Indirect(*mplace), layout: mplace.layout }
    }
}

impl<'tcx, Tag> From<ImmTy<'tcx, Tag>> for OpTy<'tcx, Tag> {
    #[inline(always)]
    fn from(val: ImmTy<'tcx, Tag>) -> Self {
        OpTy { op: Operand::Immediate(val.imm), layout: val.layout }
    }
}

impl<'tcx, Tag: Copy> ImmTy<'tcx, Tag> {
    #[inline]
    pub fn from_scalar(val: Scalar<Tag>, layout: TyLayout<'tcx>) -> Self {
        ImmTy { imm: val.into(), layout }
    }

    #[inline]
    pub fn try_from_uint(i: impl Into<u128>, layout: TyLayout<'tcx>) -> Option<Self> {
        Some(Self::from_scalar(Scalar::try_from_uint(i, layout.size)?, layout))
    }
    #[inline]
    pub fn from_uint(i: impl Into<u128>, layout: TyLayout<'tcx>) -> Self {
        Self::from_scalar(Scalar::from_uint(i, layout.size), layout)
    }

    #[inline]
    pub fn try_from_int(i: impl Into<i128>, layout: TyLayout<'tcx>) -> Option<Self> {
        Some(Self::from_scalar(Scalar::try_from_int(i, layout.size)?, layout))
    }

    #[inline]
    pub fn from_int(i: impl Into<i128>, layout: TyLayout<'tcx>) -> Self {
        Self::from_scalar(Scalar::from_int(i, layout.size), layout)
    }
}

// Use the existing layout if given (but sanity check in debug mode),
// or compute the layout.
#[inline(always)]
pub(super) fn from_known_layout<'tcx>(
    layout: Option<TyLayout<'tcx>>,
    compute: impl FnOnce() -> InterpResult<'tcx, TyLayout<'tcx>>,
) -> InterpResult<'tcx, TyLayout<'tcx>> {
    match layout {
        None => compute(),
        Some(layout) => {
            if cfg!(debug_assertions) {
                let layout2 = compute()?;
                assert_eq!(
                    layout.details, layout2.details,
                    "mismatch in layout of supposedly equal-layout types {:?} and {:?}",
                    layout.ty, layout2.ty
                );
            }
            Ok(layout)
        }
    }
}

impl<'mir, 'tcx, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
    /// Normalice `place.ptr` to a `Pointer` if this is a place and not a ZST.
    /// Can be helpful to avoid lots of `force_ptr` calls later, if this place is used a lot.
    #[inline]
    pub fn force_op_ptr(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        match op.try_as_mplace(self) {
            Ok(mplace) => Ok(self.force_mplace_ptr(mplace)?.into()),
            Err(imm) => Ok(imm.into()), // Nothing to cast/force
        }
    }

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

        let ptr = match self
            .check_mplace_access(mplace, None)
            .expect("places should be checked on creation")
        {
            Some(ptr) => ptr,
            None => {
                return Ok(Some(ImmTy {
                    // zero-sized type
                    imm: Scalar::zst().into(),
                    layout: mplace.layout,
                }));
            }
        };

        match mplace.layout.abi {
            layout::Abi::Scalar(..) => {
                let scalar = self.memory.get_raw(ptr.alloc_id)?.read_scalar(
                    self,
                    ptr,
                    mplace.layout.size,
                )?;
                Ok(Some(ImmTy { imm: scalar.into(), layout: mplace.layout }))
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
                let a_val = self.memory.get_raw(ptr.alloc_id)?.read_scalar(self, a_ptr, a_size)?;
                let b_val = self.memory.get_raw(ptr.alloc_id)?.read_scalar(self, b_ptr, b_size)?;
                Ok(Some(ImmTy { imm: Immediate::ScalarPair(a_val, b_val), layout: mplace.layout }))
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
        Ok(match src.try_as_mplace(self) {
            Ok(mplace) => {
                if let Some(val) = self.try_read_immediate_from_mplace(mplace)? {
                    Ok(val)
                } else {
                    Err(mplace)
                }
            }
            Err(val) => Ok(val),
        })
    }

    /// Read an immediate from a place, asserting that that is possible with the given layout.
    #[inline(always)]
    pub fn read_immediate(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
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
        op: OpTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, ScalarMaybeUndef<M::PointerTag>> {
        Ok(self.read_immediate(op)?.to_scalar_or_undef())
    }

    // Turn the wide MPlace into a string (must already be dereferenced!)
    pub fn read_str(&self, mplace: MPlaceTy<'tcx, M::PointerTag>) -> InterpResult<'tcx, &str> {
        let len = mplace.len(self)?;
        let bytes = self.memory.read_bytes(mplace.ptr, Size::from_bytes(len as u64))?;
        let str = ::std::str::from_utf8(bytes)
            .map_err(|err| err_ub_format!("this string is not valid UTF-8: {}", err))?;
        Ok(str)
    }

    /// Projection functions
    pub fn operand_field(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
        field: u64,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        let base = match op.try_as_mplace(self) {
            Ok(mplace) => {
                // We can reuse the mplace field computation logic for indirect operands.
                let field = self.mplace_field(mplace, field)?;
                return Ok(field.into());
            }
            Err(value) => value,
        };

        let field = field.try_into().unwrap();
        let field_layout = op.layout.field(self, field)?;
        if field_layout.is_zst() {
            let immediate = Scalar::zst().into();
            return Ok(OpTy { op: Operand::Immediate(immediate), layout: field_layout });
        }
        let offset = op.layout.fields.offset(field);
        let immediate = match *base {
            // the field covers the entire type
            _ if offset.bytes() == 0 && field_layout.size == op.layout.size => *base,
            // extract fields from types with `ScalarPair` ABI
            Immediate::ScalarPair(a, b) => {
                let val = if offset.bytes() == 0 { a } else { b };
                Immediate::from(val)
            }
            Immediate::Scalar(val) => {
                bug!("field access on non aggregate {:#?}, {:#?}", val, op.layout)
            }
        };
        Ok(OpTy { op: Operand::Immediate(immediate), layout: field_layout })
    }

    pub fn operand_downcast(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
        variant: VariantIdx,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        // Downcasts only change the layout
        Ok(match op.try_as_mplace(self) {
            Ok(mplace) => self.mplace_downcast(mplace, variant)?.into(),
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
            Subslice { .. } | ConstantIndex { .. } | Index(_) => {
                // The rest should only occur as mplace, we do not use Immediates for types
                // allowing such operations.  This matches place_projection forcing an allocation.
                let mplace = base.assert_mem_place(self);
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
            Operand::Immediate(Scalar::zst().into())
        } else {
            M::access_local(&self, frame, local)?
        };
        Ok(OpTy { op, layout })
    }

    /// Every place can be read from, so we can turn them into an operand
    #[inline(always)]
    pub fn place_to_op(
        &self,
        place: PlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        let op = match *place {
            Place::Ptr(mplace) => Operand::Indirect(mplace),
            Place::Local { frame, local } => *self.access_local(&self.stack[frame], local, None)?,
        };
        Ok(OpTy { op, layout: place.layout })
    }

    // Evaluate a place with the goal of reading from it.  This lets us sometimes
    // avoid allocations.
    pub fn eval_place_to_op(
        &self,
        place: &mir::Place<'tcx>,
        layout: Option<TyLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        let base_op = match place.local {
            mir::RETURN_PLACE => throw_ub!(ReadFromReturnPlace),
            local => {
                // Do not use the layout passed in as argument if the base we are looking at
                // here is not the entire place.
                let layout = if place.projection.is_empty() { layout } else { None };

                self.access_local(self.frame(), local, layout)?
            }
        };

        let op = place
            .projection
            .iter()
            .try_fold(base_op, |op, elem| self.operand_projection(op, elem))?;

        trace!("eval_place_to_op: got {:?}", *op);
        Ok(op)
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
            Copy(ref place) | Move(ref place) => self.eval_place_to_op(place, layout)?,

            Constant(ref constant) => {
                let val =
                    self.subst_from_current_frame_and_normalize_erasing_regions(constant.literal);
                self.eval_const_to_op(val, layout)?
            }
        };
        trace!("{:?}: {:?}", mir_op, *op);
        Ok(op)
    }

    /// Evaluate a bunch of operands at once
    pub(super) fn eval_operands(
        &self,
        ops: &[mir::Operand<'tcx>],
    ) -> InterpResult<'tcx, Vec<OpTy<'tcx, M::PointerTag>>> {
        ops.iter().map(|op| self.eval_operand(op, None)).collect()
    }

    // Used when the miri-engine runs into a constant and for extracting information from constants
    // in patterns via the `const_eval` module
    /// The `val` and `layout` are assumed to already be in our interpreter
    /// "universe" (param_env).
    crate fn eval_const_to_op(
        &self,
        val: &ty::Const<'tcx>,
        layout: Option<TyLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        let tag_scalar = |scalar| match scalar {
            Scalar::Ptr(ptr) => Scalar::Ptr(self.tag_static_base_pointer(ptr)),
            Scalar::Raw { data, size } => Scalar::Raw { data, size },
        };
        // Early-return cases.
        let val_val = match val.val {
            ty::ConstKind::Param(_) => throw_inval!(TooGeneric),
            ty::ConstKind::Unevaluated(def_id, substs, promoted) => {
                let instance = self.resolve(def_id, substs)?;
                // We use `const_eval` here and `const_eval_raw` elsewhere in mir interpretation.
                // The reason we use `const_eval_raw` everywhere else is to prevent cycles during
                // validation, because validation automatically reads through any references, thus
                // potentially requiring the current static to be evaluated again. This is not a
                // problem here, because we are building an operand which means an actual read is
                // happening.
                return Ok(self.const_eval(GlobalId { instance, promoted }, val.ty)?);
            }
            ty::ConstKind::Infer(..)
            | ty::ConstKind::Bound(..)
            | ty::ConstKind::Placeholder(..) => {
                bug!("eval_const_to_op: Unexpected ConstKind {:?}", val)
            }
            ty::ConstKind::Value(val_val) => val_val,
        };
        // Other cases need layout.
        let layout = from_known_layout(layout, || self.layout_of(val.ty))?;
        let op = match val_val {
            ConstValue::ByRef { alloc, offset } => {
                let id = self.tcx.alloc_map.lock().create_memory_alloc(alloc);
                // We rely on mutability being set correctly in that allocation to prevent writes
                // where none should happen.
                let ptr = self.tag_static_base_pointer(Pointer::new(id, offset));
                Operand::Indirect(MemPlace::from_ptr(ptr, layout.align.abi))
            }
            ConstValue::Scalar(x) => Operand::Immediate(tag_scalar(x).into()),
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
        };
        Ok(OpTy { op, layout })
    }

    /// Read discriminant, return the runtime value as well as the variant index.
    pub fn read_discriminant(
        &self,
        rval: OpTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, (u128, VariantIdx)> {
        trace!("read_discriminant_value {:#?}", rval.layout);

        let (discr_layout, discr_kind, discr_index) = match rval.layout.variants {
            layout::Variants::Single { index } => {
                let discr_val = rval
                    .layout
                    .ty
                    .discriminant_for_variant(*self.tcx, index)
                    .map_or(index.as_u32() as u128, |discr| discr.val);
                return Ok((discr_val, index));
            }
            layout::Variants::Multiple {
                discr: ref discr_layout,
                ref discr_kind,
                discr_index,
                ..
            } => (discr_layout, discr_kind, discr_index),
        };

        // read raw discriminant value
        let discr_op = self.operand_field(rval, discr_index as u64)?;
        let discr_val = self.read_immediate(discr_op)?;
        let raw_discr = discr_val.to_scalar_or_undef();
        trace!("discr value: {:?}", raw_discr);
        // post-process
        Ok(match *discr_kind {
            layout::DiscriminantKind::Tag => {
                let bits_discr = raw_discr
                    .not_undef()
                    .and_then(|raw_discr| self.force_bits(raw_discr, discr_val.layout.size))
                    .map_err(|_| err_ub!(InvalidDiscriminant(raw_discr.erase_tag())))?;
                let real_discr = if discr_val.layout.ty.is_signed() {
                    // going from layout tag type to typeck discriminant type
                    // requires first sign extending with the discriminant layout
                    let sexted = sign_extend(bits_discr, discr_val.layout.size) as i128;
                    // and then zeroing with the typeck discriminant type
                    let discr_ty = rval
                        .layout
                        .ty
                        .ty_adt_def()
                        .expect("tagged layout corresponds to adt")
                        .repr
                        .discr_type();
                    let size = layout::Integer::from_attr(self, discr_ty).size();
                    let truncatee = sexted as u128;
                    truncate(truncatee, size)
                } else {
                    bits_discr
                };
                // Make sure we catch invalid discriminants
                let index = match rval.layout.ty.kind {
                    ty::Adt(adt, _) => {
                        adt.discriminants(self.tcx.tcx).find(|(_, var)| var.val == real_discr)
                    }
                    ty::Generator(def_id, substs, _) => {
                        let substs = substs.as_generator();
                        substs
                            .discriminants(def_id, self.tcx.tcx)
                            .find(|(_, var)| var.val == real_discr)
                    }
                    _ => bug!("tagged layout for non-adt non-generator"),
                }
                .ok_or_else(|| err_ub!(InvalidDiscriminant(raw_discr.erase_tag())))?;
                (real_discr, index.0)
            }
            layout::DiscriminantKind::Niche {
                dataful_variant,
                ref niche_variants,
                niche_start,
            } => {
                let variants_start = niche_variants.start().as_u32();
                let variants_end = niche_variants.end().as_u32();
                let raw_discr = raw_discr
                    .not_undef()
                    .map_err(|_| err_ub!(InvalidDiscriminant(ScalarMaybeUndef::Undef)))?;
                match raw_discr.to_bits_or_ptr(discr_val.layout.size, self) {
                    Err(ptr) => {
                        // The niche must be just 0 (which an inbounds pointer value never is)
                        let ptr_valid = niche_start == 0
                            && variants_start == variants_end
                            && !self.memory.ptr_may_be_null(ptr);
                        if !ptr_valid {
                            throw_ub!(InvalidDiscriminant(raw_discr.erase_tag().into()))
                        }
                        (dataful_variant.as_u32() as u128, dataful_variant)
                    }
                    Ok(raw_discr) => {
                        // We need to use machine arithmetic to get the relative variant idx:
                        // variant_index_relative = discr_val - niche_start_val
                        let discr_layout =
                            self.layout_of(discr_layout.value.to_int_ty(*self.tcx))?;
                        let discr_val = ImmTy::from_uint(raw_discr, discr_layout);
                        let niche_start_val = ImmTy::from_uint(niche_start, discr_layout);
                        let variant_index_relative_val =
                            self.binary_op(mir::BinOp::Sub, discr_val, niche_start_val)?;
                        let variant_index_relative = variant_index_relative_val
                            .to_scalar()?
                            .assert_bits(discr_val.layout.size);
                        // Check if this is in the range that indicates an actual discriminant.
                        if variant_index_relative <= u128::from(variants_end - variants_start) {
                            let variant_index_relative = u32::try_from(variant_index_relative)
                                .expect("we checked that this fits into a u32");
                            // Then computing the absolute variant idx should not overflow any more.
                            let variant_index = variants_start
                                .checked_add(variant_index_relative)
                                .expect("overflow computing absolute variant idx");
                            let variants_len = rval
                                .layout
                                .ty
                                .ty_adt_def()
                                .expect("tagged layout for non adt")
                                .variants
                                .len();
                            assert!((variant_index as usize) < variants_len);
                            (u128::from(variant_index), VariantIdx::from_u32(variant_index))
                        } else {
                            (u128::from(dataful_variant.as_u32()), dataful_variant)
                        }
                    }
                }
            }
        })
    }
}
