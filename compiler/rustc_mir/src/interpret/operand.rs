//! Functions concerning immediate values and operands, and reading from operands.
//! All high-level functions to read from memory work on operands as sources.

use std::convert::TryFrom;
use std::fmt::Write;

use rustc_errors::ErrorReported;
use rustc_hir::def::Namespace;
use rustc_macros::HashStable;
use rustc_middle::ty::layout::{PrimitiveExt, TyAndLayout};
use rustc_middle::ty::print::{FmtPrinter, PrettyPrinter, Printer};
use rustc_middle::ty::{ConstInt, Ty};
use rustc_middle::{mir, ty};
use rustc_target::abi::{Abi, HasDataLayout, LayoutOf, Size, TagEncoding};
use rustc_target::abi::{VariantIdx, Variants};

use super::{
    from_known_layout, mir_assign_valid_types, ConstValue, GlobalId, InterpCx, InterpResult,
    MPlaceTy, Machine, MemPlace, Place, PlaceTy, Pointer, Scalar, ScalarMaybeUninit,
};

/// An `Immediate` represents a single immediate self-contained Rust value.
///
/// For optimization of a few very common cases, there is also a representation for a pair of
/// primitive values (`ScalarPair`). It allows Miri to avoid making allocations for checked binary
/// operations and wide pointers. This idea was taken from rustc's codegen.
/// In particular, thanks to `ScalarPair`, arithmetic operations and casts can be entirely
/// defined on `Immediate`, and do not have to work with a `Place`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, HashStable, Hash)]
pub enum Immediate<Tag = ()> {
    Scalar(ScalarMaybeUninit<Tag>),
    ScalarPair(ScalarMaybeUninit<Tag>, ScalarMaybeUninit<Tag>),
}

impl<Tag> From<ScalarMaybeUninit<Tag>> for Immediate<Tag> {
    #[inline(always)]
    fn from(val: ScalarMaybeUninit<Tag>) -> Self {
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
        Immediate::ScalarPair(val.into(), Scalar::from_machine_usize(len, cx).into())
    }

    pub fn new_dyn_trait(val: Scalar<Tag>, vtable: Pointer<Tag>) -> Self {
        Immediate::ScalarPair(val.into(), vtable.into())
    }

    #[inline]
    pub fn to_scalar_or_uninit(self) -> ScalarMaybeUninit<Tag> {
        match self {
            Immediate::Scalar(val) => val,
            Immediate::ScalarPair(..) => bug!("Got a wide pointer where a scalar was expected"),
        }
    }

    #[inline]
    pub fn to_scalar(self) -> InterpResult<'tcx, Scalar<Tag>> {
        self.to_scalar_or_uninit().check_init()
    }

    #[inline]
    pub fn to_scalar_pair(self) -> InterpResult<'tcx, (Scalar<Tag>, Scalar<Tag>)> {
        match self {
            Immediate::Scalar(..) => bug!("Got a thin pointer where a scalar pair was expected"),
            Immediate::ScalarPair(a, b) => Ok((a.check_init()?, b.check_init()?)),
        }
    }
}

// ScalarPair needs a type to interpret, so we often have an immediate and a type together
// as input for binary and cast operations.
#[derive(Copy, Clone, Debug)]
pub struct ImmTy<'tcx, Tag = ()> {
    imm: Immediate<Tag>,
    pub layout: TyAndLayout<'tcx>,
}

impl<Tag: Copy> std::fmt::Display for ImmTy<'tcx, Tag> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        /// Helper function for printing a scalar to a FmtPrinter
        fn p<'a, 'tcx, F: std::fmt::Write, Tag>(
            cx: FmtPrinter<'a, 'tcx, F>,
            s: ScalarMaybeUninit<Tag>,
            ty: Ty<'tcx>,
        ) -> Result<FmtPrinter<'a, 'tcx, F>, std::fmt::Error> {
            match s {
                ScalarMaybeUninit::Scalar(s) => {
                    cx.pretty_print_const_scalar(s.erase_tag(), ty, true)
                }
                ScalarMaybeUninit::Uninit => cx.typed_value(
                    |mut this| {
                        this.write_str("{uninit ")?;
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
                    if let Some(ty) = tcx.lift(self.layout.ty) {
                        let cx = FmtPrinter::new(tcx, f, Namespace::ValueNS);
                        p(cx, s, ty)?;
                        return Ok(());
                    }
                    write!(f, "{}: {}", s.erase_tag(), self.layout.ty)
                }
                Immediate::ScalarPair(a, b) => {
                    // FIXME(oli-obk): at least print tuples and slices nicely
                    write!(f, "({}, {}): {}", a.erase_tag(), b.erase_tag(), self.layout.ty,)
                }
            }
        })
    }
}

impl<'tcx, Tag> std::ops::Deref for ImmTy<'tcx, Tag> {
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
pub enum Operand<Tag = ()> {
    Immediate(Immediate<Tag>),
    Indirect(MemPlace<Tag>),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct OpTy<'tcx, Tag = ()> {
    op: Operand<Tag>, // Keep this private; it helps enforce invariants.
    pub layout: TyAndLayout<'tcx>,
}

impl<'tcx, Tag> std::ops::Deref for OpTy<'tcx, Tag> {
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
    pub fn from_scalar(val: Scalar<Tag>, layout: TyAndLayout<'tcx>) -> Self {
        ImmTy { imm: val.into(), layout }
    }

    #[inline]
    pub fn from_immediate(imm: Immediate<Tag>, layout: TyAndLayout<'tcx>) -> Self {
        ImmTy { imm, layout }
    }

    #[inline]
    pub fn try_from_uint(i: impl Into<u128>, layout: TyAndLayout<'tcx>) -> Option<Self> {
        Some(Self::from_scalar(Scalar::try_from_uint(i, layout.size)?, layout))
    }
    #[inline]
    pub fn from_uint(i: impl Into<u128>, layout: TyAndLayout<'tcx>) -> Self {
        Self::from_scalar(Scalar::from_uint(i, layout.size), layout)
    }

    #[inline]
    pub fn try_from_int(i: impl Into<i128>, layout: TyAndLayout<'tcx>) -> Option<Self> {
        Some(Self::from_scalar(Scalar::try_from_int(i, layout.size)?, layout))
    }

    #[inline]
    pub fn from_int(i: impl Into<i128>, layout: TyAndLayout<'tcx>) -> Self {
        Self::from_scalar(Scalar::from_int(i, layout.size), layout)
    }

    #[inline]
    pub fn to_const_int(self) -> ConstInt {
        assert!(self.layout.ty.is_integral());
        ConstInt::new(
            self.to_scalar()
                .expect("to_const_int doesn't work on scalar pairs")
                .assert_bits(self.layout.size),
            self.layout.size,
            self.layout.ty.is_signed(),
            self.layout.ty.is_ptr_sized_integral(),
        )
    }
}

impl<'mir, 'tcx: 'mir, M: Machine<'mir, 'tcx>> InterpCx<'mir, 'tcx, M> {
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
                if let Scalar::Ptr(ptr) = mplace.ptr {
                    // We may be reading from a static.
                    // In order to ensure that `static FOO: Type = FOO;` causes a cycle error
                    // instead of magically pulling *any* ZST value from the ether, we need to
                    // actually access the referenced allocation.
                    self.memory.get_raw(ptr.alloc_id)?;
                }
                return Ok(Some(ImmTy {
                    // zero-sized type
                    imm: Scalar::zst().into(),
                    layout: mplace.layout,
                }));
            }
        };

        let alloc = self.memory.get_raw(ptr.alloc_id)?;

        match mplace.layout.abi {
            Abi::Scalar(..) => {
                let scalar = alloc.read_scalar(self, ptr, mplace.layout.size)?;
                Ok(Some(ImmTy { imm: scalar.into(), layout: mplace.layout }))
            }
            Abi::ScalarPair(ref a, ref b) => {
                // We checked `ptr_align` above, so all fields will have the alignment they need.
                // We would anyway check against `ptr_align.restrict_for_offset(b_offset)`,
                // which `ptr.offset(b_offset)` cannot possibly fail to satisfy.
                let (a, b) = (&a.value, &b.value);
                let (a_size, b_size) = (a.size(self), b.size(self));
                let a_ptr = ptr;
                let b_offset = a_size.align_to(b.align(self).abi);
                assert!(b_offset.bytes() > 0); // we later use the offset to tell apart the fields
                let b_ptr = ptr.offset(b_offset, self)?;
                let a_val = alloc.read_scalar(self, a_ptr, a_size)?;
                let b_val = alloc.read_scalar(self, b_ptr, b_size)?;
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
            span_bug!(self.cur_span(), "primitive read failed for type: {:?}", op.layout.ty);
        }
    }

    /// Read a scalar from a place
    pub fn read_scalar(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, ScalarMaybeUninit<M::PointerTag>> {
        Ok(self.read_immediate(op)?.to_scalar_or_uninit())
    }

    // Turn the wide MPlace into a string (must already be dereferenced!)
    pub fn read_str(&self, mplace: MPlaceTy<'tcx, M::PointerTag>) -> InterpResult<'tcx, &str> {
        let len = mplace.len(self)?;
        let bytes = self.memory.read_bytes(mplace.ptr, Size::from_bytes(len))?;
        let str = std::str::from_utf8(bytes).map_err(|err| err_ub!(InvalidStr(err)))?;
        Ok(str)
    }

    /// Projection functions
    pub fn operand_field(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
        field: usize,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        let base = match op.try_as_mplace(self) {
            Ok(mplace) => {
                // We can reuse the mplace field computation logic for indirect operands.
                let field = self.mplace_field(mplace, field)?;
                return Ok(field.into());
            }
            Err(value) => value,
        };

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
            Immediate::Scalar(val) => span_bug!(
                self.cur_span(),
                "field access on non aggregate {:#?}, {:#?}",
                val,
                op.layout
            ),
        };
        Ok(OpTy { op: Operand::Immediate(immediate), layout: field_layout })
    }

    pub fn operand_index(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
        index: u64,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        if let Ok(index) = usize::try_from(index) {
            // We can just treat this as a field.
            self.operand_field(op, index)
        } else {
            // Indexing into a big array. This must be an mplace.
            let mplace = op.assert_mem_place(self);
            Ok(self.mplace_index(mplace, index)?.into())
        }
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
        proj_elem: mir::PlaceElem<'tcx>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        use rustc_middle::mir::ProjectionElem::*;
        Ok(match proj_elem {
            Field(field, _) => self.operand_field(base, field.index())?,
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

    /// Read from a local. Will not actually access the local if reading from a ZST.
    /// Will not access memory, instead an indirect `Operand` is returned.
    ///
    /// This is public because it is used by [priroda](https://github.com/oli-obk/priroda) to get an
    /// OpTy from a local
    pub fn access_local(
        &self,
        frame: &super::Frame<'mir, 'tcx, M::PointerTag, M::FrameExtra>,
        local: mir::Local,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        let layout = self.layout_of_local(frame, local, layout)?;
        let op = if layout.is_zst() {
            // Do not read from ZST, they might not be initialized
            Operand::Immediate(Scalar::zst().into())
        } else {
            M::access_local(&self, frame, local)?
        };
        Ok(OpTy { op, layout })
    }

    /// Every place can be read from, so we can turn them into an operand.
    /// This will definitely return `Indirect` if the place is a `Ptr`, i.e., this
    /// will never actually read from memory.
    #[inline(always)]
    pub fn place_to_op(
        &self,
        place: PlaceTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        let op = match *place {
            Place::Ptr(mplace) => Operand::Indirect(mplace),
            Place::Local { frame, local } => {
                *self.access_local(&self.stack()[frame], local, None)?
            }
        };
        Ok(OpTy { op, layout: place.layout })
    }

    // Evaluate a place with the goal of reading from it.  This lets us sometimes
    // avoid allocations.
    pub fn eval_place_to_op(
        &self,
        place: mir::Place<'tcx>,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        // Do not use the layout passed in as argument if the base we are looking at
        // here is not the entire place.
        let layout = if place.projection.is_empty() { layout } else { None };

        let base_op = self.access_local(self.frame(), place.local, layout)?;

        let op = place
            .projection
            .iter()
            .try_fold(base_op, |op, elem| self.operand_projection(op, elem))?;

        trace!("eval_place_to_op: got {:?}", *op);
        // Sanity-check the type we ended up with.
        debug_assert!(mir_assign_valid_types(
            *self.tcx,
            self.param_env,
            self.layout_of(self.subst_from_current_frame_and_normalize_erasing_regions(
                place.ty(&self.frame().body.local_decls, *self.tcx).ty
            ))?,
            op.layout,
        ));
        Ok(op)
    }

    /// Evaluate the operand, returning a place where you can then find the data.
    /// If you already know the layout, you can save two table lookups
    /// by passing it in here.
    pub fn eval_operand(
        &self,
        mir_op: &mir::Operand<'tcx>,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        use rustc_middle::mir::Operand::*;
        let op = match *mir_op {
            // FIXME: do some more logic on `move` to invalidate the old location
            Copy(place) | Move(place) => self.eval_place_to_op(place, layout)?,

            Constant(ref constant) => {
                let val =
                    self.subst_from_current_frame_and_normalize_erasing_regions(constant.literal);
                self.const_to_op(val, layout)?
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
    crate fn const_to_op(
        &self,
        val: &ty::Const<'tcx>,
        layout: Option<TyAndLayout<'tcx>>,
    ) -> InterpResult<'tcx, OpTy<'tcx, M::PointerTag>> {
        let tag_scalar = |scalar| -> InterpResult<'tcx, _> {
            Ok(match scalar {
                Scalar::Ptr(ptr) => Scalar::Ptr(self.global_base_pointer(ptr)?),
                Scalar::Raw { data, size } => Scalar::Raw { data, size },
            })
        };
        // Early-return cases.
        let val_val = match val.val {
            ty::ConstKind::Param(_) | ty::ConstKind::Bound(..) => throw_inval!(TooGeneric),
            ty::ConstKind::Error(_) => throw_inval!(TypeckError(ErrorReported)),
            ty::ConstKind::Unevaluated(def, substs, promoted) => {
                let instance = self.resolve(def, substs)?;
                return Ok(self.eval_to_allocation(GlobalId { instance, promoted })?.into());
            }
            ty::ConstKind::Infer(..) | ty::ConstKind::Placeholder(..) => {
                span_bug!(self.cur_span(), "const_to_op: Unexpected ConstKind {:?}", val)
            }
            ty::ConstKind::Value(val_val) => val_val,
        };
        // Other cases need layout.
        let layout =
            from_known_layout(self.tcx, self.param_env, layout, || self.layout_of(val.ty))?;
        let op = match val_val {
            ConstValue::ByRef { alloc, offset } => {
                let id = self.tcx.create_memory_alloc(alloc);
                // We rely on mutability being set correctly in that allocation to prevent writes
                // where none should happen.
                let ptr = self.global_base_pointer(Pointer::new(id, offset))?;
                Operand::Indirect(MemPlace::from_ptr(ptr, layout.align.abi))
            }
            ConstValue::Scalar(x) => Operand::Immediate(tag_scalar(x)?.into()),
            ConstValue::Slice { data, start, end } => {
                // We rely on mutability being set correctly in `data` to prevent writes
                // where none should happen.
                let ptr = Pointer::new(
                    self.tcx.create_memory_alloc(data),
                    Size::from_bytes(start), // offset: `start`
                );
                Operand::Immediate(Immediate::new_slice(
                    self.global_base_pointer(ptr)?.into(),
                    u64::try_from(end.checked_sub(start).unwrap()).unwrap(), // len: `end - start`
                    self,
                ))
            }
        };
        Ok(OpTy { op, layout })
    }

    /// Read discriminant, return the runtime value as well as the variant index.
    pub fn read_discriminant(
        &self,
        op: OpTy<'tcx, M::PointerTag>,
    ) -> InterpResult<'tcx, (Scalar<M::PointerTag>, VariantIdx)> {
        trace!("read_discriminant_value {:#?}", op.layout);
        // Get type and layout of the discriminant.
        let discr_layout = self.layout_of(op.layout.ty.discriminant_ty(*self.tcx))?;
        trace!("discriminant type: {:?}", discr_layout.ty);

        // We use "discriminant" to refer to the value associated with a particular enum variant.
        // This is not to be confused with its "variant index", which is just determining its position in the
        // declared list of variants -- they can differ with explicitly assigned discriminants.
        // We use "tag" to refer to how the discriminant is encoded in memory, which can be either
        // straight-forward (`TagEncoding::Direct`) or with a niche (`TagEncoding::Niche`).
        let (tag_scalar_layout, tag_encoding, tag_field) = match op.layout.variants {
            Variants::Single { index } => {
                let discr = match op.layout.ty.discriminant_for_variant(*self.tcx, index) {
                    Some(discr) => {
                        // This type actually has discriminants.
                        assert_eq!(discr.ty, discr_layout.ty);
                        Scalar::from_uint(discr.val, discr_layout.size)
                    }
                    None => {
                        // On a type without actual discriminants, variant is 0.
                        assert_eq!(index.as_u32(), 0);
                        Scalar::from_uint(index.as_u32(), discr_layout.size)
                    }
                };
                return Ok((discr, index));
            }
            Variants::Multiple { ref tag, ref tag_encoding, tag_field, .. } => {
                (tag, tag_encoding, tag_field)
            }
        };

        // There are *three* layouts that come into play here:
        // - The discriminant has a type for typechecking. This is `discr_layout`, and is used for
        //   the `Scalar` we return.
        // - The tag (encoded discriminant) has layout `tag_layout`. This is always an integer type,
        //   and used to interpret the value we read from the tag field.
        //   For the return value, a cast to `discr_layout` is performed.
        // - The field storing the tag has a layout, which is very similar to `tag_layout` but
        //   may be a pointer. This is `tag_val.layout`; we just use it for sanity checks.

        // Get layout for tag.
        let tag_layout = self.layout_of(tag_scalar_layout.value.to_int_ty(*self.tcx))?;

        // Read tag and sanity-check `tag_layout`.
        let tag_val = self.read_immediate(self.operand_field(op, tag_field)?)?;
        assert_eq!(tag_layout.size, tag_val.layout.size);
        assert_eq!(tag_layout.abi.is_signed(), tag_val.layout.abi.is_signed());
        let tag_val = tag_val.to_scalar()?;
        trace!("tag value: {:?}", tag_val);

        // Figure out which discriminant and variant this corresponds to.
        Ok(match *tag_encoding {
            TagEncoding::Direct => {
                let tag_bits = self
                    .force_bits(tag_val, tag_layout.size)
                    .map_err(|_| err_ub!(InvalidTag(tag_val.erase_tag())))?;
                // Cast bits from tag layout to discriminant layout.
                let discr_val = self.cast_from_scalar(tag_bits, tag_layout, discr_layout.ty);
                let discr_bits = discr_val.assert_bits(discr_layout.size);
                // Convert discriminant to variant index, and catch invalid discriminants.
                let index = match *op.layout.ty.kind() {
                    ty::Adt(adt, _) => {
                        adt.discriminants(*self.tcx).find(|(_, var)| var.val == discr_bits)
                    }
                    ty::Generator(def_id, substs, _) => {
                        let substs = substs.as_generator();
                        substs
                            .discriminants(def_id, *self.tcx)
                            .find(|(_, var)| var.val == discr_bits)
                    }
                    _ => span_bug!(self.cur_span(), "tagged layout for non-adt non-generator"),
                }
                .ok_or_else(|| err_ub!(InvalidTag(tag_val.erase_tag())))?;
                // Return the cast value, and the index.
                (discr_val, index.0)
            }
            TagEncoding::Niche { dataful_variant, ref niche_variants, niche_start } => {
                // Compute the variant this niche value/"tag" corresponds to. With niche layout,
                // discriminant (encoded in niche/tag) and variant index are the same.
                let variants_start = niche_variants.start().as_u32();
                let variants_end = niche_variants.end().as_u32();
                let variant = match tag_val.to_bits_or_ptr(tag_layout.size, self) {
                    Err(ptr) => {
                        // The niche must be just 0 (which an inbounds pointer value never is)
                        let ptr_valid = niche_start == 0
                            && variants_start == variants_end
                            && !self.memory.ptr_may_be_null(ptr);
                        if !ptr_valid {
                            throw_ub!(InvalidTag(tag_val.erase_tag()))
                        }
                        dataful_variant
                    }
                    Ok(tag_bits) => {
                        // We need to use machine arithmetic to get the relative variant idx:
                        // variant_index_relative = tag_val - niche_start_val
                        let tag_val = ImmTy::from_uint(tag_bits, tag_layout);
                        let niche_start_val = ImmTy::from_uint(niche_start, tag_layout);
                        let variant_index_relative_val =
                            self.binary_op(mir::BinOp::Sub, tag_val, niche_start_val)?;
                        let variant_index_relative = variant_index_relative_val
                            .to_scalar()?
                            .assert_bits(tag_val.layout.size);
                        // Check if this is in the range that indicates an actual discriminant.
                        if variant_index_relative <= u128::from(variants_end - variants_start) {
                            let variant_index_relative = u32::try_from(variant_index_relative)
                                .expect("we checked that this fits into a u32");
                            // Then computing the absolute variant idx should not overflow any more.
                            let variant_index = variants_start
                                .checked_add(variant_index_relative)
                                .expect("overflow computing absolute variant idx");
                            let variants_len = op
                                .layout
                                .ty
                                .ty_adt_def()
                                .expect("tagged layout for non adt")
                                .variants
                                .len();
                            assert!(usize::try_from(variant_index).unwrap() < variants_len);
                            VariantIdx::from_u32(variant_index)
                        } else {
                            dataful_variant
                        }
                    }
                };
                // Compute the size of the scalar we need to return.
                // No need to cast, because the variant index directly serves as discriminant and is
                // encoded in the tag.
                (Scalar::from_uint(variant.as_u32(), discr_layout.size), variant)
            }
        })
    }
}
