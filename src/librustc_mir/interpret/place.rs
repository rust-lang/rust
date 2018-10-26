// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Computations on places -- field projections, going from mir::Place, and writing
//! into a place.
//! All high-level functions to write to memory work on places as destinations.

use std::convert::TryFrom;
use std::hash::Hash;

use rustc::mir;
use rustc::ty;
use rustc::ty::layout::{self, Size, Align, LayoutOf, TyLayout, HasDataLayout};

use rustc::mir::interpret::{
    GlobalId, AllocId, Allocation, Scalar, EvalResult, Pointer, PointerArithmetic
};
use super::{
    EvalContext, Machine, AllocMap,
    Value, ValTy, ScalarMaybeUndef, Operand, OpTy, MemoryKind
};

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Place<Tag=(), Id=AllocId> {
    /// A place may have an integral pointer for ZSTs, and since it might
    /// be turned back into a reference before ever being dereferenced.
    /// However, it may never be undef.
    pub ptr: Scalar<Tag, Id>,
    pub align: Align,
    /// Metadata for unsized places.  Interpretation is up to the type.
    /// Must not be present for sized types, but can be missing for unsized types
    /// (e.g. `extern type`).
    pub meta: Option<Scalar<Tag, Id>>,
}

/// A Place with its layout. Constructing it is only possible in this module.
#[derive(Copy, Clone, Debug)]
pub struct PlaceTy<'tcx, Tag=()> {
    place: Place<Tag>,
    pub layout: TyLayout<'tcx>,
}

impl<'tcx, Tag> ::std::ops::Deref for PlaceTy<'tcx, Tag> {
    type Target = Place<Tag>;
    #[inline(always)]
    fn deref(&self) -> &Place<Tag> {
        &self.place
    }
}

impl Place {
    #[inline]
    pub fn with_default_tag<Tag>(self) -> Place<Tag>
        where Tag: Default
    {
        Place {
            ptr: self.ptr.with_default_tag(),
            align: self.align,
            meta: self.meta.map(Scalar::with_default_tag),
        }
    }
}

impl<Tag> Place<Tag> {
    #[inline]
    pub fn erase_tag(self) -> Place
    {
        Place {
            ptr: self.ptr.erase_tag(),
            align: self.align,
            meta: self.meta.map(Scalar::erase_tag),
        }
    }

    #[inline(always)]
    pub fn from_scalar_ptr(ptr: Scalar<Tag>, align: Align) -> Self {
        Place {
            ptr,
            align,
            meta: None,
        }
    }

    /// Produces a Place that will error if attempted to be read from or written to
    #[inline(always)]
    pub fn null(cx: impl HasDataLayout) -> Self {
        Self::from_scalar_ptr(Scalar::ptr_null(cx), Align::from_bytes(1, 1).unwrap())
    }

    #[inline(always)]
    pub fn from_ptr(ptr: Pointer<Tag>, align: Align) -> Self {
        Self::from_scalar_ptr(ptr.into(), align)
    }

    #[inline(always)]
    pub fn to_scalar_ptr_align(self) -> (Scalar<Tag>, Align) {
        assert!(self.meta.is_none());
        (self.ptr, self.align)
    }

    /// metact the ptr part of the place
    #[inline(always)]
    pub fn to_ptr(self) -> EvalResult<'tcx, Pointer<Tag>> {
        // At this point, we forget about the alignment information --
        // the place has been turned into a reference, and no matter where it came from,
        // it now must be aligned.
        self.to_scalar_ptr_align().0.to_ptr()
    }
}

impl<'tcx, Tag> PlaceTy<'tcx, Tag> {
    /// Produces a Place that works for ZST but nothing else
    #[inline]
    pub fn dangling(layout: TyLayout<'tcx>, cx: impl HasDataLayout) -> Self {
        PlaceTy {
            place: Place::from_scalar_ptr(
                Scalar::from_uint(layout.align.abi(), cx.pointer_size()),
                layout.align
            ),
            layout
        }
    }

    #[inline]
    fn from_aligned_ptr(ptr: Pointer<Tag>, layout: TyLayout<'tcx>) -> Self {
        PlaceTy { place: Place::from_ptr(ptr, layout.align), layout }
    }

    #[inline]
    pub(super) fn len(self, cx: impl HasDataLayout) -> EvalResult<'tcx, u64> {
        if self.layout.is_unsized() {
            // We need to consult `meta` metadata
            match self.layout.ty.sty {
                ty::Slice(..) | ty::Str =>
                    return self.place.meta.unwrap().to_usize(cx),
                _ => bug!("len not supported on unsized type {:?}", self.layout.ty),
            }
        } else {
            // Go through the layout.  There are lots of types that support a length,
            // e.g. SIMD types.
            match self.layout.fields {
                layout::FieldPlacement::Array { count, .. } => Ok(count),
                _ => bug!("len not supported on sized type {:?}", self.layout.ty),
            }
        }
    }

    #[inline]
    pub(super) fn vtable(self) -> EvalResult<'tcx, Pointer<Tag>> {
        match self.layout.ty.sty {
            ty::Dynamic(..) => self.place.meta.unwrap().to_ptr(),
            _ => bug!("vtable not supported on type {:?}", self.layout.ty),
        }
    }
}

impl<'tcx, Tag: ::std::fmt::Debug> OpTy<'tcx, Tag> {
    #[inline(always)]
    pub fn try_as_place(self) -> Result<PlaceTy<'tcx, Tag>, Value<Tag>> {
        match self.op {
            Operand::Indirect(place) => Ok(PlaceTy { place, layout: self.layout }),
            Operand::Immediate(value) => Err(value),
        }
    }

    #[inline(always)]
    pub fn to_place(self) -> PlaceTy<'tcx, Tag> {
        self.try_as_place().unwrap()
    }
}

// separating the pointer tag for `impl Trait`, see https://github.com/rust-lang/rust/issues/54385
impl<'a, 'mir, 'tcx, Tag, M> EvalContext<'a, 'mir, 'tcx, M>
where
    Tag: ::std::fmt::Debug+Default+Copy+Eq+Hash+'static,
    M: Machine<'a, 'mir, 'tcx, PointerTag=Tag>,
    M::MemoryMap: AllocMap<AllocId, (MemoryKind<M::MemoryKinds>, Allocation<Tag, M::AllocExtra>)>,
{
    /// Take a value, which represents a (thin or fat) reference, and make it a place.
    /// Alignment is just based on the type.  This is the inverse of `create_ref`.
    pub fn ref_to_place(
        &self,
        val: ValTy<'tcx, M::PointerTag>,
    ) -> EvalResult<'tcx, PlaceTy<'tcx, M::PointerTag>> {
        let ptr = match val.to_scalar_ptr()? {
            Scalar::Ptr(ptr) if M::ENABLE_PTR_TRACKING_HOOKS => {
                // Machine might want to track the `*` operator
                let tag = M::tag_dereference(self, ptr, val.layout.ty)?;
                Scalar::Ptr(Pointer::new_with_tag(ptr.alloc_id, ptr.offset, tag))
            }
            other => other,
        };

        let pointee_type = val.layout.ty.builtin_deref(true).unwrap().ty;
        let layout = self.layout_of(pointee_type)?;
        let align = layout.align;

        let place = match *val {
            Value::Scalar(_) =>
                Place { ptr, align, meta: None },
            Value::ScalarPair(_, meta) =>
                Place { ptr, align, meta: Some(meta.not_undef()?) },
        };
        Ok(PlaceTy { place, layout })
    }

    /// Turn a place into a (thin or fat) pointer, as a reference, pointing to the same space.
    /// This is the inverse of `ref_to_place`.
    pub fn create_ref(
        &mut self,
        place: PlaceTy<'tcx, M::PointerTag>,
        borrow_kind: Option<mir::BorrowKind>,
    ) -> EvalResult<'tcx, Value<M::PointerTag>> {
        let ptr = match place.ptr {
            Scalar::Ptr(ptr) if M::ENABLE_PTR_TRACKING_HOOKS => {
                // Machine might want to track the `&` operator
                let (size, _) = self.size_and_align_of_place(place)?
                    .expect("create_ref cannot determine size");
                let tag = M::tag_reference(self, ptr, place.layout.ty, size, borrow_kind)?;
                Scalar::Ptr(Pointer::new_with_tag(ptr.alloc_id, ptr.offset, tag))
            },
            other => other,
        };
        Ok(match place.meta {
            None => Value::Scalar(ptr.into()),
            Some(meta) => Value::ScalarPair(ptr.into(), meta.into()),
        })
    }

    /// Offset a pointer to project to a field. Unlike place_field, this is always
    /// possible without allocating, so it can take &self. Also return the field's layout.
    /// This supports both struct and array fields.
    #[inline(always)]
    pub fn place_field(
        &self,
        base: PlaceTy<'tcx, M::PointerTag>,
        field: u64,
    ) -> EvalResult<'tcx, PlaceTy<'tcx, M::PointerTag>> {
        // Not using the layout method because we want to compute on u64
        let offset = match base.layout.fields {
            layout::FieldPlacement::Arbitrary { ref offsets, .. } =>
                offsets[usize::try_from(field).unwrap()],
            layout::FieldPlacement::Array { stride, .. } => {
                let len = base.len(self)?;
                assert!(field < len, "Tried to access element {} of array/slice with length {}",
                    field, len);
                stride * field
            }
            layout::FieldPlacement::Union(count) => {
                assert!(field < count as u64,
                        "Tried to access field {} of union with {} fields", field, count);
                // Offset is always 0
                Size::from_bytes(0)
            }
        };
        // the only way conversion can fail if is this is an array (otherwise we already panicked
        // above). In that case, all fields are equal.
        let field_layout = base.layout.field(self, usize::try_from(field).unwrap_or(0))?;

        // Offset may need adjustment for unsized fields
        let (meta, offset) = if field_layout.is_unsized() {
            // re-use parent metadata to determine dynamic field layout
            let (_, align) = self.size_and_align_of(base.meta, field_layout)?
                .expect("Fields cannot be extern types");
            (base.meta, offset.abi_align(align))
        } else {
            // base.meta could be present; we might be accessing a sized field of an unsized
            // struct.
            (None, offset)
        };

        let ptr = base.ptr.ptr_offset(offset, self)?;
        let align = base.align
            // We do not look at `base.layout.align` nor `field_layout.align`, unlike
            // codegen -- mostly to see if we can get away with that
            .restrict_for_offset(offset); // must be last thing that happens

        Ok(PlaceTy { place: Place { ptr, align, meta }, layout: field_layout })
    }

    // Iterates over all fields of an array. Much more efficient than doing the
    // same by repeatedly calling `place_array`.
    pub fn place_array_fields(
        &self,
        base: PlaceTy<'tcx, Tag>,
    ) ->
        EvalResult<'tcx, impl Iterator<Item=EvalResult<'tcx, PlaceTy<'tcx, Tag>>> + 'a>
    {
        let len = base.len(self)?; // also asserts that we have a type where this makes sense
        let stride = match base.layout.fields {
            layout::FieldPlacement::Array { stride, .. } => stride,
            _ => bug!("place_array_fields: expected an array layout"),
        };
        let layout = base.layout.field(self, 0)?;
        let dl = &self.tcx.data_layout;
        Ok((0..len).map(move |i| {
            let ptr = base.ptr.ptr_offset(i * stride, dl)?;
            Ok(PlaceTy {
                place: Place { ptr, align: base.align, meta: None },
                layout
            })
        }))
    }

    pub fn place_subslice(
        &self,
        base: PlaceTy<'tcx, M::PointerTag>,
        from: u64,
        to: u64,
    ) -> EvalResult<'tcx, PlaceTy<'tcx, M::PointerTag>> {
        let len = base.len(self)?; // also asserts that we have a type where this makes sense
        assert!(from <= len - to);

        // Not using layout method because that works with usize, and does not work with slices
        // (that have count 0 in their layout).
        let from_offset = match base.layout.fields {
            layout::FieldPlacement::Array { stride, .. } =>
                stride * from,
            _ => bug!("Unexpected layout of index access: {:#?}", base.layout),
        };
        let ptr = base.ptr.ptr_offset(from_offset, self)?;

        // Compute meta and new layout
        let inner_len = len - to - from;
        let (meta, ty) = match base.layout.ty.sty {
            // It is not nice to match on the type, but that seems to be the only way to
            // implement this.
            ty::Array(inner, _) =>
                (None, self.tcx.mk_array(inner, inner_len)),
            ty::Slice(..) => {
                let len = Scalar::from_uint(inner_len, self.pointer_size());
                (Some(len), base.layout.ty)
            }
            _ =>
                bug!("cannot subslice non-array type: `{:?}`", base.layout.ty),
        };
        let layout = self.layout_of(ty)?;

        Ok(PlaceTy {
            place: Place { ptr, align: base.align, meta },
            layout
        })
    }

    pub fn place_downcast(
        &self,
        base: PlaceTy<'tcx, M::PointerTag>,
        variant: usize,
    ) -> EvalResult<'tcx, PlaceTy<'tcx, M::PointerTag>> {
        // Downcasts only change the layout
        assert!(base.meta.is_none());
        Ok(PlaceTy { layout: base.layout.for_variant(self, variant), ..base })
    }

    /// Project into an place
    pub fn place_projection(
        &self,
        base: PlaceTy<'tcx, M::PointerTag>,
        proj_elem: &mir::PlaceElem<'tcx>,
    ) -> EvalResult<'tcx, PlaceTy<'tcx, M::PointerTag>> {
        use rustc::mir::ProjectionElem::*;
        Ok(match *proj_elem {
            Field(field, _) => self.place_field(base, field.index() as u64)?,
            Downcast(_, variant) => self.place_downcast(base, variant)?,
            Deref => self.deref_operand(base.into())?,

            Index(local) => {
                let n = *self.frame().locals[local].access()?;
                let n_layout = self.layout_of(self.tcx.types.usize)?;
                let n = self.read_scalar(OpTy { op: Operand::Indirect(n), layout: n_layout })?;
                let n = n.to_bits(self.tcx.data_layout.pointer_size)?;
                self.place_field(base, u64::try_from(n).unwrap())?
            }

            ConstantIndex {
                offset,
                min_length,
                from_end,
            } => {
                let n = base.len(self)?;
                assert!(n >= min_length as u64);

                let index = if from_end {
                    n - u64::from(offset)
                } else {
                    u64::from(offset)
                };

                self.place_field(base, index)?
            }

            Subslice { from, to } =>
                self.place_subslice(base, u64::from(from), u64::from(to))?,
        })
    }

    /// Evaluate statics and promoteds to an `MPlace`.
    pub(super) fn eval_place(
        &self,
        mir_place: &mir::Place<'tcx>
    ) -> EvalResult<'tcx, PlaceTy<'tcx, M::PointerTag>> {
        use rustc::mir::Place::*;
        Ok(match *mir_place {
            Promoted(ref promoted) => {
                let instance = self.frame().instance;
                let op = self.global_to_op(GlobalId {
                    instance,
                    promoted: Some(promoted.0),
                })?;
                let place = op.to_place(); // these are always in memory
                let ty = self.monomorphize(promoted.1, self.substs());
                PlaceTy {
                    place,
                    layout: self.layout_of(ty)?,
                }
            }

            Static(ref static_) => {
                let ty = self.monomorphize(static_.ty, self.substs());
                let layout = self.layout_of(ty)?;
                let instance = ty::Instance::mono(*self.tcx, static_.def_id);
                let cid = GlobalId {
                    instance,
                    promoted: None
                };
                // Just create a lazy reference, so we can support recursive statics.
                // tcx takes are of assigning every static one and only one unique AllocId.
                // When the data here is ever actually used, memory will notice,
                // and it knows how to deal with alloc_id that are present in the
                // global table but not in its local memory: It calls back into tcx through
                // a query, triggering the CTFE machinery to actually turn this lazy reference
                // into a bunch of bytes.  IOW, statics are evaluated with CTFE even when
                // this EvalContext uses another Machine (e.g., in miri).  This is what we
                // want!  This way, computing statics works concistently between codegen
                // and miri: They use the same query to eventually obtain a `ty::Const`
                // and use that for further computation.
                let alloc = self.tcx.alloc_map.lock().intern_static(cid.instance.def_id());
                PlaceTy::from_aligned_ptr(Pointer::from(alloc).with_default_tag(), layout)
            }

            Local(mir::RETURN_PLACE) => match self.frame().return_place {
                Some(return_place) =>
                    // We use our layout to verify our assumption; caller will validate
                    // their layout on return.
                    PlaceTy {
                        place: *return_place,
                        layout: self.layout_of_local(self.frame(), mir::RETURN_PLACE)?,
                    },
                None => return err!(InvalidNullPointerUsage),
            },
            Local(local) => PlaceTy {
                place: *self.frame().locals[local].access()?,
                layout: self.layout_of_local(self.frame(), local)?,
            },

            Projection(ref proj) => {
                let place = self.eval_place(&proj.base)?;
                self.place_projection(place, &proj.elem)?
            },
        })
    }

    /// Write a scalar to a place
    pub fn write_scalar(
        &mut self,
        val: impl Into<ScalarMaybeUndef<M::PointerTag>>,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> EvalResult<'tcx> {
        self.write_value(Value::Scalar(val.into()), dest)
    }

    /// Write a value to a place
    #[inline(always)]
    pub fn write_value(
        &mut self,
        src_val: Value<M::PointerTag>,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> EvalResult<'tcx> {
        self.write_value_no_validate(src_val, dest)?;

        if M::enforce_validity(self) {
            // Data got changed, better make sure it matches the type!
            self.validate_operand(dest.into(), &mut vec![], None, /*const_mode*/false)?;
        }

        Ok(())
    }

    /// Write a value to memory.
    /// If you use this you are responsible for validating that things git copied at the
    /// right type.
    fn write_value_no_validate(
        &mut self,
        value: Value<M::PointerTag>,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> EvalResult<'tcx> {
        let (ptr, ptr_align) = dest.to_scalar_ptr_align();
        // Note that it is really important that the type here is the right one, and matches the
        // type things are read at. In case `src_val` is a `ScalarPair`, we don't do any magic here
        // to handle padding properly, which is only correct if we never look at this data with the
        // wrong type.

        // Nothing to do for ZSTs, other than checking alignment
        if dest.layout.is_zst() {
            self.memory.check_align(ptr, ptr_align)?;
            return Ok(());
        }

        let ptr = ptr.to_ptr()?;
        // FIXME: We should check that there are dest.layout.size many bytes available in
        // memory.  The code below is not sufficient, with enough padding it might not
        // cover all the bytes!
        match value {
            Value::Scalar(scalar) => {
                match dest.layout.abi {
                    layout::Abi::Scalar(_) => {}, // fine
                    _ => bug!("write_value: invalid Scalar layout: {:#?}",
                            dest.layout)
                }

                self.memory.write_scalar(
                    ptr, ptr_align.min(dest.layout.align), scalar, dest.layout.size
                )
            }
            Value::ScalarPair(a_val, b_val) => {
                let (a, b) = match dest.layout.abi {
                    layout::Abi::ScalarPair(ref a, ref b) => (&a.value, &b.value),
                    _ => bug!("write_value: invalid ScalarPair layout: {:#?}",
                              dest.layout)
                };
                let (a_size, b_size) = (a.size(&self), b.size(&self));
                let (a_align, b_align) = (a.align(&self), b.align(&self));
                let b_offset = a_size.abi_align(b_align);
                let b_ptr = ptr.offset(b_offset, &self)?.into();

                // It is tempting to verify `b_offset` against `layout.fields.offset(1)`,
                // but that does not work: We could be a newtype around a pair, then the
                // fields do not match the `ScalarPair` components.

                self.memory.write_scalar(ptr, ptr_align.min(a_align), a_val, a_size)?;
                self.memory.write_scalar(b_ptr, ptr_align.min(b_align), b_val, b_size)
            }
        }
    }

    /// Copy the data from an operand to a place.  This does not support transmuting!
    /// Use `copy_op_transmute` if the layouts could disagree.
    #[inline(always)]
    pub fn copy_op(
        &mut self,
        src: OpTy<'tcx, M::PointerTag>,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> EvalResult<'tcx> {
        self.copy_op_no_validate(src, dest)?;

        if M::enforce_validity(self) {
            // Data got changed, better make sure it matches the type!
            self.validate_operand(dest.into(), &mut vec![], None, /*const_mode*/false)?;
        }

        Ok(())
    }

    /// Copy the data from an operand to a place.  This does not support transmuting!
    /// Use `copy_op_transmute` if the layouts could disagree.
    /// Also, if you use this you are responsible for validating that things git copied at the
    /// right type.
    fn copy_op_no_validate(
        &mut self,
        src: OpTy<'tcx, M::PointerTag>,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> EvalResult<'tcx> {
        debug_assert!(!src.layout.is_unsized() && !dest.layout.is_unsized(),
            "Cannot copy unsized data");
        // We do NOT compare the types for equality, because well-typed code can
        // actually "transmute" `&mut T` to `&T` in an assignment without a cast.
        assert!(src.layout.details == dest.layout.details,
            "Layout mismatch when copying!\nsrc: {:#?}\ndest: {:#?}", src, dest);

        // Let us see if the layout is simple so we take a shortcut
        let src = match self.try_read_value(src)? {
            Ok(src_val) => {
                // Yay, we got a value that we can write directly.
                return self.write_value_no_validate(src_val, dest);
            }
            Err(place) => place,
        };
        trace!("copy_op: {:?} <- {:?}: {}", *dest, src, dest.layout.ty);

        let (src_ptr, src_align) = src.to_scalar_ptr_align();
        let (dest_ptr, dest_align) = dest.to_scalar_ptr_align();
        self.memory.copy(
            src_ptr, src_align,
            dest_ptr, dest_align,
            dest.layout.size, false
        )?;

        Ok(())
    }

    /// Copy the data from an operand to a place.  The layouts may disagree, but they must
    /// have the same size.
    pub fn copy_op_transmute(
        &mut self,
        src: OpTy<'tcx, M::PointerTag>,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> EvalResult<'tcx> {
        if src.layout.details == dest.layout.details {
            // Fast path: Just use normal `copy_op`
            return self.copy_op(src, dest);
        }
        // We still require the sizes to match
        debug_assert!(!src.layout.is_unsized() && !dest.layout.is_unsized(),
            "Cannot copy unsized data");
        assert!(src.layout.size == dest.layout.size,
            "Size mismatch when transmuting!\nsrc: {:#?}\ndest: {:#?}", src, dest);

        // The hard case is `ScalarPair`.  `src` is already read from memory in this case,
        // using `src.layout` to figure out which bytes to use for the 1st and 2nd field.
        // We have to write them to `dest` at the offsets they were *read at*, which is
        // not necessarily the same as the offsets in `dest.layout`!
        // Hence we do the copy with the source layout on both sides.  We also make sure to write
        // into memory, because if `dest` is a local we would not even have a way to write
        // at the `src` offsets; the fact that we came from a different layout would
        // just be lost.
        self.copy_op_no_validate(
            src,
            PlaceTy { place: *dest, layout: src.layout },
        )?;

        if M::enforce_validity(self) {
            // Data got changed, better make sure it matches the type!
            self.validate_operand(dest.into(), &mut vec![], None, /*const_mode*/false)?;
        }

        Ok(())
    }

    pub fn allocate(
        &mut self,
        layout: TyLayout<'tcx>,
        kind: MemoryKind<M::MemoryKinds>,
    ) -> EvalResult<'tcx, PlaceTy<'tcx, M::PointerTag>> {
        if layout.is_unsized() {
            assert!(self.tcx.features().unsized_locals, "cannot alloc memory for unsized type");
            // FIXME: What should we do here?
            Ok(PlaceTy::dangling(layout, &self))
        } else {
            let ptr = self.memory.allocate(layout.size, layout.align, kind)?;
            Ok(PlaceTy::from_aligned_ptr(ptr, layout))
        }
    }

    pub fn write_discriminant_index(
        &mut self,
        variant_index: usize,
        dest: PlaceTy<'tcx, M::PointerTag>,
    ) -> EvalResult<'tcx> {
        match dest.layout.variants {
            layout::Variants::Single { index } => {
                assert_eq!(index, variant_index);
            }
            layout::Variants::Tagged { ref tag, .. } => {
                let adt_def = dest.layout.ty.ty_adt_def().unwrap();
                assert!(variant_index < adt_def.variants.len());
                let discr_val = adt_def
                    .discriminant_for_variant(*self.tcx, variant_index)
                    .val;

                // raw discriminants for enums are isize or bigger during
                // their computation, but the in-memory tag is the smallest possible
                // representation
                let size = tag.value.size(self.tcx.tcx);
                let shift = 128 - size.bits();
                let discr_val = (discr_val << shift) >> shift;

                let discr_dest = self.place_field(dest, 0)?;
                self.write_scalar(Scalar::from_uint(discr_val, size), discr_dest)?;
            }
            layout::Variants::NicheFilling {
                dataful_variant,
                ref niche_variants,
                niche_start,
                ..
            } => {
                assert!(variant_index < dest.layout.ty.ty_adt_def().unwrap().variants.len());
                if variant_index != dataful_variant {
                    let niche_dest =
                        self.place_field(dest, 0)?;
                    let niche_value = ((variant_index - niche_variants.start()) as u128)
                        .wrapping_add(niche_start);
                    self.write_scalar(
                        Scalar::from_uint(niche_value, niche_dest.layout.size),
                        niche_dest
                    )?;
                }
            }
        }

        Ok(())
    }

    /// Turn a place with a `dyn Trait` type into a place with the actual dynamic type.
    /// Also return some more information so drop doesn't have to run the same code twice.
    pub(super) fn unpack_dyn_trait(&self, place: PlaceTy<'tcx, M::PointerTag>)
    -> EvalResult<'tcx, (ty::Instance<'tcx>, PlaceTy<'tcx, M::PointerTag>)> {
        let vtable = place.vtable()?; // also sanity checks the type
        let (instance, ty) = self.read_drop_type_from_vtable(vtable)?;
        let layout = self.layout_of(ty)?;

        // More sanity checks
        if cfg!(debug_assertions) {
            let (size, align) = self.read_size_and_align_from_vtable(vtable)?;
            assert_eq!(size, layout.size);
            assert_eq!(align.abi(), layout.align.abi()); // only ABI alignment is preserved
        }

        let place = PlaceTy {
            place: Place { meta: None, ..*place },
            layout
        };
        Ok((instance, place))
    }
}
