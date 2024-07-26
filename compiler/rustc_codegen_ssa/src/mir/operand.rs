use super::place::{PlaceRef, PlaceValue};
use super::{FunctionCx, LocalRef};

use crate::size_of_val;
use crate::traits::*;
use crate::MemFlags;

use rustc_middle::bug;
use rustc_middle::mir::interpret::{alloc_range, Pointer, Scalar};
use rustc_middle::mir::{self, ConstValue};
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::Ty;
use rustc_target::abi::{self, Abi, Align, Size};

use std::fmt;

use arrayvec::ArrayVec;
use either::Either;
use tracing::debug;

/// The representation of a Rust value. The enum variant is in fact
/// uniquely determined by the value's type, but is kept as a
/// safety check.
#[derive(Copy, Clone, Debug)]
pub enum OperandValue<V> {
    /// A reference to the actual operand. The data is guaranteed
    /// to be valid for the operand's lifetime.
    /// The second value, if any, is the extra data (vtable or length)
    /// which indicates that it refers to an unsized rvalue.
    ///
    /// An `OperandValue` *must* be this variant for any type for which
    /// [`LayoutTypeMethods::is_backend_ref`] returns `true`.
    /// (That basically amounts to "isn't one of the other variants".)
    ///
    /// This holds a [`PlaceValue`] (like a [`PlaceRef`] does) with a pointer
    /// to the location holding the value. The type behind that pointer is the
    /// one returned by [`LayoutTypeMethods::backend_type`].
    Ref(PlaceValue<V>),
    /// A single LLVM immediate value.
    ///
    /// An `OperandValue` *must* be this variant for any type for which
    /// [`LayoutTypeMethods::is_backend_immediate`] returns `true`.
    /// The backend value in this variant must be the *immediate* backend type,
    /// as returned by [`LayoutTypeMethods::immediate_backend_type`].
    Immediate(V),
    /// A pair of immediate LLVM values. Used by fat pointers too.
    ///
    /// An `OperandValue` *must* be this variant for any type for which
    /// [`LayoutTypeMethods::is_backend_scalar_pair`] returns `true`.
    /// The backend values in this variant must be the *immediate* backend types,
    /// as returned by [`LayoutTypeMethods::scalar_pair_element_backend_type`]
    /// with `immediate: true`.
    Pair(V, V),
    /// A value taking no bytes, and which therefore needs no LLVM value at all.
    ///
    /// If you ever need a `V` to pass to something, get a fresh poison value
    /// from [`ConstMethods::const_poison`].
    ///
    /// An `OperandValue` *must* be this variant for any type for which
    /// `is_zst` on its `Layout` returns `true`. Note however that
    /// these values can still require alignment.
    ZeroSized,
}

impl<V: CodegenObject> OperandValue<V> {
    /// If this is ZeroSized/Immediate/Pair, return an array of the 0/1/2 values.
    /// If this is Ref, return the place.
    #[inline]
    pub fn immediates_or_place(self) -> Either<ArrayVec<V, 2>, PlaceValue<V>> {
        match self {
            OperandValue::ZeroSized => Either::Left(ArrayVec::new()),
            OperandValue::Immediate(a) => Either::Left(ArrayVec::from_iter([a])),
            OperandValue::Pair(a, b) => Either::Left([a, b].into()),
            OperandValue::Ref(p) => Either::Right(p),
        }
    }

    /// Given an array of 0/1/2 immediate values, return ZeroSized/Immediate/Pair.
    #[inline]
    pub fn from_immediates(immediates: ArrayVec<V, 2>) -> Self {
        let mut it = immediates.into_iter();
        let Some(a) = it.next() else {
            return OperandValue::ZeroSized;
        };
        let Some(b) = it.next() else {
            return OperandValue::Immediate(a);
        };
        OperandValue::Pair(a, b)
    }

    /// Treat this value as a pointer and return the data pointer and
    /// optional metadata as backend values.
    ///
    /// If you're making a place, use [`Self::deref`] instead.
    pub fn pointer_parts(self) -> (V, Option<V>) {
        match self {
            OperandValue::Immediate(llptr) => (llptr, None),
            OperandValue::Pair(llptr, llextra) => (llptr, Some(llextra)),
            _ => bug!("OperandValue cannot be a pointer: {self:?}"),
        }
    }

    /// Treat this value as a pointer and return the place to which it points.
    ///
    /// The pointer immediate doesn't inherently know its alignment,
    /// so you need to pass it in. If you want to get it from a type's ABI
    /// alignment, then maybe you want [`OperandRef::deref`] instead.
    ///
    /// This is the inverse of [`PlaceValue::address`].
    pub fn deref(self, align: Align) -> PlaceValue<V> {
        let (llval, llextra) = self.pointer_parts();
        PlaceValue { llval, llextra, align }
    }

    pub(crate) fn is_expected_variant_for_type<'tcx, Cx: LayoutTypeMethods<'tcx>>(
        &self,
        cx: &Cx,
        ty: TyAndLayout<'tcx>,
    ) -> bool {
        match self {
            OperandValue::ZeroSized => ty.is_zst(),
            OperandValue::Immediate(_) => cx.is_backend_immediate(ty),
            OperandValue::Pair(_, _) => cx.is_backend_scalar_pair(ty),
            OperandValue::Ref(_) => cx.is_backend_ref(ty),
        }
    }
}

/// An `OperandRef` is an "SSA" reference to a Rust value, along with
/// its type.
///
/// NOTE: unless you know a value's type exactly, you should not
/// generate LLVM opcodes acting on it and instead act via methods,
/// to avoid nasty edge cases. In particular, using `Builder::store`
/// directly is sure to cause problems -- use `OperandRef::store`
/// instead.
#[derive(Copy, Clone)]
pub struct OperandRef<'tcx, V> {
    /// The value.
    pub val: OperandValue<V>,

    /// The layout of value, based on its Rust type.
    pub layout: TyAndLayout<'tcx>,
}

impl<V: CodegenObject> fmt::Debug for OperandRef<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OperandRef({:?} @ {:?})", self.val, self.layout)
    }
}

impl<'a, 'tcx, V: CodegenObject> OperandRef<'tcx, V> {
    pub fn zero_sized(layout: TyAndLayout<'tcx>) -> OperandRef<'tcx, V> {
        assert!(layout.is_zst());
        OperandRef { val: OperandValue::ZeroSized, layout }
    }

    pub fn from_const<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        val: mir::ConstValue<'tcx>,
        ty: Ty<'tcx>,
    ) -> Self {
        let layout = bx.layout_of(ty);

        let val = match val {
            ConstValue::Scalar(x) => {
                let Abi::Scalar(scalar) = layout.abi else {
                    bug!("from_const: invalid ByVal layout: {:#?}", layout);
                };
                let llval = bx.scalar_to_backend(x, scalar, bx.immediate_backend_type(layout));
                OperandValue::Immediate(llval)
            }
            ConstValue::ZeroSized => return OperandRef::zero_sized(layout),
            ConstValue::Slice { data, meta } => {
                let Abi::ScalarPair(a_scalar, _) = layout.abi else {
                    bug!("from_const: invalid ScalarPair layout: {:#?}", layout);
                };
                let a = Scalar::from_pointer(
                    Pointer::new(bx.tcx().reserve_and_set_memory_alloc(data).into(), Size::ZERO),
                    &bx.tcx(),
                );
                let a_llval = bx.scalar_to_backend(
                    a,
                    a_scalar,
                    bx.scalar_pair_element_backend_type(layout, 0, true),
                );
                let b_llval = bx.const_usize(meta);
                OperandValue::Pair(a_llval, b_llval)
            }
            ConstValue::Indirect { alloc_id, offset } => {
                let alloc = bx.tcx().global_alloc(alloc_id).unwrap_memory();
                return Self::from_const_alloc(bx, layout, alloc, offset);
            }
        };

        OperandRef { val, layout }
    }

    fn from_const_alloc<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
        alloc: rustc_middle::mir::interpret::ConstAllocation<'tcx>,
        offset: Size,
    ) -> Self {
        let alloc_align = alloc.inner().align;
        assert!(alloc_align >= layout.align.abi);

        let read_scalar = |start, size, s: abi::Scalar, ty| {
            match alloc.0.read_scalar(
                bx,
                alloc_range(start, size),
                /*read_provenance*/ matches!(s.primitive(), abi::Pointer(_)),
            ) {
                Ok(val) => bx.scalar_to_backend(val, s, ty),
                Err(_) => bx.const_poison(ty),
            }
        };

        // It may seem like all types with `Scalar` or `ScalarPair` ABI are fair game at this point.
        // However, `MaybeUninit<u64>` is considered a `Scalar` as far as its layout is concerned --
        // and yet cannot be represented by an interpreter `Scalar`, since we have to handle the
        // case where some of the bytes are initialized and others are not. So, we need an extra
        // check that walks over the type of `mplace` to make sure it is truly correct to treat this
        // like a `Scalar` (or `ScalarPair`).
        match layout.abi {
            Abi::Scalar(s @ abi::Scalar::Initialized { .. }) => {
                let size = s.size(bx);
                assert_eq!(size, layout.size, "abi::Scalar size does not match layout size");
                let val = read_scalar(offset, size, s, bx.immediate_backend_type(layout));
                OperandRef { val: OperandValue::Immediate(val), layout }
            }
            Abi::ScalarPair(
                a @ abi::Scalar::Initialized { .. },
                b @ abi::Scalar::Initialized { .. },
            ) => {
                let (a_size, b_size) = (a.size(bx), b.size(bx));
                let b_offset = (offset + a_size).align_to(b.align(bx).abi);
                assert!(b_offset.bytes() > 0);
                let a_val = read_scalar(
                    offset,
                    a_size,
                    a,
                    bx.scalar_pair_element_backend_type(layout, 0, true),
                );
                let b_val = read_scalar(
                    b_offset,
                    b_size,
                    b,
                    bx.scalar_pair_element_backend_type(layout, 1, true),
                );
                OperandRef { val: OperandValue::Pair(a_val, b_val), layout }
            }
            _ if layout.is_zst() => OperandRef::zero_sized(layout),
            _ => {
                // Neither a scalar nor scalar pair. Load from a place
                // FIXME: should we cache `const_data_from_alloc` to avoid repeating this for the
                // same `ConstAllocation`?
                let init = bx.const_data_from_alloc(alloc);
                let base_addr = bx.static_addr_of(init, alloc_align, None);

                let llval = bx.const_ptr_byte_offset(base_addr, offset);
                bx.load_operand(PlaceRef::new_sized(llval, layout))
            }
        }
    }

    /// Asserts that this operand refers to a scalar and returns
    /// a reference to its value.
    pub fn immediate(self) -> V {
        match self.val {
            OperandValue::Immediate(s) => s,
            _ => bug!("not immediate: {:?}", self),
        }
    }

    /// Asserts that this operand is a pointer (or reference) and returns
    /// the place to which it points.  (This requires no code to be emitted
    /// as we represent places using the pointer to the place.)
    ///
    /// This uses [`Ty::builtin_deref`] to include the type of the place and
    /// assumes the place is aligned to the pointee's usual ABI alignment.
    ///
    /// If you don't need the type, see [`OperandValue::pointer_parts`]
    /// or [`OperandValue::deref`].
    pub fn deref<Cx: LayoutTypeMethods<'tcx>>(self, cx: &Cx) -> PlaceRef<'tcx, V> {
        if self.layout.ty.is_box() {
            // Derefer should have removed all Box derefs
            bug!("dereferencing {:?} in codegen", self.layout.ty);
        }

        let projected_ty = self
            .layout
            .ty
            .builtin_deref(true)
            .unwrap_or_else(|| bug!("deref of non-pointer {:?}", self));

        let layout = cx.layout_of(projected_ty);
        self.val.deref(layout.align.abi).with_type(layout)
    }

    /// If this operand is a `Pair`, we return an aggregate with the two values.
    /// For other cases, see `immediate`.
    pub fn immediate_or_packed_pair<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
    ) -> V {
        if let OperandValue::Pair(a, b) = self.val {
            let llty = bx.cx().immediate_backend_type(self.layout);
            debug!("Operand::immediate_or_packed_pair: packing {:?} into {:?}", self, llty);
            // Reconstruct the immediate aggregate.
            let mut llpair = bx.cx().const_poison(llty);
            llpair = bx.insert_value(llpair, a, 0);
            llpair = bx.insert_value(llpair, b, 1);
            llpair
        } else {
            self.immediate()
        }
    }

    /// If the type is a pair, we return a `Pair`, otherwise, an `Immediate`.
    pub fn from_immediate_or_packed_pair<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        llval: V,
        layout: TyAndLayout<'tcx>,
    ) -> Self {
        let val = if let Abi::ScalarPair(..) = layout.abi {
            debug!("Operand::from_immediate_or_packed_pair: unpacking {:?} @ {:?}", llval, layout);

            // Deconstruct the immediate aggregate.
            let a_llval = bx.extract_value(llval, 0);
            let b_llval = bx.extract_value(llval, 1);
            OperandValue::Pair(a_llval, b_llval)
        } else {
            OperandValue::Immediate(llval)
        };
        OperandRef { val, layout }
    }

    pub fn extract_field<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        &self,
        bx: &mut Bx,
        i: usize,
    ) -> Self {
        let field = self.layout.field(bx.cx(), i);
        let offset = self.layout.fields.offset(i);

        let mut val = match (self.val, self.layout.abi) {
            // If the field is ZST, it has no data.
            _ if field.is_zst() => OperandValue::ZeroSized,

            // Newtype of a scalar, scalar pair or vector.
            (OperandValue::Immediate(_) | OperandValue::Pair(..), _)
                if field.size == self.layout.size =>
            {
                assert_eq!(offset.bytes(), 0);
                self.val
            }

            // Extract a scalar component from a pair.
            (OperandValue::Pair(a_llval, b_llval), Abi::ScalarPair(a, b)) => {
                if offset.bytes() == 0 {
                    assert_eq!(field.size, a.size(bx.cx()));
                    OperandValue::Immediate(a_llval)
                } else {
                    assert_eq!(offset, a.size(bx.cx()).align_to(b.align(bx.cx()).abi));
                    assert_eq!(field.size, b.size(bx.cx()));
                    OperandValue::Immediate(b_llval)
                }
            }

            // `#[repr(simd)]` types are also immediate.
            (OperandValue::Immediate(llval), Abi::Vector { .. }) => {
                OperandValue::Immediate(bx.extract_element(llval, bx.cx().const_usize(i as u64)))
            }

            _ => bug!("OperandRef::extract_field({:?}): not applicable", self),
        };

        match (&mut val, field.abi) {
            (OperandValue::ZeroSized, _) => {}
            (
                OperandValue::Immediate(llval),
                Abi::Scalar(_) | Abi::ScalarPair(..) | Abi::Vector { .. },
            ) => {
                // Bools in union fields needs to be truncated.
                *llval = bx.to_immediate(*llval, field);
            }
            (OperandValue::Pair(a, b), Abi::ScalarPair(a_abi, b_abi)) => {
                // Bools in union fields needs to be truncated.
                *a = bx.to_immediate_scalar(*a, a_abi);
                *b = bx.to_immediate_scalar(*b, b_abi);
            }
            // Newtype vector of array, e.g. #[repr(simd)] struct S([i32; 4]);
            (OperandValue::Immediate(llval), Abi::Aggregate { sized: true }) => {
                assert!(matches!(self.layout.abi, Abi::Vector { .. }));

                let llfield_ty = bx.cx().backend_type(field);

                // Can't bitcast an aggregate, so round trip through memory.
                let llptr = bx.alloca(field.size, field.align.abi);
                bx.store(*llval, llptr, field.align.abi);
                *llval = bx.load(llfield_ty, llptr, field.align.abi);
            }
            (OperandValue::Immediate(_), Abi::Uninhabited | Abi::Aggregate { sized: false }) => {
                bug!()
            }
            (OperandValue::Pair(..), _) => bug!(),
            (OperandValue::Ref(..), _) => bug!(),
        }

        OperandRef { val, layout: field }
    }
}

impl<'a, 'tcx, V: CodegenObject> OperandValue<V> {
    /// Returns an `OperandValue` that's generally UB to use in any way.
    ///
    /// Depending on the `layout`, returns `ZeroSized` for ZSTs, an `Immediate` or
    /// `Pair` containing poison value(s), or a `Ref` containing a poison pointer.
    ///
    /// Supports sized types only.
    pub fn poison<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
    ) -> OperandValue<V> {
        assert!(layout.is_sized());
        if layout.is_zst() {
            OperandValue::ZeroSized
        } else if bx.cx().is_backend_immediate(layout) {
            let ibty = bx.cx().immediate_backend_type(layout);
            OperandValue::Immediate(bx.const_poison(ibty))
        } else if bx.cx().is_backend_scalar_pair(layout) {
            let ibty0 = bx.cx().scalar_pair_element_backend_type(layout, 0, true);
            let ibty1 = bx.cx().scalar_pair_element_backend_type(layout, 1, true);
            OperandValue::Pair(bx.const_poison(ibty0), bx.const_poison(ibty1))
        } else {
            let ptr = bx.cx().type_ptr();
            OperandValue::Ref(PlaceValue::new_sized(bx.const_poison(ptr), layout.align.abi))
        }
    }

    pub fn store<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        dest: PlaceRef<'tcx, V>,
    ) {
        self.store_with_flags(bx, dest, MemFlags::empty());
    }

    pub fn volatile_store<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        dest: PlaceRef<'tcx, V>,
    ) {
        self.store_with_flags(bx, dest, MemFlags::VOLATILE);
    }

    pub fn unaligned_volatile_store<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        dest: PlaceRef<'tcx, V>,
    ) {
        self.store_with_flags(bx, dest, MemFlags::VOLATILE | MemFlags::UNALIGNED);
    }

    pub fn nontemporal_store<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        dest: PlaceRef<'tcx, V>,
    ) {
        self.store_with_flags(bx, dest, MemFlags::NONTEMPORAL);
    }

    pub(crate) fn store_with_flags<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        dest: PlaceRef<'tcx, V>,
        flags: MemFlags,
    ) {
        debug!("OperandRef::store: operand={:?}, dest={:?}", self, dest);
        match self {
            OperandValue::ZeroSized => {
                // Avoid generating stores of zero-sized values, because the only way to have a zero-sized
                // value is through `undef`/`poison`, and the store itself is useless.
            }
            OperandValue::Ref(val) => {
                assert!(dest.layout.is_sized(), "cannot directly store unsized values");
                if val.llextra.is_some() {
                    bug!("cannot directly store unsized values");
                }
                bx.typed_place_copy_with_flags(dest.val, val, dest.layout, flags);
            }
            OperandValue::Immediate(s) => {
                let val = bx.from_immediate(s);
                bx.store_with_flags(val, dest.val.llval, dest.val.align, flags);
            }
            OperandValue::Pair(a, b) => {
                let Abi::ScalarPair(a_scalar, b_scalar) = dest.layout.abi else {
                    bug!("store_with_flags: invalid ScalarPair layout: {:#?}", dest.layout);
                };
                let b_offset = a_scalar.size(bx).align_to(b_scalar.align(bx).abi);

                let val = bx.from_immediate(a);
                let align = dest.val.align;
                bx.store_with_flags(val, dest.val.llval, align, flags);

                let llptr = bx.inbounds_ptradd(dest.val.llval, bx.const_usize(b_offset.bytes()));
                let val = bx.from_immediate(b);
                let align = dest.val.align.restrict_for_offset(b_offset);
                bx.store_with_flags(val, llptr, align, flags);
            }
        }
    }

    pub fn store_unsized<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        indirect_dest: PlaceRef<'tcx, V>,
    ) {
        debug!("OperandRef::store_unsized: operand={:?}, indirect_dest={:?}", self, indirect_dest);
        // `indirect_dest` must have `*mut T` type. We extract `T` out of it.
        let unsized_ty = indirect_dest
            .layout
            .ty
            .builtin_deref(true)
            .unwrap_or_else(|| bug!("indirect_dest has non-pointer type: {:?}", indirect_dest));

        let OperandValue::Ref(PlaceValue { llval: llptr, llextra: Some(llextra), .. }) = self
        else {
            bug!("store_unsized called with a sized value (or with an extern type)")
        };

        // Allocate an appropriate region on the stack, and copy the value into it. Since alloca
        // doesn't support dynamic alignment, we allocate an extra align - 1 bytes, and align the
        // pointer manually.
        let (size, align) = size_of_val::size_and_align_of_dst(bx, unsized_ty, Some(llextra));
        let one = bx.const_usize(1);
        let align_minus_1 = bx.sub(align, one);
        let size_extra = bx.add(size, align_minus_1);
        let min_align = Align::ONE;
        let alloca = bx.dynamic_alloca(size_extra, min_align);
        let address = bx.ptrtoint(alloca, bx.type_isize());
        let neg_address = bx.neg(address);
        let offset = bx.and(neg_address, align_minus_1);
        let dst = bx.inbounds_ptradd(alloca, offset);
        bx.memcpy(dst, min_align, llptr, min_align, size, MemFlags::empty());

        // Store the allocated region and the extra to the indirect place.
        let indirect_operand = OperandValue::Pair(dst, llextra);
        indirect_operand.store(bx, indirect_dest);
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    fn maybe_codegen_consume_direct(
        &mut self,
        bx: &mut Bx,
        place_ref: mir::PlaceRef<'tcx>,
    ) -> Option<OperandRef<'tcx, Bx::Value>> {
        debug!("maybe_codegen_consume_direct(place_ref={:?})", place_ref);

        match self.locals[place_ref.local] {
            LocalRef::Operand(mut o) => {
                // Moves out of scalar and scalar pair fields are trivial.
                for elem in place_ref.projection.iter() {
                    match elem {
                        mir::ProjectionElem::Field(ref f, _) => {
                            assert!(
                                !o.layout.ty.is_any_ptr(),
                                "Bad PlaceRef: destructing pointers should use cast/PtrMetadata, \
                                 but tried to access field {f:?} of pointer {o:?}",
                            );
                            o = o.extract_field(bx, f.index());
                        }
                        mir::ProjectionElem::Index(_)
                        | mir::ProjectionElem::ConstantIndex { .. } => {
                            // ZSTs don't require any actual memory access.
                            // FIXME(eddyb) deduplicate this with the identical
                            // checks in `codegen_consume` and `extract_field`.
                            let elem = o.layout.field(bx.cx(), 0);
                            if elem.is_zst() {
                                o = OperandRef::zero_sized(elem);
                            } else {
                                return None;
                            }
                        }
                        _ => return None,
                    }
                }

                Some(o)
            }
            LocalRef::PendingOperand => {
                bug!("use of {:?} before def", place_ref);
            }
            LocalRef::Place(..) | LocalRef::UnsizedPlace(..) => {
                // watch out for locals that do not have an
                // alloca; they are handled somewhat differently
                None
            }
        }
    }

    pub fn codegen_consume(
        &mut self,
        bx: &mut Bx,
        place_ref: mir::PlaceRef<'tcx>,
    ) -> OperandRef<'tcx, Bx::Value> {
        debug!("codegen_consume(place_ref={:?})", place_ref);

        let ty = self.monomorphized_place_ty(place_ref);
        let layout = bx.cx().layout_of(ty);

        // ZSTs don't require any actual memory access.
        if layout.is_zst() {
            return OperandRef::zero_sized(layout);
        }

        if let Some(o) = self.maybe_codegen_consume_direct(bx, place_ref) {
            return o;
        }

        // for most places, to consume them we just load them
        // out from their home
        let place = self.codegen_place(bx, place_ref);
        bx.load_operand(place)
    }

    pub fn codegen_operand(
        &mut self,
        bx: &mut Bx,
        operand: &mir::Operand<'tcx>,
    ) -> OperandRef<'tcx, Bx::Value> {
        debug!("codegen_operand(operand={:?})", operand);

        match *operand {
            mir::Operand::Copy(ref place) | mir::Operand::Move(ref place) => {
                self.codegen_consume(bx, place.as_ref())
            }

            mir::Operand::Constant(ref constant) => self.eval_mir_constant_to_operand(bx, constant),
        }
    }
}
