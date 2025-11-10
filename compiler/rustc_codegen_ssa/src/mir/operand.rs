use std::fmt;

use itertools::Either;
use rustc_abi as abi;
use rustc_abi::{
    Align, BackendRepr, FIRST_VARIANT, FieldIdx, Primitive, Size, TagEncoding, VariantIdx, Variants,
};
use rustc_hir::LangItem;
use rustc_middle::mir::interpret::{Pointer, Scalar, alloc_range};
use rustc_middle::mir::{self, ConstValue};
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, Ty};
use rustc_middle::{bug, span_bug};
use rustc_session::config::{AnnotateMoves, DebugInfo, OptLevel};
use tracing::{debug, instrument};

use super::place::{PlaceRef, PlaceValue};
use super::rvalue::transmute_scalar;
use super::{FunctionCx, LocalRef};
use crate::MemFlags;
use crate::common::IntPredicate;
use crate::traits::*;

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
    /// [`LayoutTypeCodegenMethods::is_backend_ref`] returns `true`.
    /// (That basically amounts to "isn't one of the other variants".)
    ///
    /// This holds a [`PlaceValue`] (like a [`PlaceRef`] does) with a pointer
    /// to the location holding the value. The type behind that pointer is the
    /// one returned by [`LayoutTypeCodegenMethods::backend_type`].
    Ref(PlaceValue<V>),
    /// A single LLVM immediate value.
    ///
    /// An `OperandValue` *must* be this variant for any type for which
    /// [`LayoutTypeCodegenMethods::is_backend_immediate`] returns `true`.
    /// The backend value in this variant must be the *immediate* backend type,
    /// as returned by [`LayoutTypeCodegenMethods::immediate_backend_type`].
    Immediate(V),
    /// A pair of immediate LLVM values. Used by wide pointers too.
    ///
    /// # Invariants
    /// - For `Pair(a, b)`, `a` is always at offset 0, but may have `FieldIdx(1..)`
    /// - `b` is not at offset 0, because `V` is not a 1ZST type.
    /// - `a` and `b` will have a different FieldIdx, but otherwise `b`'s may be lower
    ///   or they may not be adjacent, due to arbitrary numbers of 1ZST fields that
    ///   will not affect the shape of the data which determines if `Pair` will be used.
    /// - An `OperandValue` *must* be this variant for any type for which
    /// [`LayoutTypeCodegenMethods::is_backend_scalar_pair`] returns `true`.
    /// - The backend values in this variant must be the *immediate* backend types,
    /// as returned by [`LayoutTypeCodegenMethods::scalar_pair_element_backend_type`]
    /// with `immediate: true`.
    Pair(V, V),
    /// A value taking no bytes, and which therefore needs no LLVM value at all.
    ///
    /// If you ever need a `V` to pass to something, get a fresh poison value
    /// from [`ConstCodegenMethods::const_poison`].
    ///
    /// An `OperandValue` *must* be this variant for any type for which
    /// `is_zst` on its `Layout` returns `true`. Note however that
    /// these values can still require alignment.
    ZeroSized,
}

impl<V: CodegenObject> OperandValue<V> {
    /// Return the data pointer and optional metadata as backend values
    /// if this value can be treat as a pointer.
    pub(crate) fn try_pointer_parts(self) -> Option<(V, Option<V>)> {
        match self {
            OperandValue::Immediate(llptr) => Some((llptr, None)),
            OperandValue::Pair(llptr, llextra) => Some((llptr, Some(llextra))),
            OperandValue::Ref(_) | OperandValue::ZeroSized => None,
        }
    }

    /// Treat this value as a pointer and return the data pointer and
    /// optional metadata as backend values.
    ///
    /// If you're making a place, use [`Self::deref`] instead.
    pub(crate) fn pointer_parts(self) -> (V, Option<V>) {
        self.try_pointer_parts()
            .unwrap_or_else(|| bug!("OperandValue cannot be a pointer: {self:?}"))
    }

    /// Treat this value as a pointer and return the place to which it points.
    ///
    /// The pointer immediate doesn't inherently know its alignment,
    /// so you need to pass it in. If you want to get it from a type's ABI
    /// alignment, then maybe you want [`OperandRef::deref`] instead.
    ///
    /// This is the inverse of [`PlaceValue::address`].
    pub(crate) fn deref(self, align: Align) -> PlaceValue<V> {
        let (llval, llextra) = self.pointer_parts();
        PlaceValue { llval, llextra, align }
    }

    pub(crate) fn is_expected_variant_for_type<'tcx, Cx: LayoutTypeCodegenMethods<'tcx>>(
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

    /// Annotation for profiler visibility of move/copy operations.
    /// When set, the store operation should appear as an inlined call to this function.
    pub move_annotation: Option<ty::Instance<'tcx>>,
}

impl<V: CodegenObject> fmt::Debug for OperandRef<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OperandRef({:?} @ {:?})", self.val, self.layout)
    }
}

impl<'a, 'tcx, V: CodegenObject> OperandRef<'tcx, V> {
    pub fn zero_sized(layout: TyAndLayout<'tcx>) -> OperandRef<'tcx, V> {
        assert!(layout.is_zst());
        OperandRef { val: OperandValue::ZeroSized, layout, move_annotation: None }
    }

    pub(crate) fn from_const<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        val: mir::ConstValue,
        ty: Ty<'tcx>,
    ) -> Self {
        let layout = bx.layout_of(ty);

        let val = match val {
            ConstValue::Scalar(x) => {
                let BackendRepr::Scalar(scalar) = layout.backend_repr else {
                    bug!("from_const: invalid ByVal layout: {:#?}", layout);
                };
                let llval = bx.scalar_to_backend(x, scalar, bx.immediate_backend_type(layout));
                OperandValue::Immediate(llval)
            }
            ConstValue::ZeroSized => return OperandRef::zero_sized(layout),
            ConstValue::Slice { alloc_id, meta } => {
                let BackendRepr::ScalarPair(a_scalar, _) = layout.backend_repr else {
                    bug!("from_const: invalid ScalarPair layout: {:#?}", layout);
                };
                let a = Scalar::from_pointer(Pointer::new(alloc_id.into(), Size::ZERO), &bx.tcx());
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

        OperandRef { val, layout, move_annotation: None }
    }

    fn from_const_alloc<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
        alloc: rustc_middle::mir::interpret::ConstAllocation<'tcx>,
        offset: Size,
    ) -> Self {
        let alloc_align = alloc.inner().align;
        assert!(alloc_align >= layout.align.abi, "{alloc_align:?} < {:?}", layout.align.abi);

        let read_scalar = |start, size, s: abi::Scalar, ty| {
            match alloc.0.read_scalar(
                bx,
                alloc_range(start, size),
                /*read_provenance*/ matches!(s.primitive(), abi::Primitive::Pointer(_)),
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
        match layout.backend_repr {
            BackendRepr::Scalar(s @ abi::Scalar::Initialized { .. }) => {
                let size = s.size(bx);
                assert_eq!(size, layout.size, "abi::Scalar size does not match layout size");
                let val = read_scalar(offset, size, s, bx.immediate_backend_type(layout));
                OperandRef { val: OperandValue::Immediate(val), layout, move_annotation: None }
            }
            BackendRepr::ScalarPair(
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
                OperandRef { val: OperandValue::Pair(a_val, b_val), layout, move_annotation: None }
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
    pub fn deref<Cx: CodegenMethods<'tcx>>(self, cx: &Cx) -> PlaceRef<'tcx, V> {
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

    /// Store this operand into a place, applying move/copy annotation if present.
    ///
    /// This is the preferred method for storing operands, as it automatically
    /// applies profiler annotations for tracked move/copy operations.
    pub fn store_with_annotation<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        dest: PlaceRef<'tcx, V>,
    ) {
        if let Some(instance) = self.move_annotation {
            bx.with_move_annotation(instance, |bx| self.val.store(bx, dest))
        } else {
            self.val.store(bx, dest)
        }
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
        let val = if let BackendRepr::ScalarPair(..) = layout.backend_repr {
            debug!("Operand::from_immediate_or_packed_pair: unpacking {:?} @ {:?}", llval, layout);

            // Deconstruct the immediate aggregate.
            let a_llval = bx.extract_value(llval, 0);
            let b_llval = bx.extract_value(llval, 1);
            OperandValue::Pair(a_llval, b_llval)
        } else {
            OperandValue::Immediate(llval)
        };
        OperandRef { val, layout, move_annotation: None }
    }

    pub(crate) fn extract_field<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        &self,
        fx: &mut FunctionCx<'a, 'tcx, Bx>,
        bx: &mut Bx,
        i: usize,
    ) -> Self {
        let field = self.layout.field(bx.cx(), i);
        let offset = self.layout.fields.offset(i);

        if !bx.is_backend_ref(self.layout) && bx.is_backend_ref(field) {
            // Part of https://github.com/rust-lang/compiler-team/issues/838
            span_bug!(
                fx.mir.span,
                "Non-ref type {self:?} cannot project to ref field type {field:?}",
            );
        }

        let val = if field.is_zst() {
            OperandValue::ZeroSized
        } else if field.size == self.layout.size {
            assert_eq!(offset.bytes(), 0);
            fx.codegen_transmute_operand(bx, *self, field)
        } else {
            let (in_scalar, imm) = match (self.val, self.layout.backend_repr) {
                // Extract a scalar component from a pair.
                (OperandValue::Pair(a_llval, b_llval), BackendRepr::ScalarPair(a, b)) => {
                    if offset.bytes() == 0 {
                        assert_eq!(field.size, a.size(bx.cx()));
                        (Some(a), a_llval)
                    } else {
                        assert_eq!(offset, a.size(bx.cx()).align_to(b.align(bx.cx()).abi));
                        assert_eq!(field.size, b.size(bx.cx()));
                        (Some(b), b_llval)
                    }
                }

                _ => {
                    span_bug!(fx.mir.span, "OperandRef::extract_field({:?}): not applicable", self)
                }
            };
            OperandValue::Immediate(match field.backend_repr {
                BackendRepr::SimdVector { .. } => imm,
                BackendRepr::Scalar(out_scalar) => {
                    let Some(in_scalar) = in_scalar else {
                        span_bug!(
                            fx.mir.span,
                            "OperandRef::extract_field({:?}): missing input scalar for output scalar",
                            self
                        )
                    };
                    if in_scalar != out_scalar {
                        // If the backend and backend_immediate types might differ,
                        // flip back to the backend type then to the new immediate.
                        // This avoids nop truncations, but still handles things like
                        // Bools in union fields needs to be truncated.
                        let backend = bx.from_immediate(imm);
                        bx.to_immediate_scalar(backend, out_scalar)
                    } else {
                        imm
                    }
                }
                BackendRepr::ScalarPair(_, _) | BackendRepr::Memory { .. } => bug!(),
            })
        };

        OperandRef { val, layout: field, move_annotation: None }
    }

    /// Obtain the actual discriminant of a value.
    #[instrument(level = "trace", skip(fx, bx))]
    pub fn codegen_get_discr<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        fx: &mut FunctionCx<'a, 'tcx, Bx>,
        bx: &mut Bx,
        cast_to: Ty<'tcx>,
    ) -> V {
        let dl = &bx.tcx().data_layout;
        let cast_to_layout = bx.cx().layout_of(cast_to);
        let cast_to = bx.cx().immediate_backend_type(cast_to_layout);

        // We check uninhabitedness separately because a type like
        // `enum Foo { Bar(i32, !) }` is still reported as `Variants::Single`,
        // *not* as `Variants::Empty`.
        if self.layout.is_uninhabited() {
            return bx.cx().const_poison(cast_to);
        }

        let (tag_scalar, tag_encoding, tag_field) = match self.layout.variants {
            Variants::Empty => unreachable!("we already handled uninhabited types"),
            Variants::Single { index } => {
                let discr_val =
                    if let Some(discr) = self.layout.ty.discriminant_for_variant(bx.tcx(), index) {
                        discr.val
                    } else {
                        // This arm is for types which are neither enums nor coroutines,
                        // and thus for which the only possible "variant" should be the first one.
                        assert_eq!(index, FIRST_VARIANT);
                        // There's thus no actual discriminant to return, so we return
                        // what it would have been if this was a single-variant enum.
                        0
                    };
                return bx.cx().const_uint_big(cast_to, discr_val);
            }
            Variants::Multiple { tag, ref tag_encoding, tag_field, .. } => {
                (tag, tag_encoding, tag_field)
            }
        };

        // Read the tag/niche-encoded discriminant from memory.
        let tag_op = match self.val {
            OperandValue::ZeroSized => bug!(),
            OperandValue::Immediate(_) | OperandValue::Pair(_, _) => {
                self.extract_field(fx, bx, tag_field.as_usize())
            }
            OperandValue::Ref(place) => {
                let tag = place.with_type(self.layout).project_field(bx, tag_field.as_usize());
                bx.load_operand(tag)
            }
        };
        let tag_imm = tag_op.immediate();

        // Decode the discriminant (specifically if it's niche-encoded).
        match *tag_encoding {
            TagEncoding::Direct => {
                let signed = match tag_scalar.primitive() {
                    // We use `i1` for bytes that are always `0` or `1`,
                    // e.g., `#[repr(i8)] enum E { A, B }`, but we can't
                    // let LLVM interpret the `i1` as signed, because
                    // then `i1 1` (i.e., `E::B`) is effectively `i8 -1`.
                    Primitive::Int(_, signed) => !tag_scalar.is_bool() && signed,
                    _ => false,
                };
                bx.intcast(tag_imm, cast_to, signed)
            }
            TagEncoding::Niche { untagged_variant, ref niche_variants, niche_start } => {
                // Cast to an integer so we don't have to treat a pointer as a
                // special case.
                let (tag, tag_llty) = match tag_scalar.primitive() {
                    // FIXME(erikdesjardins): handle non-default addrspace ptr sizes
                    Primitive::Pointer(_) => {
                        let t = bx.type_from_integer(dl.ptr_sized_integer());
                        let tag = bx.ptrtoint(tag_imm, t);
                        (tag, t)
                    }
                    _ => (tag_imm, bx.cx().immediate_backend_type(tag_op.layout)),
                };

                // `layout_sanity_check` ensures that we only get here for cases where the discriminant
                // value and the variant index match, since that's all `Niche` can encode.

                let relative_max = niche_variants.end().as_u32() - niche_variants.start().as_u32();
                let niche_start_const = bx.cx().const_uint_big(tag_llty, niche_start);

                // We have a subrange `niche_start..=niche_end` inside `range`.
                // If the value of the tag is inside this subrange, it's a
                // "niche value", an increment of the discriminant. Otherwise it
                // indicates the untagged variant.
                // A general algorithm to extract the discriminant from the tag
                // is:
                // relative_tag = tag - niche_start
                // is_niche = relative_tag <= (ule) relative_max
                // discr = if is_niche {
                //     cast(relative_tag) + niche_variants.start()
                // } else {
                //     untagged_variant
                // }
                // However, we will likely be able to emit simpler code.
                let (is_niche, tagged_discr, delta) = if relative_max == 0 {
                    // Best case scenario: only one tagged variant. This will
                    // likely become just a comparison and a jump.
                    // The algorithm is:
                    // is_niche = tag == niche_start
                    // discr = if is_niche {
                    //     niche_start
                    // } else {
                    //     untagged_variant
                    // }
                    let is_niche = bx.icmp(IntPredicate::IntEQ, tag, niche_start_const);
                    let tagged_discr =
                        bx.cx().const_uint(cast_to, niche_variants.start().as_u32() as u64);
                    (is_niche, tagged_discr, 0)
                } else {
                    // Thanks to parameter attributes and load metadata, LLVM already knows
                    // the general valid range of the tag. It's possible, though, for there
                    // to be an impossible value *in the middle*, which those ranges don't
                    // communicate, so it's worth an `assume` to let the optimizer know.
                    // Most importantly, this means when optimizing a variant test like
                    // `SELECT(is_niche, complex, CONST) == CONST` it's ok to simplify that
                    // to `!is_niche` because the `complex` part can't possibly match.
                    //
                    // This was previously asserted on `tagged_discr` below, where the
                    // impossible value is more obvious, but that caused an intermediate
                    // value to become multi-use and thus not optimize, so instead this
                    // assumes on the original input which is always multi-use. See
                    // <https://github.com/llvm/llvm-project/issues/134024#issuecomment-3131782555>
                    //
                    // FIXME: If we ever get range assume operand bundles in LLVM (so we
                    // don't need the `icmp`s in the instruction stream any more), it
                    // might be worth moving this back to being on the switch argument
                    // where it's more obviously applicable.
                    if niche_variants.contains(&untagged_variant)
                        && bx.cx().sess().opts.optimize != OptLevel::No
                    {
                        let impossible = niche_start
                            .wrapping_add(u128::from(untagged_variant.as_u32()))
                            .wrapping_sub(u128::from(niche_variants.start().as_u32()));
                        let impossible = bx.cx().const_uint_big(tag_llty, impossible);
                        let ne = bx.icmp(IntPredicate::IntNE, tag, impossible);
                        bx.assume(ne);
                    }

                    // With multiple niched variants we'll have to actually compute
                    // the variant index from the stored tag.
                    //
                    // However, there's still one small optimization we can often do for
                    // determining *whether* a tag value is a natural value or a niched
                    // variant. The general algorithm involves a subtraction that often
                    // wraps in practice, making it tricky to analyse. However, in cases
                    // where there are few enough possible values of the tag that it doesn't
                    // need to wrap around, we can instead just look for the contiguous
                    // tag values on the end of the range with a single comparison.
                    //
                    // For example, take the type `enum Demo { A, B, Untagged(bool) }`.
                    // The `bool` is {0, 1}, and the two other variants are given the
                    // tags {2, 3} respectively. That means the `tag_range` is
                    // `[0, 3]`, which doesn't wrap as unsigned (nor as signed), so
                    // we can test for the niched variants with just `>= 2`.
                    //
                    // That means we're looking either for the niche values *above*
                    // the natural values of the untagged variant:
                    //
                    //             niche_start                  niche_end
                    //                  |                           |
                    //                  v                           v
                    // MIN -------------+---------------------------+---------- MAX
                    //         ^        |         is niche          |
                    //         |        +---------------------------+
                    //         |                                    |
                    //   tag_range.start                      tag_range.end
                    //
                    // Or *below* the natural values:
                    //
                    //    niche_start              niche_end
                    //         |                       |
                    //         v                       v
                    // MIN ----+-----------------------+---------------------- MAX
                    //         |       is niche        |           ^
                    //         +-----------------------+           |
                    //         |                                   |
                    //   tag_range.start                      tag_range.end
                    //
                    // With those two options and having the flexibility to choose
                    // between a signed or unsigned comparison on the tag, that
                    // covers most realistic scenarios. The tests have a (contrived)
                    // example of a 1-byte enum with over 128 niched variants which
                    // wraps both as signed as unsigned, though, and for something
                    // like that we're stuck with the general algorithm.

                    let tag_range = tag_scalar.valid_range(&dl);
                    let tag_size = tag_scalar.size(&dl);
                    let niche_end = u128::from(relative_max).wrapping_add(niche_start);
                    let niche_end = tag_size.truncate(niche_end);

                    let relative_discr = bx.sub(tag, niche_start_const);
                    let cast_tag = bx.intcast(relative_discr, cast_to, false);
                    let is_niche = if tag_range.no_unsigned_wraparound(tag_size) == Ok(true) {
                        if niche_start == tag_range.start {
                            let niche_end_const = bx.cx().const_uint_big(tag_llty, niche_end);
                            bx.icmp(IntPredicate::IntULE, tag, niche_end_const)
                        } else {
                            assert_eq!(niche_end, tag_range.end);
                            bx.icmp(IntPredicate::IntUGE, tag, niche_start_const)
                        }
                    } else if tag_range.no_signed_wraparound(tag_size) == Ok(true) {
                        if niche_start == tag_range.start {
                            let niche_end_const = bx.cx().const_uint_big(tag_llty, niche_end);
                            bx.icmp(IntPredicate::IntSLE, tag, niche_end_const)
                        } else {
                            assert_eq!(niche_end, tag_range.end);
                            bx.icmp(IntPredicate::IntSGE, tag, niche_start_const)
                        }
                    } else {
                        bx.icmp(
                            IntPredicate::IntULE,
                            relative_discr,
                            bx.cx().const_uint(tag_llty, relative_max as u64),
                        )
                    };

                    (is_niche, cast_tag, niche_variants.start().as_u32() as u128)
                };

                let tagged_discr = if delta == 0 {
                    tagged_discr
                } else {
                    bx.add(tagged_discr, bx.cx().const_uint_big(cast_to, delta))
                };

                let untagged_variant_const =
                    bx.cx().const_uint(cast_to, u64::from(untagged_variant.as_u32()));

                let discr = bx.select(is_niche, tagged_discr, untagged_variant_const);

                // In principle we could insert assumes on the possible range of `discr`, but
                // currently in LLVM this isn't worth it because the original `tag` will
                // have either a `range` parameter attribute or `!range` metadata,
                // or come from a `transmute` that already `assume`d it.

                discr
            }
        }
    }
}

/// Each of these variants starts out as `Either::Right` when it's uninitialized,
/// then setting the field changes that to `Either::Left` with the backend value.
#[derive(Debug, Copy, Clone)]
enum OperandValueBuilder<V> {
    ZeroSized,
    Immediate(Either<V, abi::Scalar>),
    Pair(Either<V, abi::Scalar>, Either<V, abi::Scalar>),
    /// `repr(simd)` types need special handling because they each have a non-empty
    /// array field (which uses [`OperandValue::Ref`]) despite the SIMD type itself
    /// using [`OperandValue::Immediate`] which for any other kind of type would
    /// mean that its one non-ZST field would also be [`OperandValue::Immediate`].
    Vector(Either<V, ()>),
}

/// Allows building up an `OperandRef` by setting fields one at a time.
#[derive(Debug, Copy, Clone)]
pub(super) struct OperandRefBuilder<'tcx, V> {
    val: OperandValueBuilder<V>,
    layout: TyAndLayout<'tcx>,
}

impl<'a, 'tcx, V: CodegenObject> OperandRefBuilder<'tcx, V> {
    /// Creates an uninitialized builder for an instance of the `layout`.
    ///
    /// ICEs for [`BackendRepr::Memory`] types (other than ZSTs), which should
    /// be built up inside a [`PlaceRef`] instead as they need an allocated place
    /// into which to write the values of the fields.
    pub(super) fn new(layout: TyAndLayout<'tcx>) -> Self {
        let val = match layout.backend_repr {
            BackendRepr::Memory { .. } if layout.is_zst() => OperandValueBuilder::ZeroSized,
            BackendRepr::Scalar(s) => OperandValueBuilder::Immediate(Either::Right(s)),
            BackendRepr::ScalarPair(a, b) => {
                OperandValueBuilder::Pair(Either::Right(a), Either::Right(b))
            }
            BackendRepr::SimdVector { .. } => OperandValueBuilder::Vector(Either::Right(())),
            BackendRepr::Memory { .. } => {
                bug!("Cannot use non-ZST Memory-ABI type in operand builder: {layout:?}");
            }
        };
        OperandRefBuilder { val, layout }
    }

    pub(super) fn insert_field<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        &mut self,
        bx: &mut Bx,
        variant: VariantIdx,
        field: FieldIdx,
        field_operand: OperandRef<'tcx, V>,
    ) {
        if let OperandValue::ZeroSized = field_operand.val {
            // A ZST never adds any state, so just ignore it.
            // This special-casing is worth it because of things like
            // `Result<!, !>` where `Ok(never)` is legal to write,
            // but the type shows as FieldShape::Primitive so we can't
            // actually look at the layout for the field being set.
            return;
        }

        let is_zero_offset = if let abi::FieldsShape::Primitive = self.layout.fields {
            // The other branch looking at field layouts ICEs for primitives,
            // so we need to handle them separately.
            // Because we handled ZSTs above (like the metadata in a thin pointer),
            // the only possibility is that we're setting the one-and-only field.
            assert!(!self.layout.is_zst());
            assert_eq!(variant, FIRST_VARIANT);
            assert_eq!(field, FieldIdx::ZERO);
            true
        } else {
            let variant_layout = self.layout.for_variant(bx.cx(), variant);
            let field_offset = variant_layout.fields.offset(field.as_usize());
            field_offset == Size::ZERO
        };

        let mut update = |tgt: &mut Either<V, abi::Scalar>, src, from_scalar| {
            let to_scalar = tgt.unwrap_right();
            // We transmute here (rather than just `from_immediate`) because in
            // `Result<usize, *const ()>` the field of the `Ok` is an integer,
            // but the corresponding scalar in the enum is a pointer.
            let imm = transmute_scalar(bx, src, from_scalar, to_scalar);
            *tgt = Either::Left(imm);
        };

        match (field_operand.val, field_operand.layout.backend_repr) {
            (OperandValue::ZeroSized, _) => unreachable!("Handled above"),
            (OperandValue::Immediate(v), BackendRepr::Scalar(from_scalar)) => match &mut self.val {
                OperandValueBuilder::Immediate(val @ Either::Right(_)) if is_zero_offset => {
                    update(val, v, from_scalar);
                }
                OperandValueBuilder::Pair(fst @ Either::Right(_), _) if is_zero_offset => {
                    update(fst, v, from_scalar);
                }
                OperandValueBuilder::Pair(_, snd @ Either::Right(_)) if !is_zero_offset => {
                    update(snd, v, from_scalar);
                }
                _ => {
                    bug!("Tried to insert {field_operand:?} into {variant:?}.{field:?} of {self:?}")
                }
            },
            (OperandValue::Immediate(v), BackendRepr::SimdVector { .. }) => match &mut self.val {
                OperandValueBuilder::Vector(val @ Either::Right(())) if is_zero_offset => {
                    *val = Either::Left(v);
                }
                _ => {
                    bug!("Tried to insert {field_operand:?} into {variant:?}.{field:?} of {self:?}")
                }
            },
            (OperandValue::Pair(a, b), BackendRepr::ScalarPair(from_sa, from_sb)) => {
                match &mut self.val {
                    OperandValueBuilder::Pair(fst @ Either::Right(_), snd @ Either::Right(_)) => {
                        update(fst, a, from_sa);
                        update(snd, b, from_sb);
                    }
                    _ => bug!(
                        "Tried to insert {field_operand:?} into {variant:?}.{field:?} of {self:?}"
                    ),
                }
            }
            (OperandValue::Ref(place), BackendRepr::Memory { .. }) => match &mut self.val {
                OperandValueBuilder::Vector(val @ Either::Right(())) => {
                    let ibty = bx.cx().immediate_backend_type(self.layout);
                    let simd = bx.load_from_place(ibty, place);
                    *val = Either::Left(simd);
                }
                _ => {
                    bug!("Tried to insert {field_operand:?} into {variant:?}.{field:?} of {self:?}")
                }
            },
            _ => bug!("Operand cannot be used with `insert_field`: {field_operand:?}"),
        }
    }

    /// Insert the immediate value `imm` for field `f` in the *type itself*,
    /// rather than into one of the variants.
    ///
    /// Most things want [`Self::insert_field`] instead, but this one is
    /// necessary for writing things like enum tags that aren't in any variant.
    pub(super) fn insert_imm(&mut self, f: FieldIdx, imm: V) {
        let field_offset = self.layout.fields.offset(f.as_usize());
        let is_zero_offset = field_offset == Size::ZERO;
        match &mut self.val {
            OperandValueBuilder::Immediate(val @ Either::Right(_)) if is_zero_offset => {
                *val = Either::Left(imm);
            }
            OperandValueBuilder::Pair(fst @ Either::Right(_), _) if is_zero_offset => {
                *fst = Either::Left(imm);
            }
            OperandValueBuilder::Pair(_, snd @ Either::Right(_)) if !is_zero_offset => {
                *snd = Either::Left(imm);
            }
            _ => bug!("Tried to insert {imm:?} into field {f:?} of {self:?}"),
        }
    }

    /// After having set all necessary fields, this converts the builder back
    /// to the normal `OperandRef`.
    ///
    /// ICEs if any required fields were not set.
    pub(super) fn build(&self, cx: &impl CodegenMethods<'tcx, Value = V>) -> OperandRef<'tcx, V> {
        let OperandRefBuilder { val, layout } = *self;

        // For something like `Option::<u32>::None`, it's expected that the
        // payload scalar will not actually have been set, so this converts
        // unset scalars to corresponding `undef` values so long as the scalar
        // from the layout allows uninit.
        let unwrap = |r: Either<V, abi::Scalar>| match r {
            Either::Left(v) => v,
            Either::Right(s) if s.is_uninit_valid() => {
                let bty = cx.type_from_scalar(s);
                cx.const_undef(bty)
            }
            Either::Right(_) => bug!("OperandRef::build called while fields are missing {self:?}"),
        };

        let val = match val {
            OperandValueBuilder::ZeroSized => OperandValue::ZeroSized,
            OperandValueBuilder::Immediate(v) => OperandValue::Immediate(unwrap(v)),
            OperandValueBuilder::Pair(a, b) => OperandValue::Pair(unwrap(a), unwrap(b)),
            OperandValueBuilder::Vector(v) => match v {
                Either::Left(v) => OperandValue::Immediate(v),
                Either::Right(())
                    if let BackendRepr::SimdVector { element, .. } = layout.backend_repr
                        && element.is_uninit_valid() =>
                {
                    let bty = cx.immediate_backend_type(layout);
                    OperandValue::Immediate(cx.const_undef(bty))
                }
                Either::Right(()) => {
                    bug!("OperandRef::build called while fields are missing {self:?}")
                }
            },
        };
        OperandRef { val, layout, move_annotation: None }
    }
}

/// Default size limit for move/copy annotations (in bytes). 64 bytes is a common size of a cache
/// line, and the assumption is that anything this size or below is very cheap to move/copy, so only
/// annotate copies larger than this.
const MOVE_ANNOTATION_DEFAULT_LIMIT: u64 = 65;

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
                // Avoid generating stores of zero-sized values, because the only way to have a
                // zero-sized value is through `undef`/`poison`, and the store itself is useless.
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
                let BackendRepr::ScalarPair(a_scalar, b_scalar) = dest.layout.backend_repr else {
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
                // We only need to handle the projections that
                // `LocalAnalyzer::process_place` let make it here.
                for elem in place_ref.projection {
                    match *elem {
                        mir::ProjectionElem::Field(f, _) => {
                            assert!(
                                !o.layout.ty.is_any_ptr(),
                                "Bad PlaceRef: destructing pointers should use cast/PtrMetadata, \
                                 but tried to access field {f:?} of pointer {o:?}",
                            );
                            o = o.extract_field(self, bx, f.index());
                        }
                        mir::PlaceElem::Downcast(_, vidx) => {
                            debug_assert_eq!(
                                o.layout.variants,
                                abi::Variants::Single { index: vidx },
                            );
                            let layout = o.layout.for_variant(bx.cx(), vidx);
                            o = OperandRef { layout, ..o }
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
                let kind = match operand {
                    mir::Operand::Move(_) => LangItem::CompilerMove,
                    mir::Operand::Copy(_) => LangItem::CompilerCopy,
                    _ => unreachable!(),
                };

                // Check if we should annotate this move/copy for profiling
                let move_annotation = self.move_copy_annotation_instance(bx, place.as_ref(), kind);

                OperandRef { move_annotation, ..self.codegen_consume(bx, place.as_ref()) }
            }

            mir::Operand::Constant(ref constant) => {
                let constant_ty = self.monomorphize(constant.ty());
                // Most SIMD vector constants should be passed as immediates.
                // (In particular, some intrinsics really rely on this.)
                if constant_ty.is_simd() {
                    // However, some SIMD types do not actually use the vector ABI
                    // (in particular, packed SIMD types do not). Ensure we exclude those.
                    let layout = bx.layout_of(constant_ty);
                    if let BackendRepr::SimdVector { .. } = layout.backend_repr {
                        let (llval, ty) = self.immediate_const_vector(bx, constant);
                        return OperandRef {
                            val: OperandValue::Immediate(llval),
                            layout: bx.layout_of(ty),
                            move_annotation: None,
                        };
                    }
                }
                self.eval_mir_constant_to_operand(bx, constant)
            }
        }
    }

    /// Creates an `Instance` for annotating a move/copy operation at codegen time.
    ///
    /// Returns `Some(instance)` if the operation should be annotated with debug info, `None`
    /// otherwise. The instance represents a monomorphized `compiler_move<T, SIZE>` or
    /// `compiler_copy<T, SIZE>` function that can be used to create debug scopes.
    ///
    /// There are a number of conditions that must be met for an annotation to be created, but aside
    /// from the basics (annotation is enabled, we're generating debuginfo), the primary concern is
    /// moves/copies which could result in a real `memcpy`. So we check for the size limit, but also
    /// that the underlying representation of the type is in memory.
    fn move_copy_annotation_instance(
        &self,
        bx: &Bx,
        place: mir::PlaceRef<'tcx>,
        kind: LangItem,
    ) -> Option<ty::Instance<'tcx>> {
        let tcx = bx.tcx();
        let sess = tcx.sess;

        // Skip if we're not generating debuginfo
        if sess.opts.debuginfo == DebugInfo::None {
            return None;
        }

        // Check if annotation is enabled and get size limit (otherwise skip)
        let size_limit = match sess.opts.unstable_opts.annotate_moves {
            AnnotateMoves::Disabled => return None,
            AnnotateMoves::Enabled(None) => MOVE_ANNOTATION_DEFAULT_LIMIT,
            AnnotateMoves::Enabled(Some(limit)) => limit,
        };

        let ty = self.monomorphized_place_ty(place);
        let layout = bx.cx().layout_of(ty);
        let ty_size = layout.size.bytes();

        // Only annotate if type has a memory representation and exceeds size limit (and has a
        // non-zero size)
        if layout.is_zst()
            || ty_size < size_limit
            || !matches!(layout.backend_repr, BackendRepr::Memory { .. })
        {
            return None;
        }

        // Look up the DefId for compiler_move or compiler_copy lang item
        let def_id = tcx.lang_items().get(kind)?;

        // Create generic args: compiler_move<T, SIZE> or compiler_copy<T, SIZE>
        let size_const = ty::Const::from_target_usize(tcx, ty_size);
        let generic_args = tcx.mk_args(&[ty.into(), size_const.into()]);

        // Create the Instance
        let typing_env = self.mir.typing_env(tcx);
        let instance = ty::Instance::expect_resolve(
            tcx,
            typing_env,
            def_id,
            generic_args,
            rustc_span::DUMMY_SP, // span only used for error messages
        );

        Some(instance)
    }
}
