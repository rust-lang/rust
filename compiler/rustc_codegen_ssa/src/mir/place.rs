use super::operand::OperandValue;
use super::{FunctionCx, LocalRef};

use crate::common::IntPredicate;
use crate::size_of_val;
use crate::traits::*;

use rustc_middle::bug;
use rustc_middle::mir;
use rustc_middle::mir::tcx::PlaceTy;
use rustc_middle::ty::layout::{HasTyCtxt, LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, Ty};
use rustc_target::abi::{Align, FieldsShape, Int, Pointer, Size, TagEncoding};
use rustc_target::abi::{VariantIdx, Variants};
use tracing::{debug, instrument};

/// The location and extra runtime properties of the place.
///
/// Typically found in a [`PlaceRef`] or an [`OperandValue::Ref`].
///
/// As a location in memory, this has no specific type. If you want to
/// load or store it using a typed operation, use [`Self::with_type`].
#[derive(Copy, Clone, Debug)]
pub struct PlaceValue<V> {
    /// A pointer to the contents of the place.
    pub llval: V,

    /// This place's extra data if it is unsized, or `None` if null.
    pub llextra: Option<V>,

    /// The alignment we know for this place.
    pub align: Align,
}

impl<V: CodegenObject> PlaceValue<V> {
    /// Constructor for the ordinary case of `Sized` types.
    ///
    /// Sets `llextra` to `None`.
    pub fn new_sized(llval: V, align: Align) -> PlaceValue<V> {
        PlaceValue { llval, llextra: None, align }
    }

    /// Allocates a stack slot in the function for a value
    /// of the specified size and alignment.
    ///
    /// The allocation itself is untyped.
    pub fn alloca<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        size: Size,
        align: Align,
    ) -> PlaceValue<V> {
        let llval = bx.alloca(size, align);
        PlaceValue::new_sized(llval, align)
    }

    /// Creates a `PlaceRef` to this location with the given type.
    pub fn with_type<'tcx>(self, layout: TyAndLayout<'tcx>) -> PlaceRef<'tcx, V> {
        assert!(
            layout.is_unsized() || layout.abi.is_uninhabited() || self.llextra.is_none(),
            "Had pointer metadata {:?} for sized type {layout:?}",
            self.llextra,
        );
        PlaceRef { val: self, layout }
    }

    /// Gets the pointer to this place as an [`OperandValue::Immediate`]
    /// or, for those needing metadata, an [`OperandValue::Pair`].
    ///
    /// This is the inverse of [`OperandValue::deref`].
    pub fn address(self) -> OperandValue<V> {
        if let Some(llextra) = self.llextra {
            OperandValue::Pair(self.llval, llextra)
        } else {
            OperandValue::Immediate(self.llval)
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct PlaceRef<'tcx, V> {
    /// The location and extra runtime properties of the place.
    pub val: PlaceValue<V>,

    /// The monomorphized type of this place, including variant information.
    ///
    /// You probably shouldn't use the alignment from this layout;
    /// rather you should use the `.val.align` of the actual place,
    /// which might be different from the type's normal alignment.
    pub layout: TyAndLayout<'tcx>,
}

impl<'a, 'tcx, V: CodegenObject> PlaceRef<'tcx, V> {
    pub fn new_sized(llval: V, layout: TyAndLayout<'tcx>) -> PlaceRef<'tcx, V> {
        PlaceRef::new_sized_aligned(llval, layout, layout.align.abi)
    }

    pub fn new_sized_aligned(
        llval: V,
        layout: TyAndLayout<'tcx>,
        align: Align,
    ) -> PlaceRef<'tcx, V> {
        assert!(layout.is_sized());
        PlaceValue::new_sized(llval, align).with_type(layout)
    }

    // FIXME(eddyb) pass something else for the name so no work is done
    // unless LLVM IR names are turned on (e.g. for `--emit=llvm-ir`).
    pub fn alloca<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
    ) -> Self {
        Self::alloca_size(bx, layout.size, layout)
    }

    pub fn alloca_size<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        size: Size,
        layout: TyAndLayout<'tcx>,
    ) -> Self {
        assert!(layout.is_sized(), "tried to statically allocate unsized place");
        PlaceValue::alloca(bx, size, layout.align.abi).with_type(layout)
    }

    /// Returns a place for an indirect reference to an unsized place.
    // FIXME(eddyb) pass something else for the name so no work is done
    // unless LLVM IR names are turned on (e.g. for `--emit=llvm-ir`).
    pub fn alloca_unsized_indirect<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
    ) -> Self {
        assert!(layout.is_unsized(), "tried to allocate indirect place for sized values");
        let ptr_ty = Ty::new_mut_ptr(bx.cx().tcx(), layout.ty);
        let ptr_layout = bx.cx().layout_of(ptr_ty);
        Self::alloca(bx, ptr_layout)
    }

    pub fn len<Cx: ConstMethods<'tcx, Value = V>>(&self, cx: &Cx) -> V {
        if let FieldsShape::Array { count, .. } = self.layout.fields {
            if self.layout.is_unsized() {
                assert_eq!(count, 0);
                self.val.llextra.unwrap()
            } else {
                cx.const_usize(count)
            }
        } else {
            bug!("unexpected layout `{:#?}` in PlaceRef::len", self.layout)
        }
    }
}

impl<'a, 'tcx, V: CodegenObject> PlaceRef<'tcx, V> {
    /// Access a field, at a point when the value's case is known.
    pub fn project_field<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        ix: usize,
    ) -> Self {
        let field = self.layout.field(bx.cx(), ix);
        let offset = self.layout.fields.offset(ix);
        let effective_field_align = self.val.align.restrict_for_offset(offset);

        // `simple` is called when we don't need to adjust the offset to
        // the dynamic alignment of the field.
        let mut simple = || {
            let llval = if offset.bytes() == 0 {
                self.val.llval
            } else {
                bx.inbounds_ptradd(self.val.llval, bx.const_usize(offset.bytes()))
            };
            let val = PlaceValue {
                llval,
                llextra: if bx.cx().type_has_metadata(field.ty) { self.val.llextra } else { None },
                align: effective_field_align,
            };
            val.with_type(field)
        };

        // Simple cases, which don't need DST adjustment:
        //   * known alignment - sized types, `[T]`, `str`
        //   * offset 0 -- rounding up to alignment cannot change the offset
        // Note that looking at `field.align` is incorrect since that is not necessarily equal
        // to the dynamic alignment of the type.
        match field.ty.kind() {
            _ if field.is_sized() => return simple(),
            ty::Slice(..) | ty::Str => return simple(),
            _ if offset.bytes() == 0 => return simple(),
            _ => {}
        }

        // We need to get the pointer manually now.
        // We do this by casting to a `*i8`, then offsetting it by the appropriate amount.
        // We do this instead of, say, simply adjusting the pointer from the result of a GEP
        // because the field may have an arbitrary alignment in the LLVM representation.
        //
        // To demonstrate:
        //
        //     struct Foo<T: ?Sized> {
        //         x: u16,
        //         y: T
        //     }
        //
        // The type `Foo<Foo<Trait>>` is represented in LLVM as `{ u16, { u16, u8 }}`, meaning that
        // the `y` field has 16-bit alignment.

        let meta = self.val.llextra;

        let unaligned_offset = bx.cx().const_usize(offset.bytes());

        // Get the alignment of the field
        let (_, mut unsized_align) = size_of_val::size_and_align_of_dst(bx, field.ty, meta);

        // For packed types, we need to cap alignment.
        if let ty::Adt(def, _) = self.layout.ty.kind()
            && let Some(packed) = def.repr().pack
        {
            let packed = bx.const_usize(packed.bytes());
            let cmp = bx.icmp(IntPredicate::IntULT, unsized_align, packed);
            unsized_align = bx.select(cmp, unsized_align, packed)
        }

        // Bump the unaligned offset up to the appropriate alignment
        let offset = round_up_const_value_to_alignment(bx, unaligned_offset, unsized_align);

        debug!("struct_field_ptr: DST field offset: {:?}", offset);

        // Adjust pointer.
        let ptr = bx.inbounds_ptradd(self.val.llval, offset);
        let val =
            PlaceValue { llval: ptr, llextra: self.val.llextra, align: effective_field_align };
        val.with_type(field)
    }

    /// Obtain the actual discriminant of a value.
    #[instrument(level = "trace", skip(bx))]
    pub fn codegen_get_discr<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        cast_to: Ty<'tcx>,
    ) -> V {
        let dl = &bx.tcx().data_layout;
        let cast_to_layout = bx.cx().layout_of(cast_to);
        let cast_to = bx.cx().immediate_backend_type(cast_to_layout);
        if self.layout.abi.is_uninhabited() {
            return bx.cx().const_poison(cast_to);
        }
        let (tag_scalar, tag_encoding, tag_field) = match self.layout.variants {
            Variants::Single { index } => {
                let discr_val = self
                    .layout
                    .ty
                    .discriminant_for_variant(bx.cx().tcx(), index)
                    .map_or(index.as_u32() as u128, |discr| discr.val);
                return bx.cx().const_uint_big(cast_to, discr_val);
            }
            Variants::Multiple { tag, ref tag_encoding, tag_field, .. } => {
                (tag, tag_encoding, tag_field)
            }
        };

        // Read the tag/niche-encoded discriminant from memory.
        let tag = self.project_field(bx, tag_field);
        let tag_op = bx.load_operand(tag);
        let tag_imm = tag_op.immediate();

        // Decode the discriminant (specifically if it's niche-encoded).
        match *tag_encoding {
            TagEncoding::Direct => {
                let signed = match tag_scalar.primitive() {
                    // We use `i1` for bytes that are always `0` or `1`,
                    // e.g., `#[repr(i8)] enum E { A, B }`, but we can't
                    // let LLVM interpret the `i1` as signed, because
                    // then `i1 1` (i.e., `E::B`) is effectively `i8 -1`.
                    Int(_, signed) => !tag_scalar.is_bool() && signed,
                    _ => false,
                };
                bx.intcast(tag_imm, cast_to, signed)
            }
            TagEncoding::Niche { untagged_variant, ref niche_variants, niche_start } => {
                // Cast to an integer so we don't have to treat a pointer as a
                // special case.
                let (tag, tag_llty) = match tag_scalar.primitive() {
                    // FIXME(erikdesjardins): handle non-default addrspace ptr sizes
                    Pointer(_) => {
                        let t = bx.type_from_integer(dl.ptr_sized_integer());
                        let tag = bx.ptrtoint(tag_imm, t);
                        (tag, t)
                    }
                    _ => (tag_imm, bx.cx().immediate_backend_type(tag_op.layout)),
                };

                let relative_max = niche_variants.end().as_u32() - niche_variants.start().as_u32();

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
                    let niche_start = bx.cx().const_uint_big(tag_llty, niche_start);
                    let is_niche = bx.icmp(IntPredicate::IntEQ, tag, niche_start);
                    let tagged_discr =
                        bx.cx().const_uint(cast_to, niche_variants.start().as_u32() as u64);
                    (is_niche, tagged_discr, 0)
                } else {
                    // The special cases don't apply, so we'll have to go with
                    // the general algorithm.
                    let relative_discr = bx.sub(tag, bx.cx().const_uint_big(tag_llty, niche_start));
                    let cast_tag = bx.intcast(relative_discr, cast_to, false);
                    let is_niche = bx.icmp(
                        IntPredicate::IntULE,
                        relative_discr,
                        bx.cx().const_uint(tag_llty, relative_max as u64),
                    );
                    (is_niche, cast_tag, niche_variants.start().as_u32() as u128)
                };

                let tagged_discr = if delta == 0 {
                    tagged_discr
                } else {
                    bx.add(tagged_discr, bx.cx().const_uint_big(cast_to, delta))
                };

                let discr = bx.select(
                    is_niche,
                    tagged_discr,
                    bx.cx().const_uint(cast_to, untagged_variant.as_u32() as u64),
                );

                // In principle we could insert assumes on the possible range of `discr`, but
                // currently in LLVM this seems to be a pessimization.

                discr
            }
        }
    }

    /// Sets the discriminant for a new value of the given case of the given
    /// representation.
    pub fn codegen_set_discr<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        &self,
        bx: &mut Bx,
        variant_index: VariantIdx,
    ) {
        if self.layout.for_variant(bx.cx(), variant_index).abi.is_uninhabited() {
            // We play it safe by using a well-defined `abort`, but we could go for immediate UB
            // if that turns out to be helpful.
            bx.abort();
            return;
        }
        match self.layout.variants {
            Variants::Single { index } => {
                assert_eq!(index, variant_index);
            }
            Variants::Multiple { tag_encoding: TagEncoding::Direct, tag_field, .. } => {
                let ptr = self.project_field(bx, tag_field);
                let to =
                    self.layout.ty.discriminant_for_variant(bx.tcx(), variant_index).unwrap().val;
                bx.store_to_place(
                    bx.cx().const_uint_big(bx.cx().backend_type(ptr.layout), to),
                    ptr.val,
                );
            }
            Variants::Multiple {
                tag_encoding:
                    TagEncoding::Niche { untagged_variant, ref niche_variants, niche_start },
                tag_field,
                ..
            } => {
                if variant_index != untagged_variant {
                    let niche = self.project_field(bx, tag_field);
                    let niche_llty = bx.cx().immediate_backend_type(niche.layout);
                    let niche_value = variant_index.as_u32() - niche_variants.start().as_u32();
                    let niche_value = (niche_value as u128).wrapping_add(niche_start);
                    // FIXME(eddyb): check the actual primitive type here.
                    let niche_llval = if niche_value == 0 {
                        // HACK(eddyb): using `c_null` as it works on all types.
                        bx.cx().const_null(niche_llty)
                    } else {
                        bx.cx().const_uint_big(niche_llty, niche_value)
                    };
                    OperandValue::Immediate(niche_llval).store(bx, niche);
                }
            }
        }
    }

    pub fn project_index<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        &self,
        bx: &mut Bx,
        llindex: V,
    ) -> Self {
        // Statically compute the offset if we can, otherwise just use the element size,
        // as this will yield the lowest alignment.
        let layout = self.layout.field(bx, 0);
        let offset = if let Some(llindex) = bx.const_to_opt_uint(llindex) {
            layout.size.checked_mul(llindex, bx).unwrap_or(layout.size)
        } else {
            layout.size
        };

        let llval = bx.inbounds_gep(
            bx.cx().backend_type(self.layout),
            self.val.llval,
            &[bx.cx().const_usize(0), llindex],
        );
        let align = self.val.align.restrict_for_offset(offset);
        PlaceValue::new_sized(llval, align).with_type(layout)
    }

    pub fn project_downcast<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        &self,
        bx: &mut Bx,
        variant_index: VariantIdx,
    ) -> Self {
        let mut downcast = *self;
        downcast.layout = self.layout.for_variant(bx.cx(), variant_index);
        downcast
    }

    pub fn project_type<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        &self,
        bx: &mut Bx,
        ty: Ty<'tcx>,
    ) -> Self {
        let mut downcast = *self;
        downcast.layout = bx.cx().layout_of(ty);
        downcast
    }

    pub fn storage_live<Bx: BuilderMethods<'a, 'tcx, Value = V>>(&self, bx: &mut Bx) {
        bx.lifetime_start(self.val.llval, self.layout.size);
    }

    pub fn storage_dead<Bx: BuilderMethods<'a, 'tcx, Value = V>>(&self, bx: &mut Bx) {
        bx.lifetime_end(self.val.llval, self.layout.size);
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    #[instrument(level = "trace", skip(self, bx))]
    pub fn codegen_place(
        &mut self,
        bx: &mut Bx,
        place_ref: mir::PlaceRef<'tcx>,
    ) -> PlaceRef<'tcx, Bx::Value> {
        let cx = self.cx;
        let tcx = self.cx.tcx();

        let mut base = 0;
        let mut cg_base = match self.locals[place_ref.local] {
            LocalRef::Place(place) => place,
            LocalRef::UnsizedPlace(place) => bx.load_operand(place).deref(cx),
            LocalRef::Operand(..) => {
                if place_ref.is_indirect_first_projection() {
                    base = 1;
                    let cg_base = self.codegen_consume(
                        bx,
                        mir::PlaceRef { projection: &place_ref.projection[..0], ..place_ref },
                    );
                    cg_base.deref(bx.cx())
                } else {
                    bug!("using operand local {:?} as place", place_ref);
                }
            }
            LocalRef::PendingOperand => {
                bug!("using still-pending operand local {:?} as place", place_ref);
            }
        };
        for elem in place_ref.projection[base..].iter() {
            cg_base = match *elem {
                mir::ProjectionElem::Deref => bx.load_operand(cg_base).deref(bx.cx()),
                mir::ProjectionElem::Field(ref field, _) => {
                    assert!(
                        !cg_base.layout.ty.is_any_ptr(),
                        "Bad PlaceRef: destructing pointers should use cast/PtrMetadata, \
                         but tried to access field {field:?} of pointer {cg_base:?}",
                    );
                    cg_base.project_field(bx, field.index())
                }
                mir::ProjectionElem::OpaqueCast(ty) => {
                    bug!("encountered OpaqueCast({ty}) in codegen")
                }
                mir::ProjectionElem::Subtype(ty) => cg_base.project_type(bx, self.monomorphize(ty)),
                mir::ProjectionElem::Index(index) => {
                    let index = &mir::Operand::Copy(mir::Place::from(index));
                    let index = self.codegen_operand(bx, index);
                    let llindex = index.immediate();
                    cg_base.project_index(bx, llindex)
                }
                mir::ProjectionElem::ConstantIndex { offset, from_end: false, min_length: _ } => {
                    let lloffset = bx.cx().const_usize(offset);
                    cg_base.project_index(bx, lloffset)
                }
                mir::ProjectionElem::ConstantIndex { offset, from_end: true, min_length: _ } => {
                    let lloffset = bx.cx().const_usize(offset);
                    let lllen = cg_base.len(bx.cx());
                    let llindex = bx.sub(lllen, lloffset);
                    cg_base.project_index(bx, llindex)
                }
                mir::ProjectionElem::Subslice { from, to, from_end } => {
                    let mut subslice = cg_base.project_index(bx, bx.cx().const_usize(from));
                    let projected_ty =
                        PlaceTy::from_ty(cg_base.layout.ty).projection_ty(tcx, *elem).ty;
                    subslice.layout = bx.cx().layout_of(self.monomorphize(projected_ty));

                    if subslice.layout.is_unsized() {
                        assert!(from_end, "slice subslices should be `from_end`");
                        subslice.val.llextra = Some(
                            bx.sub(cg_base.val.llextra.unwrap(), bx.cx().const_usize(from + to)),
                        );
                    }

                    subslice
                }
                mir::ProjectionElem::Downcast(_, v) => cg_base.project_downcast(bx, v),
            };
        }
        debug!("codegen_place(place={:?}) => {:?}", place_ref, cg_base);
        cg_base
    }

    pub fn monomorphized_place_ty(&self, place_ref: mir::PlaceRef<'tcx>) -> Ty<'tcx> {
        let tcx = self.cx.tcx();
        let place_ty = place_ref.ty(self.mir, tcx);
        self.monomorphize(place_ty.ty)
    }
}

fn round_up_const_value_to_alignment<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>>(
    bx: &mut Bx,
    value: Bx::Value,
    align: Bx::Value,
) -> Bx::Value {
    // In pseudo code:
    //
    //     if value & (align - 1) == 0 {
    //         value
    //     } else {
    //         (value & !(align - 1)) + align
    //     }
    //
    // Usually this is written without branches as
    //
    //     (value + align - 1) & !(align - 1)
    //
    // But this formula cannot take advantage of constant `value`. E.g. if `value` is known
    // at compile time to be `1`, this expression should be optimized to `align`. However,
    // optimization only holds if `align` is a power of two. Since the optimizer doesn't know
    // that `align` is a power of two, it cannot perform this optimization.
    //
    // Instead we use
    //
    //     value + (-value & (align - 1))
    //
    // Since `align` is used only once, the expression can be optimized. For `value = 0`
    // its optimized to `0` even in debug mode.
    //
    // NB: The previous version of this code used
    //
    //     (value + align - 1) & -align
    //
    // Even though `-align == !(align - 1)`, LLVM failed to optimize this even for
    // `value = 0`. Bug report: https://bugs.llvm.org/show_bug.cgi?id=48559
    let one = bx.const_usize(1);
    let align_minus_1 = bx.sub(align, one);
    let neg_value = bx.neg(value);
    let offset = bx.and(neg_value, align_minus_1);
    bx.add(value, offset)
}
