use super::operand::OperandValue;
use super::{FunctionCx, LocalRef};

use crate::common::IntPredicate;
use crate::glue;
use crate::traits::*;

use rustc_middle::mir;
use rustc_middle::mir::tcx::PlaceTy;
use rustc_middle::ty::layout::{HasTyCtxt, LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, Ty};
use rustc_target::abi::{Abi, Align, FieldsShape, Int, Pointer, TagEncoding};
use rustc_target::abi::{VariantIdx, Variants};

#[derive(Copy, Clone, Debug)]
pub struct PlaceRef<'tcx, V> {
    /// A pointer to the contents of the place.
    pub llval: V,

    /// This place's extra data if it is unsized, or `None` if null.
    pub llextra: Option<V>,

    /// The monomorphized type of this place, including variant information.
    pub layout: TyAndLayout<'tcx>,

    /// The alignment we know for this place.
    pub align: Align,
}

impl<'a, 'tcx, V: CodegenObject> PlaceRef<'tcx, V> {
    pub fn new_sized(llval: V, layout: TyAndLayout<'tcx>) -> PlaceRef<'tcx, V> {
        assert!(layout.is_sized());
        PlaceRef { llval, llextra: None, layout, align: layout.align.abi }
    }

    pub fn new_sized_aligned(
        llval: V,
        layout: TyAndLayout<'tcx>,
        align: Align,
    ) -> PlaceRef<'tcx, V> {
        assert!(layout.is_sized());
        PlaceRef { llval, llextra: None, layout, align }
    }

    // FIXME(eddyb) pass something else for the name so no work is done
    // unless LLVM IR names are turned on (e.g. for `--emit=llvm-ir`).
    pub fn alloca<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
    ) -> Self {
        assert!(layout.is_sized(), "tried to statically allocate unsized place");
        let tmp = bx.alloca(bx.cx().backend_type(layout), layout.align.abi);
        Self::new_sized(tmp, layout)
    }

    /// Returns a place for an indirect reference to an unsized place.
    // FIXME(eddyb) pass something else for the name so no work is done
    // unless LLVM IR names are turned on (e.g. for `--emit=llvm-ir`).
    pub fn alloca_unsized_indirect<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
    ) -> Self {
        assert!(layout.is_unsized(), "tried to allocate indirect place for sized values");
        let ptr_ty = bx.cx().tcx().mk_mut_ptr(layout.ty);
        let ptr_layout = bx.cx().layout_of(ptr_ty);
        Self::alloca(bx, ptr_layout)
    }

    pub fn len<Cx: ConstMethods<'tcx, Value = V>>(&self, cx: &Cx) -> V {
        if let FieldsShape::Array { count, .. } = self.layout.fields {
            if self.layout.is_unsized() {
                assert_eq!(count, 0);
                self.llextra.unwrap()
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
        let effective_field_align = self.align.restrict_for_offset(offset);

        let mut simple = || {
            let llval = match self.layout.abi {
                _ if offset.bytes() == 0 => {
                    // Unions and newtypes only use an offset of 0.
                    // Also handles the first field of Scalar, ScalarPair, and Vector layouts.
                    self.llval
                }
                Abi::ScalarPair(a, b)
                    if offset == a.size(bx.cx()).align_to(b.align(bx.cx()).abi) =>
                {
                    // Offset matches second field.
                    let ty = bx.backend_type(self.layout);
                    bx.struct_gep(ty, self.llval, 1)
                }
                Abi::Scalar(_) | Abi::ScalarPair(..) | Abi::Vector { .. } if field.is_zst() => {
                    // ZST fields are not included in Scalar, ScalarPair, and Vector layouts, so manually offset the pointer.
                    let byte_ptr = bx.pointercast(self.llval, bx.cx().type_i8p());
                    bx.gep(bx.cx().type_i8(), byte_ptr, &[bx.const_usize(offset.bytes())])
                }
                Abi::Scalar(_) | Abi::ScalarPair(..) => {
                    // All fields of Scalar and ScalarPair layouts must have been handled by this point.
                    // Vector layouts have additional fields for each element of the vector, so don't panic in that case.
                    bug!(
                        "offset of non-ZST field `{:?}` does not match layout `{:#?}`",
                        field,
                        self.layout
                    );
                }
                _ => {
                    let ty = bx.backend_type(self.layout);
                    bx.struct_gep(ty, self.llval, bx.cx().backend_field_index(self.layout, ix))
                }
            };
            PlaceRef {
                // HACK(eddyb): have to bitcast pointers until LLVM removes pointee types.
                llval: bx.pointercast(llval, bx.cx().type_ptr_to(bx.cx().backend_type(field))),
                llextra: if bx.cx().type_has_metadata(field.ty) { self.llextra } else { None },
                layout: field,
                align: effective_field_align,
            }
        };

        // Simple cases, which don't need DST adjustment:
        //   * no metadata available - just log the case
        //   * known alignment - sized types, `[T]`, `str` or a foreign type
        //   * packed struct - there is no alignment padding
        match field.ty.kind() {
            _ if self.llextra.is_none() => {
                debug!(
                    "unsized field `{}`, of `{:?}` has no metadata for adjustment",
                    ix, self.llval
                );
                return simple();
            }
            _ if field.is_sized() => return simple(),
            ty::Slice(..) | ty::Str | ty::Foreign(..) => return simple(),
            ty::Adt(def, _) => {
                if def.repr().packed() {
                    // FIXME(eddyb) generalize the adjustment when we
                    // start supporting packing to larger alignments.
                    assert_eq!(self.layout.align.abi.bytes(), 1);
                    return simple();
                }
            }
            _ => {}
        }

        // We need to get the pointer manually now.
        // We do this by casting to a `*i8`, then offsetting it by the appropriate amount.
        // We do this instead of, say, simply adjusting the pointer from the result of a GEP
        // because the field may have an arbitrary alignment in the LLVM representation
        // anyway.
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

        let meta = self.llextra;

        let unaligned_offset = bx.cx().const_usize(offset.bytes());

        // Get the alignment of the field
        let (_, unsized_align) = glue::size_and_align_of_dst(bx, field.ty, meta);

        // Bump the unaligned offset up to the appropriate alignment
        let offset = round_up_const_value_to_alignment(bx, unaligned_offset, unsized_align);

        debug!("struct_field_ptr: DST field offset: {:?}", offset);

        // Cast and adjust pointer.
        let byte_ptr = bx.pointercast(self.llval, bx.cx().type_i8p());
        let byte_ptr = bx.gep(bx.cx().type_i8(), byte_ptr, &[offset]);

        // Finally, cast back to the type expected.
        let ll_fty = bx.cx().backend_type(field);
        debug!("struct_field_ptr: Field type is {:?}", ll_fty);

        PlaceRef {
            llval: bx.pointercast(byte_ptr, bx.cx().type_ptr_to(ll_fty)),
            llextra: self.llextra,
            layout: field,
            align: effective_field_align,
        }
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
        let cast_to_size = cast_to_layout.layout.size();
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

                let tag_size = tag_scalar.size(bx.cx());
                let max_unsigned = tag_size.unsigned_int_max();
                let max_signed = tag_size.signed_int_max() as u128;
                let min_signed = max_signed + 1;
                let relative_max = niche_variants.end().as_u32() - niche_variants.start().as_u32();
                let niche_end = niche_start.wrapping_add(relative_max as u128) & max_unsigned;
                let range = tag_scalar.valid_range(bx.cx());

                let sle = |lhs: u128, rhs: u128| -> bool {
                    // Signed and unsigned comparisons give the same results,
                    // except that in signed comparisons an integer with the
                    // sign bit set is less than one with the sign bit clear.
                    // Toggle the sign bit to do a signed comparison.
                    (lhs ^ min_signed) <= (rhs ^ min_signed)
                };

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

                // Find the least and greatest values in `range`, considered
                // both as signed and unsigned.
                let (low_unsigned, high_unsigned) = if range.start <= range.end {
                    (range.start, range.end)
                } else {
                    (0, max_unsigned)
                };
                let (low_signed, high_signed) = if sle(range.start, range.end) {
                    (range.start, range.end)
                } else {
                    (min_signed, max_signed)
                };

                let niches_ule = niche_start <= niche_end;
                let niches_sle = sle(niche_start, niche_end);
                let cast_smaller = cast_to_size <= tag_size;

                // In the algorithm above, we can change
                // cast(relative_tag) + niche_variants.start()
                // into
                // cast(tag + (niche_variants.start() - niche_start))
                // if either the casted type is no larger than the original
                // type, or if the niche values are contiguous (in either the
                // signed or unsigned sense).
                let can_incr = cast_smaller || niches_ule || niches_sle;

                let data_for_boundary_niche = || -> Option<(IntPredicate, u128)> {
                    if !can_incr {
                        None
                    } else if niche_start == low_unsigned {
                        Some((IntPredicate::IntULE, niche_end))
                    } else if niche_end == high_unsigned {
                        Some((IntPredicate::IntUGE, niche_start))
                    } else if niche_start == low_signed {
                        Some((IntPredicate::IntSLE, niche_end))
                    } else if niche_end == high_signed {
                        Some((IntPredicate::IntSGE, niche_start))
                    } else {
                        None
                    }
                };

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
                } else if let Some((predicate, constant)) = data_for_boundary_niche() {
                    // The niche values are either the lowest or the highest in
                    // `range`. We can avoid the first subtraction in the
                    // algorithm.
                    // The algorithm is now this:
                    // is_niche = tag <= niche_end
                    // discr = if is_niche {
                    //     cast(tag + (niche_variants.start() - niche_start))
                    // } else {
                    //     untagged_variant
                    // }
                    // (the first line may instead be tag >= niche_start,
                    // and may be a signed or unsigned comparison)
                    // The arithmetic must be done before the cast, so we can
                    // have the correct wrapping behavior. See issue #104519 for
                    // the consequences of getting this wrong.
                    let is_niche =
                        bx.icmp(predicate, tag, bx.cx().const_uint_big(tag_llty, constant));
                    let delta = (niche_variants.start().as_u32() as u128).wrapping_sub(niche_start);
                    let incr_tag = if delta == 0 {
                        tag
                    } else {
                        bx.add(tag, bx.cx().const_uint_big(tag_llty, delta))
                    };

                    let cast_tag = if cast_smaller {
                        bx.intcast(incr_tag, cast_to, false)
                    } else if niches_ule {
                        bx.zext(incr_tag, cast_to)
                    } else {
                        bx.sext(incr_tag, cast_to)
                    };

                    (is_niche, cast_tag, 0)
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
                bx.store(
                    bx.cx().const_uint_big(bx.cx().backend_type(ptr.layout), to),
                    ptr.llval,
                    ptr.align,
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

        PlaceRef {
            llval: bx.inbounds_gep(
                bx.cx().backend_type(self.layout),
                self.llval,
                &[bx.cx().const_usize(0), llindex],
            ),
            llextra: None,
            layout,
            align: self.align.restrict_for_offset(offset),
        }
    }

    pub fn project_downcast<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        &self,
        bx: &mut Bx,
        variant_index: VariantIdx,
    ) -> Self {
        let mut downcast = *self;
        downcast.layout = self.layout.for_variant(bx.cx(), variant_index);

        // Cast to the appropriate variant struct type.
        let variant_ty = bx.cx().backend_type(downcast.layout);
        downcast.llval = bx.pointercast(downcast.llval, bx.cx().type_ptr_to(variant_ty));

        downcast
    }

    pub fn project_type<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        &self,
        bx: &mut Bx,
        ty: Ty<'tcx>,
    ) -> Self {
        let mut downcast = *self;
        downcast.layout = bx.cx().layout_of(ty);

        // Cast to the appropriate type.
        let variant_ty = bx.cx().backend_type(downcast.layout);
        downcast.llval = bx.pointercast(downcast.llval, bx.cx().type_ptr_to(variant_ty));

        downcast
    }

    pub fn storage_live<Bx: BuilderMethods<'a, 'tcx, Value = V>>(&self, bx: &mut Bx) {
        bx.lifetime_start(self.llval, self.layout.size);
    }

    pub fn storage_dead<Bx: BuilderMethods<'a, 'tcx, Value = V>>(&self, bx: &mut Bx) {
        bx.lifetime_end(self.llval, self.layout.size);
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
                if place_ref.has_deref() {
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
        };
        for elem in place_ref.projection[base..].iter() {
            cg_base = match *elem {
                mir::ProjectionElem::Deref => bx.load_operand(cg_base).deref(bx.cx()),
                mir::ProjectionElem::Field(ref field, _) => {
                    cg_base.project_field(bx, field.index())
                }
                mir::ProjectionElem::OpaqueCast(ty) => cg_base.project_type(bx, ty),
                mir::ProjectionElem::Index(index) => {
                    let index = &mir::Operand::Copy(mir::Place::from(index));
                    let index = self.codegen_operand(bx, index);
                    let llindex = index.immediate();
                    cg_base.project_index(bx, llindex)
                }
                mir::ProjectionElem::ConstantIndex { offset, from_end: false, min_length: _ } => {
                    let lloffset = bx.cx().const_usize(offset as u64);
                    cg_base.project_index(bx, lloffset)
                }
                mir::ProjectionElem::ConstantIndex { offset, from_end: true, min_length: _ } => {
                    let lloffset = bx.cx().const_usize(offset as u64);
                    let lllen = cg_base.len(bx.cx());
                    let llindex = bx.sub(lllen, lloffset);
                    cg_base.project_index(bx, llindex)
                }
                mir::ProjectionElem::Subslice { from, to, from_end } => {
                    let mut subslice = cg_base.project_index(bx, bx.cx().const_usize(from as u64));
                    let projected_ty =
                        PlaceTy::from_ty(cg_base.layout.ty).projection_ty(tcx, *elem).ty;
                    subslice.layout = bx.cx().layout_of(self.monomorphize(projected_ty));

                    if subslice.layout.is_unsized() {
                        assert!(from_end, "slice subslices should be `from_end`");
                        subslice.llextra = Some(bx.sub(
                            cg_base.llextra.unwrap(),
                            bx.cx().const_usize((from as u64) + (to as u64)),
                        ));
                    }

                    // Cast the place pointer type to the new
                    // array or slice type (`*[%_; new_len]`).
                    subslice.llval = bx.pointercast(
                        subslice.llval,
                        bx.cx().type_ptr_to(bx.cx().backend_type(subslice.layout)),
                    );

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
