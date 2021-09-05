use super::operand::OperandValue;
use super::{FunctionCx, LocalRef};

use crate::common::IntPredicate;
use crate::glue;
use crate::traits::*;
use crate::MemFlags;

use rustc_middle::mir;
use rustc_middle::mir::tcx::PlaceTy;
use rustc_middle::ty::layout::{HasTyCtxt, LayoutOf, TyAndLayout};
use rustc_middle::ty::{self, Ty};
use rustc_target::abi::{Abi, Align, FieldsShape, Int, TagEncoding};
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
        assert!(!layout.is_unsized());
        PlaceRef { llval, llextra: None, layout, align: layout.align.abi }
    }

    pub fn new_sized_aligned(
        llval: V,
        layout: TyAndLayout<'tcx>,
        align: Align,
    ) -> PlaceRef<'tcx, V> {
        assert!(!layout.is_unsized());
        PlaceRef { llval, llextra: None, layout, align }
    }

    // FIXME(eddyb) pass something else for the name so no work is done
    // unless LLVM IR names are turned on (e.g. for `--emit=llvm-ir`).
    pub fn alloca<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        layout: TyAndLayout<'tcx>,
    ) -> Self {
        assert!(!layout.is_unsized(), "tried to statically allocate unsized place");
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
                Abi::ScalarPair(ref a, ref b)
                    if offset == a.value.size(bx.cx()).align_to(b.value.align(bx.cx()).abi) =>
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
            _ if !field.is_unsized() => return simple(),
            ty::Slice(..) | ty::Str | ty::Foreign(..) => return simple(),
            ty::Adt(def, _) => {
                if def.repr.packed() {
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
    pub fn codegen_get_discr<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        self,
        bx: &mut Bx,
        cast_to: Ty<'tcx>,
    ) -> V {
        let cast_to = bx.cx().immediate_backend_type(bx.cx().layout_of(cast_to));
        if self.layout.abi.is_uninhabited() {
            return bx.cx().const_undef(cast_to);
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
            Variants::Multiple { ref tag, ref tag_encoding, tag_field, .. } => {
                (tag, tag_encoding, tag_field)
            }
        };

        // Read the tag/niche-encoded discriminant from memory.
        let tag = self.project_field(bx, tag_field);
        let tag = bx.load_operand(tag);

        // Decode the discriminant (specifically if it's niche-encoded).
        match *tag_encoding {
            TagEncoding::Direct => {
                let signed = match tag_scalar.value {
                    // We use `i1` for bytes that are always `0` or `1`,
                    // e.g., `#[repr(i8)] enum E { A, B }`, but we can't
                    // let LLVM interpret the `i1` as signed, because
                    // then `i1 1` (i.e., `E::B`) is effectively `i8 -1`.
                    Int(_, signed) => !tag_scalar.is_bool() && signed,
                    _ => false,
                };
                bx.intcast(tag.immediate(), cast_to, signed)
            }
            TagEncoding::Niche { dataful_variant, ref niche_variants, niche_start } => {
                // Rebase from niche values to discriminants, and check
                // whether the result is in range for the niche variants.
                let niche_llty = bx.cx().immediate_backend_type(tag.layout);
                let tag = tag.immediate();

                // We first compute the "relative discriminant" (wrt `niche_variants`),
                // that is, if `n = niche_variants.end() - niche_variants.start()`,
                // we remap `niche_start..=niche_start + n` (which may wrap around)
                // to (non-wrap-around) `0..=n`, to be able to check whether the
                // discriminant corresponds to a niche variant with one comparison.
                // We also can't go directly to the (variant index) discriminant
                // and check that it is in the range `niche_variants`, because
                // that might not fit in the same type, on top of needing an extra
                // comparison (see also the comment on `let niche_discr`).
                let relative_discr = if niche_start == 0 {
                    // Avoid subtracting `0`, which wouldn't work for pointers.
                    // FIXME(eddyb) check the actual primitive type here.
                    tag
                } else {
                    bx.sub(tag, bx.cx().const_uint_big(niche_llty, niche_start))
                };
                let relative_max = niche_variants.end().as_u32() - niche_variants.start().as_u32();
                let is_niche = if relative_max == 0 {
                    // Avoid calling `const_uint`, which wouldn't work for pointers.
                    // Also use canonical == 0 instead of non-canonical u<= 0.
                    // FIXME(eddyb) check the actual primitive type here.
                    bx.icmp(IntPredicate::IntEQ, relative_discr, bx.cx().const_null(niche_llty))
                } else {
                    let relative_max = bx.cx().const_uint(niche_llty, relative_max as u64);
                    bx.icmp(IntPredicate::IntULE, relative_discr, relative_max)
                };

                // NOTE(eddyb) this addition needs to be performed on the final
                // type, in case the niche itself can't represent all variant
                // indices (e.g. `u8` niche with more than `256` variants,
                // but enough uninhabited variants so that the remaining variants
                // fit in the niche).
                // In other words, `niche_variants.end - niche_variants.start`
                // is representable in the niche, but `niche_variants.end`
                // might not be, in extreme cases.
                let niche_discr = {
                    let relative_discr = if relative_max == 0 {
                        // HACK(eddyb) since we have only one niche, we know which
                        // one it is, and we can avoid having a dynamic value here.
                        bx.cx().const_uint(cast_to, 0)
                    } else {
                        bx.intcast(relative_discr, cast_to, false)
                    };
                    bx.add(
                        relative_discr,
                        bx.cx().const_uint(cast_to, niche_variants.start().as_u32() as u64),
                    )
                };

                bx.select(
                    is_niche,
                    niche_discr,
                    bx.cx().const_uint(cast_to, dataful_variant.as_u32() as u64),
                )
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
                    TagEncoding::Niche { dataful_variant, ref niche_variants, niche_start },
                tag_field,
                ..
            } => {
                if variant_index != dataful_variant {
                    if bx.cx().sess().target.arch == "arm"
                        || bx.cx().sess().target.arch == "aarch64"
                    {
                        // FIXME(#34427): as workaround for LLVM bug on ARM,
                        // use memset of 0 before assigning niche value.
                        let fill_byte = bx.cx().const_u8(0);
                        let size = bx.cx().const_usize(self.layout.size.bytes());
                        bx.memset(self.llval, fill_byte, size, self.align, MemFlags::empty());
                    }

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

    pub fn storage_live<Bx: BuilderMethods<'a, 'tcx, Value = V>>(&self, bx: &mut Bx) {
        bx.lifetime_start(self.llval, self.layout.size);
    }

    pub fn storage_dead<Bx: BuilderMethods<'a, 'tcx, Value = V>>(&self, bx: &mut Bx) {
        bx.lifetime_end(self.llval, self.layout.size);
    }
}

impl<'a, 'tcx, Bx: BuilderMethods<'a, 'tcx>> FunctionCx<'a, 'tcx, Bx> {
    pub fn codegen_place(
        &mut self,
        bx: &mut Bx,
        place_ref: mir::PlaceRef<'tcx>,
    ) -> PlaceRef<'tcx, Bx::Value> {
        debug!("codegen_place(place_ref={:?})", place_ref);
        let cx = self.cx;
        let tcx = self.cx.tcx();

        let result = match place_ref {
            mir::PlaceRef { local, projection: [] } => match self.locals[local] {
                LocalRef::Place(place) => {
                    return place;
                }
                LocalRef::UnsizedPlace(place) => {
                    return bx.load_operand(place).deref(cx);
                }
                LocalRef::Operand(..) => {
                    bug!("using operand local {:?} as place", place_ref);
                }
            },
            mir::PlaceRef { local, projection: [proj_base @ .., mir::ProjectionElem::Deref] } => {
                // Load the pointer from its location.
                self.codegen_consume(bx, mir::PlaceRef { local, projection: proj_base })
                    .deref(bx.cx())
            }
            mir::PlaceRef { local, projection: &[ref proj_base @ .., elem] } => {
                // FIXME turn this recursion into iteration
                let cg_base =
                    self.codegen_place(bx, mir::PlaceRef { local, projection: proj_base });

                match elem {
                    mir::ProjectionElem::Deref => bug!(),
                    mir::ProjectionElem::Field(ref field, _) => {
                        cg_base.project_field(bx, field.index())
                    }
                    mir::ProjectionElem::Index(index) => {
                        let index = &mir::Operand::Copy(mir::Place::from(index));
                        let index = self.codegen_operand(bx, index);
                        let llindex = index.immediate();
                        cg_base.project_index(bx, llindex)
                    }
                    mir::ProjectionElem::ConstantIndex {
                        offset,
                        from_end: false,
                        min_length: _,
                    } => {
                        let lloffset = bx.cx().const_usize(offset as u64);
                        cg_base.project_index(bx, lloffset)
                    }
                    mir::ProjectionElem::ConstantIndex {
                        offset,
                        from_end: true,
                        min_length: _,
                    } => {
                        let lloffset = bx.cx().const_usize(offset as u64);
                        let lllen = cg_base.len(bx.cx());
                        let llindex = bx.sub(lllen, lloffset);
                        cg_base.project_index(bx, llindex)
                    }
                    mir::ProjectionElem::Subslice { from, to, from_end } => {
                        let mut subslice =
                            cg_base.project_index(bx, bx.cx().const_usize(from as u64));
                        let projected_ty =
                            PlaceTy::from_ty(cg_base.layout.ty).projection_ty(tcx, elem).ty;
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
                }
            }
        };
        debug!("codegen_place(place={:?}) => {:?}", place_ref, result);
        result
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
