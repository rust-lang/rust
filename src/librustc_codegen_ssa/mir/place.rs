use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, Align, TyLayout, LayoutOf, VariantIdx, HasTyCtxt};
use rustc::mir;
use rustc::mir::tcx::PlaceTy;
use crate::MemFlags;
use crate::common::IntPredicate;
use crate::glue;

use crate::traits::*;

use super::{FunctionCx, LocalRef};
use super::operand::OperandValue;

#[derive(Copy, Clone, Debug)]
pub struct PlaceRef<'tcx, V> {
    /// Pointer to the contents of the place.
    pub llval: V,

    /// This place's extra data if it is unsized, or null.
    pub llextra: Option<V>,

    /// Monomorphized type of this place, including variant information.
    pub layout: TyLayout<'tcx>,

    /// What alignment we know for this place.
    pub align: Align,
}

impl<'a, 'tcx, V: CodegenObject> PlaceRef<'tcx, V> {
    pub fn new_sized(
        llval: V,
        layout: TyLayout<'tcx>,
        align: Align,
    ) -> PlaceRef<'tcx, V> {
        assert!(!layout.is_unsized());
        PlaceRef {
            llval,
            llextra: None,
            layout,
            align
        }
    }

    fn new_thin_place<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        llval: V,
        layout: TyLayout<'tcx>,
        align: Align,
    ) -> PlaceRef<'tcx, V> {
        assert!(!bx.cx().type_has_metadata(layout.ty));
        PlaceRef {
            llval,
            llextra: None,
            layout,
            align
        }
    }

    pub fn alloca<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        layout: TyLayout<'tcx>,
        name: &str
    ) -> Self {
        debug!("alloca({:?}: {:?})", name, layout);
        assert!(!layout.is_unsized(), "tried to statically allocate unsized place");
        let tmp = bx.alloca(bx.cx().backend_type(layout), name, layout.align.abi);
        Self::new_sized(tmp, layout, layout.align.abi)
    }

    /// Returns a place for an indirect reference to an unsized place.
    pub fn alloca_unsized_indirect<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        bx: &mut Bx,
        layout: TyLayout<'tcx>,
        name: &str,
    ) -> Self {
        debug!("alloca_unsized_indirect({:?}: {:?})", name, layout);
        assert!(layout.is_unsized(), "tried to allocate indirect place for sized values");
        let ptr_ty = bx.cx().tcx().mk_mut_ptr(layout.ty);
        let ptr_layout = bx.cx().layout_of(ptr_ty);
        Self::alloca(bx, ptr_layout, name)
    }

    pub fn len<Cx: ConstMethods<'tcx, Value = V>>(
        &self,
        cx: &Cx
    ) -> V {
        if let layout::FieldPlacement::Array { count, .. } = self.layout.fields {
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
        self, bx: &mut Bx,
        ix: usize,
    ) -> Self {
        let field = self.layout.field(bx.cx(), ix);
        let offset = self.layout.fields.offset(ix);
        let effective_field_align = self.align.restrict_for_offset(offset);

        let mut simple = || {
            // Unions and newtypes only use an offset of 0.
            let llval = if offset.bytes() == 0 {
                self.llval
            } else if let layout::Abi::ScalarPair(ref a, ref b) = self.layout.abi {
                // Offsets have to match either first or second field.
                assert_eq!(offset, a.value.size(bx.cx()).align_to(b.value.align(bx.cx()).abi));
                bx.struct_gep(self.llval, 1)
            } else {
                bx.struct_gep(self.llval, bx.cx().backend_field_index(self.layout, ix))
            };
            PlaceRef {
                // HACK(eddyb): have to bitcast pointers until LLVM removes pointee types.
                llval: bx.pointercast(llval, bx.cx().type_ptr_to(bx.cx().backend_type(field))),
                llextra: if bx.cx().type_has_metadata(field.ty) {
                    self.llextra
                } else {
                    None
                },
                layout: field,
                align: effective_field_align,
            }
        };

        // Simple cases, which don't need DST adjustment:
        //   * no metadata available - just log the case
        //   * known alignment - sized types, `[T]`, `str` or a foreign type
        //   * packed struct - there is no alignment padding
        match field.ty.sty {
            _ if self.llextra.is_none() => {
                debug!("Unsized field `{}`, of `{:?}` has no metadata for adjustment",
                    ix, self.llval);
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

        // Bump the unaligned offset up to the appropriate alignment using the
        // following expression:
        //
        //     (unaligned offset + (align - 1)) & -align

        // Calculate offset.
        let align_sub_1 = bx.sub(unsized_align, bx.cx().const_usize(1u64));
        let and_lhs = bx.add(unaligned_offset, align_sub_1);
        let and_rhs = bx.neg(unsized_align);
        let offset = bx.and(and_lhs, and_rhs);

        debug!("struct_field_ptr: DST field offset: {:?}", offset);

        // Cast and adjust pointer.
        let byte_ptr = bx.pointercast(self.llval, bx.cx().type_i8p());
        let byte_ptr = bx.gep(byte_ptr, &[offset]);

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
        cast_to: Ty<'tcx>
    ) -> V {
        let cast_to = bx.cx().immediate_backend_type(bx.cx().layout_of(cast_to));
        if self.layout.abi.is_uninhabited() {
            return bx.cx().const_undef(cast_to);
        }
        let (discr_scalar, discr_kind, discr_index) = match self.layout.variants {
            layout::Variants::Single { index } => {
                let discr_val = self.layout.ty.discriminant_for_variant(bx.cx().tcx(), index)
                    .map_or(index.as_u32() as u128, |discr| discr.val);
                return bx.cx().const_uint_big(cast_to, discr_val);
            }
            layout::Variants::Multiple { ref discr, ref discr_kind, discr_index, .. } => {
                (discr, discr_kind, discr_index)
            }
        };

        let discr = self.project_field(bx, discr_index);
        let lldiscr = bx.load_operand(discr).immediate();
        match *discr_kind {
            layout::DiscriminantKind::Tag => {
                let signed = match discr_scalar.value {
                    // We use `i1` for bytes that are always `0` or `1`,
                    // e.g., `#[repr(i8)] enum E { A, B }`, but we can't
                    // let LLVM interpret the `i1` as signed, because
                    // then `i1 1` (i.e., `E::B`) is effectively `i8 -1`.
                    layout::Int(_, signed) => !discr_scalar.is_bool() && signed,
                    _ => false
                };
                bx.intcast(lldiscr, cast_to, signed)
            }
            layout::DiscriminantKind::Niche {
                dataful_variant,
                ref niche_variants,
                niche_start,
            } => {
                let niche_llty = bx.cx().immediate_backend_type(discr.layout);
                if niche_variants.start() == niche_variants.end() {
                    // FIXME(eddyb): check the actual primitive type here.
                    let niche_llval = if niche_start == 0 {
                        // HACK(eddyb): using `c_null` as it works on all types.
                        bx.cx().const_null(niche_llty)
                    } else {
                        bx.cx().const_uint_big(niche_llty, niche_start)
                    };
                    let select_arg = bx.icmp(IntPredicate::IntEQ, lldiscr, niche_llval);
                    bx.select(select_arg,
                        bx.cx().const_uint(cast_to, niche_variants.start().as_u32() as u64),
                        bx.cx().const_uint(cast_to, dataful_variant.as_u32() as u64))
                } else {
                    // Rebase from niche values to discriminant values.
                    let delta = niche_start.wrapping_sub(niche_variants.start().as_u32() as u128);
                    let lldiscr = bx.sub(lldiscr, bx.cx().const_uint_big(niche_llty, delta));
                    let lldiscr_max =
                        bx.cx().const_uint(niche_llty, niche_variants.end().as_u32() as u64);
                    let select_arg = bx.icmp(IntPredicate::IntULE, lldiscr, lldiscr_max);
                    let cast = bx.intcast(lldiscr, cast_to, false);
                    bx.select(select_arg,
                        cast,
                        bx.cx().const_uint(cast_to, dataful_variant.as_u32() as u64))
                }
            }
        }
    }

    /// Sets the discriminant for a new value of the given case of the given
    /// representation.
    pub fn codegen_set_discr<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        &self,
        bx: &mut Bx,
        variant_index: VariantIdx
    ) {
        if self.layout.for_variant(bx.cx(), variant_index).abi.is_uninhabited() {
            return;
        }
        match self.layout.variants {
            layout::Variants::Single { index } => {
                assert_eq!(index, variant_index);
            }
            layout::Variants::Multiple {
                discr_kind: layout::DiscriminantKind::Tag,
                discr_index,
                ..
            } => {
                let ptr = self.project_field(bx, discr_index);
                let to =
                    self.layout.ty.discriminant_for_variant(bx.tcx(), variant_index).unwrap().val;
                bx.store(
                    bx.cx().const_uint_big(bx.cx().backend_type(ptr.layout), to),
                    ptr.llval,
                    ptr.align);
            }
            layout::Variants::Multiple {
                discr_kind: layout::DiscriminantKind::Niche {
                    dataful_variant,
                    ref niche_variants,
                    niche_start,
                },
                discr_index,
                ..
            } => {
                if variant_index != dataful_variant {
                    if bx.cx().sess().target.target.arch == "arm" ||
                       bx.cx().sess().target.target.arch == "aarch64" {
                        // FIXME(#34427): as workaround for LLVM bug on ARM,
                        // use memset of 0 before assigning niche value.
                        let fill_byte = bx.cx().const_u8(0);
                        let size = bx.cx().const_usize(self.layout.size.bytes());
                        bx.memset(self.llval, fill_byte, size, self.align, MemFlags::empty());
                    }

                    let niche = self.project_field(bx, discr_index);
                    let niche_llty = bx.cx().immediate_backend_type(niche.layout);
                    let niche_value = variant_index.as_u32() - niche_variants.start().as_u32();
                    let niche_value = (niche_value as u128)
                        .wrapping_add(niche_start);
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
        llindex: V
    ) -> Self {
        // Statically compute the offset if we can, otherwise just use the element size,
        // as this will yield the lowest alignment.
        let layout = self.layout.field(bx, 0);
        let offset = if bx.is_const_integral(llindex) {
            layout.size.checked_mul(bx.const_to_uint(llindex), bx).unwrap_or(layout.size)
        } else {
            layout.size
        };

        PlaceRef {
            llval: bx.inbounds_gep(self.llval, &[bx.cx().const_usize(0), llindex]),
            llextra: None,
            layout,
            align: self.align.restrict_for_offset(offset),
        }
    }

    pub fn project_downcast<Bx: BuilderMethods<'a, 'tcx, Value = V>>(
        &self,
        bx: &mut Bx,
        variant_index: VariantIdx
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
        place: &mir::Place<'tcx>
    ) -> PlaceRef<'tcx, Bx::Value> {
        debug!("codegen_place(place={:?})", place);

        let cx = self.cx;
        let tcx = self.cx.tcx();

        let result = match *place {
            mir::Place::Base(mir::PlaceBase::Local(index)) => {
                match self.locals[index] {
                    LocalRef::Place(place) => {
                        return place;
                    }
                    LocalRef::UnsizedPlace(place) => {
                        return bx.load_operand(place).deref(cx);
                    }
                    LocalRef::Operand(..) => {
                        bug!("using operand local {:?} as place", place);
                    }
                }
            }
            mir::Place::Base(
                mir::PlaceBase::Static(
                    box mir::Static { ty, kind: mir::StaticKind::Promoted(promoted) }
                )
            ) => {
                let param_env = ty::ParamEnv::reveal_all();
                let cid = mir::interpret::GlobalId {
                    instance: self.instance,
                    promoted: Some(promoted),
                };
                let layout = cx.layout_of(self.monomorphize(&ty));
                match bx.tcx().const_eval(param_env.and(cid)) {
                    Ok(val) => match val.val {
                        mir::interpret::ConstValue::ByRef { offset, align, alloc } => {
                            bx.cx().from_const_alloc(layout, align, alloc, offset)
                        }
                        _ => bug!("promoteds should have an allocation: {:?}", val),
                    },
                    Err(_) => {
                        // This is unreachable as long as runtime
                        // and compile-time agree on values
                        // With floats that won't always be true,
                        // so we generate an abort.
                        bx.abort();
                        let llval = bx.cx().const_undef(
                            bx.cx().type_ptr_to(bx.cx().backend_type(layout))
                        );
                        PlaceRef::new_sized(llval, layout, layout.align.abi)
                    }
                }
            }
            mir::Place::Base(
                mir::PlaceBase::Static(
                    box mir::Static { ty, kind: mir::StaticKind::Static(def_id) }
                )
            ) => {
                // NB: The layout of a static may be unsized as is the case when working
                // with a static that is an extern_type.
                let layout = cx.layout_of(self.monomorphize(&ty));
                let static_ = bx.get_static(def_id);
                PlaceRef::new_thin_place(bx, static_, layout, layout.align.abi)
            },
            mir::Place::Projection(box mir::Projection {
                ref base,
                elem: mir::ProjectionElem::Deref
            }) => {
                // Load the pointer from its location.
                self.codegen_consume(bx, base).deref(bx.cx())
            }
            mir::Place::Projection(ref projection) => {
                let cg_base = self.codegen_place(bx, &projection.base);

                match projection.elem {
                    mir::ProjectionElem::Deref => bug!(),
                    mir::ProjectionElem::Field(ref field, _) => {
                        cg_base.project_field(bx, field.index())
                    }
                    mir::ProjectionElem::Index(index) => {
                        let index = &mir::Operand::Copy(
                            mir::Place::from(index)
                        );
                        let index = self.codegen_operand(bx, index);
                        let llindex = index.immediate();
                        cg_base.project_index(bx, llindex)
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: false,
                                                         min_length: _ } => {
                        let lloffset = bx.cx().const_usize(offset as u64);
                        cg_base.project_index(bx, lloffset)
                    }
                    mir::ProjectionElem::ConstantIndex { offset,
                                                         from_end: true,
                                                         min_length: _ } => {
                        let lloffset = bx.cx().const_usize(offset as u64);
                        let lllen = cg_base.len(bx.cx());
                        let llindex = bx.sub(lllen, lloffset);
                        cg_base.project_index(bx, llindex)
                    }
                    mir::ProjectionElem::Subslice { from, to } => {
                        let mut subslice = cg_base.project_index(bx,
                            bx.cx().const_usize(from as u64));
                        let projected_ty = PlaceTy::from_ty(cg_base.layout.ty)
                            .projection_ty(tcx, &projection.elem).ty;
                        subslice.layout = bx.cx().layout_of(self.monomorphize(&projected_ty));

                        if subslice.layout.is_unsized() {
                            subslice.llextra = Some(bx.sub(cg_base.llextra.unwrap(),
                                bx.cx().const_usize((from as u64) + (to as u64))));
                        }

                        // Cast the place pointer type to the new
                        // array or slice type (`*[%_; new_len]`).
                        subslice.llval = bx.pointercast(subslice.llval,
                            bx.cx().type_ptr_to(bx.cx().backend_type(subslice.layout)));

                        subslice
                    }
                    mir::ProjectionElem::Downcast(_, v) => {
                        cg_base.project_downcast(bx, v)
                    }
                }
            }
        };
        debug!("codegen_place(place={:?}) => {:?}", place, result);
        result
    }

    pub fn monomorphized_place_ty(&self, place: &mir::Place<'tcx>) -> Ty<'tcx> {
        let tcx = self.cx.tcx();
        let place_ty = place.ty(self.mir, tcx);
        self.monomorphize(&place_ty.ty)
    }
}
