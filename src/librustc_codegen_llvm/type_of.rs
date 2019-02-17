use abi::{FnType, FnTypeExt};
use common::*;
use rustc::hir;
use rustc::ty::{self, Ty, TypeFoldable};
use rustc::ty::layout::{self, Align, LayoutOf, Size, TyLayout};
use rustc_target::abi::FloatTy;
use rustc_mir::monomorphize::item::DefPathBasedNames;
use rustc_codegen_ssa::traits::*;
use type_::Type;

use std::fmt::Write;

fn uncached_llvm_type<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                                layout: TyLayout<'tcx>,
                                defer: &mut Option<(&'a Type, TyLayout<'tcx>)>)
                                -> &'a Type {
    match layout.abi {
        layout::Abi::Scalar(_) => bug!("handled elsewhere"),
        layout::Abi::Vector { ref element, count } => {
            // LLVM has a separate type for 64-bit SIMD vectors on X86 called
            // `x86_mmx` which is needed for some SIMD operations. As a bit of a
            // hack (all SIMD definitions are super unstable anyway) we
            // recognize any one-element SIMD vector as "this should be an
            // x86_mmx" type. In general there shouldn't be a need for other
            // one-element SIMD vectors, so it's assumed this won't clash with
            // much else.
            let use_x86_mmx = count == 1 && layout.size.bits() == 64 &&
                (cx.sess().target.target.arch == "x86" ||
                 cx.sess().target.target.arch == "x86_64");
            if use_x86_mmx {
                return cx.type_x86_mmx()
            } else {
                let element = layout.scalar_llvm_type_at(cx, element, Size::ZERO);
                return cx.type_vector(element, count);
            }
        }
        layout::Abi::ScalarPair(..) => {
            return cx.type_struct( &[
                layout.scalar_pair_element_llvm_type(cx, 0, false),
                layout.scalar_pair_element_llvm_type(cx, 1, false),
            ], false);
        }
        layout::Abi::Uninhabited |
        layout::Abi::Aggregate { .. } => {}
    }

    let name = match layout.ty.sty {
        ty::Closure(..) |
        ty::Generator(..) |
        ty::Adt(..) |
        // FIXME(eddyb) producing readable type names for trait objects can result
        // in problematically distinct types due to HRTB and subtyping (see #47638).
        // ty::Dynamic(..) |
        ty::Foreign(..) |
        ty::Str => {
            let mut name = String::with_capacity(32);
            let printer = DefPathBasedNames::new(cx.tcx, true, true);
            printer.push_type_name(layout.ty, &mut name, false);
            if let (&ty::Adt(def, _), &layout::Variants::Single { index })
                 = (&layout.ty.sty, &layout.variants)
            {
                if def.is_enum() && !def.variants.is_empty() {
                    write!(&mut name, "::{}", def.variants[index].ident).unwrap();
                }
            }
            Some(name)
        }
        _ => None
    };

    match layout.fields {
        layout::FieldPlacement::Union(_) => {
            let fill = cx.type_padding_filler(layout.size, layout.align.abi);
            let packed = false;
            match name {
                None => {
                    cx.type_struct(&[fill], packed)
                }
                Some(ref name) => {
                    let llty = cx.type_named_struct(name);
                    cx.set_struct_body(llty, &[fill], packed);
                    llty
                }
            }
        }
        layout::FieldPlacement::Array { count, .. } => {
            cx.type_array(layout.field(cx, 0).llvm_type(cx), count)
        }
        layout::FieldPlacement::Arbitrary { .. } => {
            match name {
                None => {
                    let (llfields, packed) = struct_llfields(cx, layout);
                    cx.type_struct( &llfields, packed)
                }
                Some(ref name) => {
                    let llty = cx.type_named_struct( name);
                    *defer = Some((llty, layout));
                    llty
                }
            }
        }
    }
}

fn struct_llfields<'a, 'tcx>(cx: &CodegenCx<'a, 'tcx>,
                             layout: TyLayout<'tcx>)
                             -> (Vec<&'a Type>, bool) {
    debug!("struct_llfields: {:#?}", layout);
    let field_count = layout.fields.count();

    let mut packed = false;
    let mut offset = Size::ZERO;
    let mut prev_effective_align = layout.align.abi;
    let mut result: Vec<_> = Vec::with_capacity(1 + field_count * 2);
    for i in layout.fields.index_by_increasing_offset() {
        let target_offset = layout.fields.offset(i as usize);
        let field = layout.field(cx, i);
        let effective_field_align = layout.align.abi
            .min(field.align.abi)
            .restrict_for_offset(target_offset);
        packed |= effective_field_align < field.align.abi;

        debug!("struct_llfields: {}: {:?} offset: {:?} target_offset: {:?} \
                effective_field_align: {}",
               i, field, offset, target_offset, effective_field_align.bytes());
        assert!(target_offset >= offset);
        let padding = target_offset - offset;
        let padding_align = prev_effective_align.min(effective_field_align);
        assert_eq!(offset.align_to(padding_align) + padding, target_offset);
        result.push(cx.type_padding_filler( padding, padding_align));
        debug!("    padding before: {:?}", padding);

        result.push(field.llvm_type(cx));
        offset = target_offset + field.size;
        prev_effective_align = effective_field_align;
    }
    if !layout.is_unsized() && field_count > 0 {
        if offset > layout.size {
            bug!("layout: {:#?} stride: {:?} offset: {:?}",
                 layout, layout.size, offset);
        }
        let padding = layout.size - offset;
        let padding_align = prev_effective_align;
        assert_eq!(offset.align_to(padding_align) + padding, layout.size);
        debug!("struct_llfields: pad_bytes: {:?} offset: {:?} stride: {:?}",
               padding, offset, layout.size);
        result.push(cx.type_padding_filler(padding, padding_align));
        assert_eq!(result.len(), 1 + field_count * 2);
    } else {
        debug!("struct_llfields: offset: {:?} stride: {:?}",
               offset, layout.size);
    }

    (result, packed)
}

impl<'a, 'tcx> CodegenCx<'a, 'tcx> {
    pub fn align_of(&self, ty: Ty<'tcx>) -> Align {
        self.layout_of(ty).align.abi
    }

    pub fn size_of(&self, ty: Ty<'tcx>) -> Size {
        self.layout_of(ty).size
    }

    pub fn size_and_align_of(&self, ty: Ty<'tcx>) -> (Size, Align) {
        let layout = self.layout_of(ty);
        (layout.size, layout.align.abi)
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum PointerKind {
    /// Most general case, we know no restrictions to tell LLVM.
    Shared,

    /// `&T` where `T` contains no `UnsafeCell`, is `noalias` and `readonly`.
    Frozen,

    /// `&mut T`, when we know `noalias` is safe for LLVM.
    UniqueBorrowed,

    /// `Box<T>`, unlike `UniqueBorrowed`, it also has `noalias` on returns.
    UniqueOwned
}

#[derive(Copy, Clone)]
pub struct PointeeInfo {
    pub size: Size,
    pub align: Align,
    pub safe: Option<PointerKind>,
}

pub trait LayoutLlvmExt<'tcx> {
    fn is_llvm_immediate(&self) -> bool;
    fn is_llvm_scalar_pair<'a>(&self) -> bool;
    fn llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type;
    fn immediate_llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type;
    fn scalar_llvm_type_at<'a>(&self, cx: &CodegenCx<'a, 'tcx>,
                               scalar: &layout::Scalar, offset: Size) -> &'a Type;
    fn scalar_pair_element_llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>,
                                         index: usize, immediate: bool) -> &'a Type;
    fn llvm_field_index(&self, index: usize) -> u64;
    fn pointee_info_at<'a>(&self, cx: &CodegenCx<'a, 'tcx>, offset: Size)
                           -> Option<PointeeInfo>;
}

impl<'tcx> LayoutLlvmExt<'tcx> for TyLayout<'tcx> {
    fn is_llvm_immediate(&self) -> bool {
        match self.abi {
            layout::Abi::Scalar(_) |
            layout::Abi::Vector { .. } => true,
            layout::Abi::ScalarPair(..) => false,
            layout::Abi::Uninhabited |
            layout::Abi::Aggregate { .. } => self.is_zst()
        }
    }

    fn is_llvm_scalar_pair<'a>(&self) -> bool {
        match self.abi {
            layout::Abi::ScalarPair(..) => true,
            layout::Abi::Uninhabited |
            layout::Abi::Scalar(_) |
            layout::Abi::Vector { .. } |
            layout::Abi::Aggregate { .. } => false
        }
    }

    /// Gets the LLVM type corresponding to a Rust type, i.e., `rustc::ty::Ty`.
    /// The pointee type of the pointer in `PlaceRef` is always this type.
    /// For sized types, it is also the right LLVM type for an `alloca`
    /// containing a value of that type, and most immediates (except `bool`).
    /// Unsized types, however, are represented by a "minimal unit", e.g.
    /// `[T]` becomes `T`, while `str` and `Trait` turn into `i8` - this
    /// is useful for indexing slices, as `&[T]`'s data pointer is `T*`.
    /// If the type is an unsized struct, the regular layout is generated,
    /// with the inner-most trailing unsized field using the "minimal unit"
    /// of that field's type - this is useful for taking the address of
    /// that field and ensuring the struct has the right alignment.
    fn llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type {
        if let layout::Abi::Scalar(ref scalar) = self.abi {
            // Use a different cache for scalars because pointers to DSTs
            // can be either fat or thin (data pointers of fat pointers).
            if let Some(&llty) = cx.scalar_lltypes.borrow().get(&self.ty) {
                return llty;
            }
            let llty = match self.ty.sty {
                ty::Ref(_, ty, _) |
                ty::RawPtr(ty::TypeAndMut { ty, .. }) => {
                    cx.type_ptr_to(cx.layout_of(ty).llvm_type(cx))
                }
                ty::Adt(def, _) if def.is_box() => {
                    cx.type_ptr_to(cx.layout_of(self.ty.boxed_ty()).llvm_type(cx))
                }
                ty::FnPtr(sig) => {
                    let sig = cx.tcx.normalize_erasing_late_bound_regions(
                        ty::ParamEnv::reveal_all(),
                        &sig,
                    );
                    cx.fn_ptr_backend_type(&FnType::new(cx, sig, &[]))
                }
                _ => self.scalar_llvm_type_at(cx, scalar, Size::ZERO)
            };
            cx.scalar_lltypes.borrow_mut().insert(self.ty, llty);
            return llty;
        }


        // Check the cache.
        let variant_index = match self.variants {
            layout::Variants::Single { index } => Some(index),
            _ => None
        };
        if let Some(&llty) = cx.lltypes.borrow().get(&(self.ty, variant_index)) {
            return llty;
        }

        debug!("llvm_type({:#?})", self);

        assert!(!self.ty.has_escaping_bound_vars(), "{:?} has escaping bound vars", self.ty);

        // Make sure lifetimes are erased, to avoid generating distinct LLVM
        // types for Rust types that only differ in the choice of lifetimes.
        let normal_ty = cx.tcx.erase_regions(&self.ty);

        let mut defer = None;
        let llty = if self.ty != normal_ty {
            let mut layout = cx.layout_of(normal_ty);
            if let Some(v) = variant_index {
                layout = layout.for_variant(cx, v);
            }
            layout.llvm_type(cx)
        } else {
            uncached_llvm_type(cx, *self, &mut defer)
        };
        debug!("--> mapped {:#?} to llty={:?}", self, llty);

        cx.lltypes.borrow_mut().insert((self.ty, variant_index), llty);

        if let Some((llty, layout)) = defer {
            let (llfields, packed) = struct_llfields(cx, layout);
            cx.set_struct_body(llty, &llfields, packed)
        }

        llty
    }

    fn immediate_llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type {
        if let layout::Abi::Scalar(ref scalar) = self.abi {
            if scalar.is_bool() {
                return cx.type_i1();
            }
        }
        self.llvm_type(cx)
    }

    fn scalar_llvm_type_at<'a>(&self, cx: &CodegenCx<'a, 'tcx>,
                               scalar: &layout::Scalar, offset: Size) -> &'a Type {
        match scalar.value {
            layout::Int(i, _) => cx.type_from_integer( i),
            layout::Float(FloatTy::F32) => cx.type_f32(),
            layout::Float(FloatTy::F64) => cx.type_f64(),
            layout::Pointer => {
                // If we know the alignment, pick something better than i8.
                let pointee = if let Some(pointee) = self.pointee_info_at(cx, offset) {
                    cx.type_pointee_for_align(pointee.align)
                } else {
                    cx.type_i8()
                };
                cx.type_ptr_to(pointee)
            }
        }
    }

    fn scalar_pair_element_llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>,
                                         index: usize, immediate: bool) -> &'a Type {
        // HACK(eddyb) special-case fat pointers until LLVM removes
        // pointee types, to avoid bitcasting every `OperandRef::deref`.
        match self.ty.sty {
            ty::Ref(..) |
            ty::RawPtr(_) => {
                return self.field(cx, index).llvm_type(cx);
            }
            ty::Adt(def, _) if def.is_box() => {
                let ptr_ty = cx.tcx.mk_mut_ptr(self.ty.boxed_ty());
                return cx.layout_of(ptr_ty).scalar_pair_element_llvm_type(cx, index, immediate);
            }
            _ => {}
        }

        let (a, b) = match self.abi {
            layout::Abi::ScalarPair(ref a, ref b) => (a, b),
            _ => bug!("TyLayout::scalar_pair_element_llty({:?}): not applicable", self)
        };
        let scalar = [a, b][index];

        // Make sure to return the same type `immediate_llvm_type` would when
        // dealing with an immediate pair.  This means that `(bool, bool)` is
        // effectively represented as `{i8, i8}` in memory and two `i1`s as an
        // immediate, just like `bool` is typically `i8` in memory and only `i1`
        // when immediate.  We need to load/store `bool` as `i8` to avoid
        // crippling LLVM optimizations or triggering other LLVM bugs with `i1`.
        if immediate && scalar.is_bool() {
            return cx.type_i1();
        }

        let offset = if index == 0 {
            Size::ZERO
        } else {
            a.value.size(cx).align_to(b.value.align(cx).abi)
        };
        self.scalar_llvm_type_at(cx, scalar, offset)
    }

    fn llvm_field_index(&self, index: usize) -> u64 {
        match self.abi {
            layout::Abi::Scalar(_) |
            layout::Abi::ScalarPair(..) => {
                bug!("TyLayout::llvm_field_index({:?}): not applicable", self)
            }
            _ => {}
        }
        match self.fields {
            layout::FieldPlacement::Union(_) => {
                bug!("TyLayout::llvm_field_index({:?}): not applicable", self)
            }

            layout::FieldPlacement::Array { .. } => {
                index as u64
            }

            layout::FieldPlacement::Arbitrary { .. } => {
                1 + (self.fields.memory_index(index) as u64) * 2
            }
        }
    }

    fn pointee_info_at<'a>(&self, cx: &CodegenCx<'a, 'tcx>, offset: Size)
                           -> Option<PointeeInfo> {
        if let Some(&pointee) = cx.pointee_infos.borrow().get(&(self.ty, offset)) {
            return pointee;
        }

        let mut result = None;
        match self.ty.sty {
            ty::RawPtr(mt) if offset.bytes() == 0 => {
                let (size, align) = cx.size_and_align_of(mt.ty);
                result = Some(PointeeInfo {
                    size,
                    align,
                    safe: None
                });
            }

            ty::Ref(_, ty, mt) if offset.bytes() == 0 => {
                let (size, align) = cx.size_and_align_of(ty);

                let kind = match mt {
                    hir::MutImmutable => if cx.type_is_freeze(ty) {
                        PointerKind::Frozen
                    } else {
                        PointerKind::Shared
                    },
                    hir::MutMutable => {
                        // Previously we would only emit noalias annotations for LLVM >= 6 or in
                        // panic=abort mode. That was deemed right, as prior versions had many bugs
                        // in conjunction with unwinding, but later versions didn’t seem to have
                        // said issues. See issue #31681.
                        //
                        // Alas, later on we encountered a case where noalias would generate wrong
                        // code altogether even with recent versions of LLVM in *safe* code with no
                        // unwinding involved. See #54462.
                        //
                        // For now, do not enable mutable_noalias by default at all, while the
                        // issue is being figured out.
                        let mutable_noalias = cx.tcx.sess.opts.debugging_opts.mutable_noalias
                            .unwrap_or(false);
                        if mutable_noalias {
                            PointerKind::UniqueBorrowed
                        } else {
                            PointerKind::Shared
                        }
                    }
                };

                result = Some(PointeeInfo {
                    size,
                    align,
                    safe: Some(kind)
                });
            }

            _ => {
                let mut data_variant = match self.variants {
                    layout::Variants::NicheFilling { dataful_variant, .. } => {
                        // Only the niche itself is always initialized,
                        // so only check for a pointer at its offset.
                        //
                        // If the niche is a pointer, it's either valid
                        // (according to its type), or null (which the
                        // niche field's scalar validity range encodes).
                        // This allows using `dereferenceable_or_null`
                        // for e.g., `Option<&T>`, and this will continue
                        // to work as long as we don't start using more
                        // niches than just null (e.g., the first page
                        // of the address space, or unaligned pointers).
                        if self.fields.offset(0) == offset {
                            Some(self.for_variant(cx, dataful_variant))
                        } else {
                            None
                        }
                    }
                    _ => Some(*self)
                };

                if let Some(variant) = data_variant {
                    // We're not interested in any unions.
                    if let layout::FieldPlacement::Union(_) = variant.fields {
                        data_variant = None;
                    }
                }

                if let Some(variant) = data_variant {
                    let ptr_end = offset + layout::Pointer.size(cx);
                    for i in 0..variant.fields.count() {
                        let field_start = variant.fields.offset(i);
                        if field_start <= offset {
                            let field = variant.field(cx, i);
                            if ptr_end <= field_start + field.size {
                                // We found the right field, look inside it.
                                result = field.pointee_info_at(cx, offset - field_start);
                                break;
                            }
                        }
                    }
                }

                // FIXME(eddyb) This should be for `ptr::Unique<T>`, not `Box<T>`.
                if let Some(ref mut pointee) = result {
                    if let ty::Adt(def, _) = self.ty.sty {
                        if def.is_box() && offset.bytes() == 0 {
                            pointee.safe = Some(PointerKind::UniqueOwned);
                        }
                    }
                }
            }
        }

        cx.pointee_infos.borrow_mut().insert((self.ty, offset), result);
        result
    }
}
