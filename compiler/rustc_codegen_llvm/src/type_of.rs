use std::fmt::Write;

use rustc_abi::Primitive::{Float, Int, Pointer};
use rustc_abi::{Align, BackendRepr, FieldsShape, Scalar, Size, Variants};
use rustc_codegen_ssa::traits::*;
use rustc_middle::bug;
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::print::{with_no_trimmed_paths, with_no_visible_paths};
use rustc_middle::ty::{self, CoroutineArgsExt, Ty, TypeVisitableExt};
use rustc_span::{DUMMY_SP, Span};
use tracing::debug;

use crate::common::*;
use crate::type_::Type;

fn uncached_llvm_type<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    layout: TyAndLayout<'tcx>,
    defer: &mut Option<(&'a Type, TyAndLayout<'tcx>)>,
) -> &'a Type {
    match layout.backend_repr {
        BackendRepr::Scalar(_) => bug!("handled elsewhere"),
        BackendRepr::SimdVector { element, count } => {
            let element = layout.scalar_llvm_type_at(cx, element);
            return cx.type_vector(element, count);
        }
        BackendRepr::Memory { .. } | BackendRepr::ScalarPair(..) => {}
    }

    let name = match layout.ty.kind() {
        // FIXME(eddyb) producing readable type names for trait objects can result
        // in problematically distinct types due to HRTB and subtyping (see #47638).
        // ty::Dynamic(..) |
        ty::Adt(..) | ty::Closure(..) | ty::CoroutineClosure(..) | ty::Foreign(..) | ty::Coroutine(..) | ty::Str
            // For performance reasons we use names only when emitting LLVM IR.
            if !cx.sess().fewer_names() =>
        {
            let mut name = with_no_visible_paths!(with_no_trimmed_paths!(layout.ty.to_string()));
            if let (&ty::Adt(def, _), &Variants::Single { index }) =
                (layout.ty.kind(), &layout.variants)
            {
                if def.is_enum() {
                    write!(&mut name, "::{}", def.variant(index).name).unwrap();
                }
            }
            if let (&ty::Coroutine(_, _), &Variants::Single { index }) =
                (layout.ty.kind(), &layout.variants)
            {
                write!(&mut name, "::{}", ty::CoroutineArgs::variant_name(index)).unwrap();
            }
            Some(name)
        }
        _ => None,
    };

    match layout.fields {
        FieldsShape::Primitive | FieldsShape::Union(_) => {
            let fill = cx.type_padding_filler(layout.size, layout.align.abi);
            let packed = false;
            match name {
                None => cx.type_struct(&[fill], packed),
                Some(ref name) => {
                    let llty = cx.type_named_struct(name);
                    cx.set_struct_body(llty, &[fill], packed);
                    llty
                }
            }
        }
        FieldsShape::Array { count, .. } => cx.type_array(layout.field(cx, 0).llvm_type(cx), count),
        FieldsShape::Arbitrary { .. } => match name {
            None => {
                let (llfields, packed) = struct_llfields(cx, layout);
                cx.type_struct(&llfields, packed)
            }
            Some(ref name) => {
                let llty = cx.type_named_struct(name);
                *defer = Some((llty, layout));
                llty
            }
        },
    }
}

fn struct_llfields<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    layout: TyAndLayout<'tcx>,
) -> (Vec<&'a Type>, bool) {
    debug!("struct_llfields: {:#?}", layout);
    let field_count = layout.fields.count();

    let mut packed = false;
    let mut offset = Size::ZERO;
    let mut prev_effective_align = layout.align.abi;
    let mut result: Vec<_> = Vec::with_capacity(1 + field_count * 2);
    for i in layout.fields.index_by_increasing_offset() {
        let target_offset = layout.fields.offset(i as usize);
        let field = layout.field(cx, i);
        let effective_field_align =
            layout.align.abi.min(field.align.abi).restrict_for_offset(target_offset);
        packed |= effective_field_align < field.align.abi;

        debug!(
            "struct_llfields: {}: {:?} offset: {:?} target_offset: {:?} \
                effective_field_align: {}",
            i,
            field,
            offset,
            target_offset,
            effective_field_align.bytes()
        );
        assert!(target_offset >= offset);
        let padding = target_offset - offset;
        if padding != Size::ZERO {
            let padding_align = prev_effective_align.min(effective_field_align);
            assert_eq!(offset.align_to(padding_align) + padding, target_offset);
            result.push(cx.type_padding_filler(padding, padding_align));
            debug!("    padding before: {:?}", padding);
        }
        result.push(field.llvm_type(cx));
        offset = target_offset + field.size;
        prev_effective_align = effective_field_align;
    }
    if layout.is_sized() && field_count > 0 {
        if offset > layout.size {
            bug!("layout: {:#?} stride: {:?} offset: {:?}", layout, layout.size, offset);
        }
        let padding = layout.size - offset;
        if padding != Size::ZERO {
            let padding_align = prev_effective_align;
            assert_eq!(offset.align_to(padding_align) + padding, layout.size);
            debug!(
                "struct_llfields: pad_bytes: {:?} offset: {:?} stride: {:?}",
                padding, offset, layout.size
            );
            result.push(cx.type_padding_filler(padding, padding_align));
        }
    } else {
        debug!("struct_llfields: offset: {:?} stride: {:?}", offset, layout.size);
    }
    (result, packed)
}

impl<'a, 'tcx> CodegenCx<'a, 'tcx> {
    pub(crate) fn align_of(&self, ty: Ty<'tcx>) -> Align {
        self.layout_of(ty).align.abi
    }

    pub(crate) fn size_of(&self, ty: Ty<'tcx>) -> Size {
        self.layout_of(ty).size
    }

    pub(crate) fn size_and_align_of(&self, ty: Ty<'tcx>) -> (Size, Align) {
        self.spanned_size_and_align_of(ty, DUMMY_SP)
    }

    pub(crate) fn spanned_size_and_align_of(&self, ty: Ty<'tcx>, span: Span) -> (Size, Align) {
        let layout = self.spanned_layout_of(ty, span);
        (layout.size, layout.align.abi)
    }
}

pub(crate) trait LayoutLlvmExt<'tcx> {
    fn is_llvm_immediate(&self) -> bool;
    fn is_llvm_scalar_pair(&self) -> bool;
    fn llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type;
    fn immediate_llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type;
    fn scalar_llvm_type_at<'a>(&self, cx: &CodegenCx<'a, 'tcx>, scalar: Scalar) -> &'a Type;
    fn scalar_pair_element_llvm_type<'a>(
        &self,
        cx: &CodegenCx<'a, 'tcx>,
        index: usize,
        immediate: bool,
    ) -> &'a Type;
}

impl<'tcx> LayoutLlvmExt<'tcx> for TyAndLayout<'tcx> {
    fn is_llvm_immediate(&self) -> bool {
        match self.backend_repr {
            BackendRepr::Scalar(_) | BackendRepr::SimdVector { .. } => true,
            BackendRepr::ScalarPair(..) | BackendRepr::Memory { .. } => false,
        }
    }

    fn is_llvm_scalar_pair(&self) -> bool {
        match self.backend_repr {
            BackendRepr::ScalarPair(..) => true,
            BackendRepr::Scalar(_)
            | BackendRepr::SimdVector { .. }
            | BackendRepr::Memory { .. } => false,
        }
    }

    /// Gets the LLVM type corresponding to a Rust type, i.e., `rustc_middle::ty::Ty`.
    /// The pointee type of the pointer in `PlaceRef` is always this type.
    /// For sized types, it is also the right LLVM type for an `alloca`
    /// containing a value of that type, and most immediates (except `bool`).
    /// Unsized types, however, are represented by a "minimal unit", e.g.
    /// `[T]` becomes `T`, while `str` and `Trait` turn into `i8` - this
    /// is useful for indexing slices, as `&[T]`'s data pointer is `T*`.
    /// If the type is an unsized struct, the regular layout is generated,
    /// with the innermost trailing unsized field using the "minimal unit"
    /// of that field's type - this is useful for taking the address of
    /// that field and ensuring the struct has the right alignment.
    fn llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type {
        // This must produce the same result for `repr(transparent)` wrappers as for the inner type!
        // In other words, this should generally not look at the type at all, but only at the
        // layout.
        if let BackendRepr::Scalar(scalar) = self.backend_repr {
            // Use a different cache for scalars because pointers to DSTs
            // can be either wide or thin (data pointers of wide pointers).
            if let Some(&llty) = cx.scalar_lltypes.borrow().get(&self.ty) {
                return llty;
            }
            let llty = self.scalar_llvm_type_at(cx, scalar);
            cx.scalar_lltypes.borrow_mut().insert(self.ty, llty);
            return llty;
        }

        // Check the cache.
        let variant_index = match self.variants {
            Variants::Single { index } => Some(index),
            _ => None,
        };
        if let Some(llty) = cx.type_lowering.borrow().get(&(self.ty, variant_index)) {
            return llty;
        }

        debug!("llvm_type({:#?})", self);

        assert!(!self.ty.has_escaping_bound_vars(), "{:?} has escaping bound vars", self.ty);

        // Make sure lifetimes are erased, to avoid generating distinct LLVM
        // types for Rust types that only differ in the choice of lifetimes.
        let normal_ty = cx.tcx.erase_and_anonymize_regions(self.ty);

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

        cx.type_lowering.borrow_mut().insert((self.ty, variant_index), llty);

        if let Some((llty, layout)) = defer {
            let (llfields, packed) = struct_llfields(cx, layout);
            cx.set_struct_body(llty, &llfields, packed);
        }
        llty
    }

    fn immediate_llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type {
        match self.backend_repr {
            BackendRepr::Scalar(scalar) => {
                if scalar.is_bool() {
                    return cx.type_i1();
                }
            }
            BackendRepr::ScalarPair(..) => {
                // An immediate pair always contains just the two elements, without any padding
                // filler, as it should never be stored to memory.
                return cx.type_struct(
                    &[
                        self.scalar_pair_element_llvm_type(cx, 0, true),
                        self.scalar_pair_element_llvm_type(cx, 1, true),
                    ],
                    false,
                );
            }
            _ => {}
        };
        self.llvm_type(cx)
    }

    fn scalar_llvm_type_at<'a>(&self, cx: &CodegenCx<'a, 'tcx>, scalar: Scalar) -> &'a Type {
        match scalar.primitive() {
            Int(i, _) => cx.type_from_integer(i),
            Float(f) => cx.type_from_float(f),
            Pointer(address_space) => cx.type_ptr_ext(address_space),
        }
    }

    fn scalar_pair_element_llvm_type<'a>(
        &self,
        cx: &CodegenCx<'a, 'tcx>,
        index: usize,
        immediate: bool,
    ) -> &'a Type {
        // This must produce the same result for `repr(transparent)` wrappers as for the inner type!
        // In other words, this should generally not look at the type at all, but only at the
        // layout.
        let BackendRepr::ScalarPair(a, b) = self.backend_repr else {
            bug!("TyAndLayout::scalar_pair_element_llty({:?}): not applicable", self);
        };
        let scalar = [a, b][index];

        // Make sure to return the same type `immediate_llvm_type` would when
        // dealing with an immediate pair. This means that `(bool, bool)` is
        // effectively represented as `{i8, i8}` in memory and two `i1`s as an
        // immediate, just like `bool` is typically `i8` in memory and only `i1`
        // when immediate. We need to load/store `bool` as `i8` to avoid
        // crippling LLVM optimizations or triggering other LLVM bugs with `i1`.
        if immediate && scalar.is_bool() {
            return cx.type_i1();
        }

        self.scalar_llvm_type_at(cx, scalar)
    }
}
