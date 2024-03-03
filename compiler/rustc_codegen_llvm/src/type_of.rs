use crate::common::*;
use crate::context::TypeLowering;
use crate::type_::Type;
use rustc_codegen_ssa::traits::*;
use rustc_middle::bug;
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::print::{with_no_trimmed_paths, with_no_visible_paths};
use rustc_middle::ty::{self, Ty, TypeVisitableExt};
use rustc_target::abi::HasDataLayout;
use rustc_target::abi::{Abi, Align, FieldsShape};
use rustc_target::abi::{Int, Pointer, F128, F16, F32, F64};
use rustc_target::abi::{Scalar, Size, Variants};
use smallvec::{smallvec, SmallVec};

use std::fmt::Write;

fn uncached_llvm_type<'a, 'tcx>(
    cx: &CodegenCx<'a, 'tcx>,
    layout: TyAndLayout<'tcx>,
    defer: &mut Option<(&'a Type, TyAndLayout<'tcx>)>,
    field_remapping: &mut Option<SmallVec<[u32; 4]>>,
) -> &'a Type {
    match layout.abi {
        Abi::Scalar(_) => bug!("handled elsewhere"),
        Abi::Vector { element, count } => {
            let element = layout.scalar_llvm_type_at(cx, element);
            return cx.type_vector(element, count);
        }
        Abi::Uninhabited | Abi::Aggregate { .. } | Abi::ScalarPair(..) => {}
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
                if def.is_enum() && !def.variants().is_empty() {
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
                let (llfields, packed, new_field_remapping) = struct_llfields(cx, layout);
                *field_remapping = new_field_remapping;
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
) -> (Vec<&'a Type>, bool, Option<SmallVec<[u32; 4]>>) {
    debug!("struct_llfields: {:#?}", layout);
    let field_count = layout.fields.count();

    let mut packed = false;
    let mut offset = Size::ZERO;
    let mut prev_effective_align = layout.align.abi;
    let mut result: Vec<_> = Vec::with_capacity(1 + field_count * 2);
    let mut field_remapping = smallvec![0; field_count];
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
        field_remapping[i] = result.len() as u32;
        result.push(field.llvm_type(cx));
        offset = target_offset + field.size;
        prev_effective_align = effective_field_align;
    }
    let padding_used = result.len() > field_count;
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
    let field_remapping = padding_used.then_some(field_remapping);
    (result, packed, field_remapping)
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

pub trait LayoutLlvmExt<'tcx> {
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
    fn scalar_copy_llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> Option<&'a Type>;
}

impl<'tcx> LayoutLlvmExt<'tcx> for TyAndLayout<'tcx> {
    fn is_llvm_immediate(&self) -> bool {
        match self.abi {
            Abi::Scalar(_) | Abi::Vector { .. } => true,
            Abi::ScalarPair(..) | Abi::Uninhabited | Abi::Aggregate { .. } => false,
        }
    }

    fn is_llvm_scalar_pair(&self) -> bool {
        match self.abi {
            Abi::ScalarPair(..) => true,
            Abi::Uninhabited | Abi::Scalar(_) | Abi::Vector { .. } | Abi::Aggregate { .. } => false,
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
    /// with the inner-most trailing unsized field using the "minimal unit"
    /// of that field's type - this is useful for taking the address of
    /// that field and ensuring the struct has the right alignment.
    fn llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type {
        // This must produce the same result for `repr(transparent)` wrappers as for the inner type!
        // In other words, this should generally not look at the type at all, but only at the
        // layout.
        if let Abi::Scalar(scalar) = self.abi {
            // Use a different cache for scalars because pointers to DSTs
            // can be either fat or thin (data pointers of fat pointers).
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
            return llty.lltype;
        }

        debug!("llvm_type({:#?})", self);

        assert!(!self.ty.has_escaping_bound_vars(), "{:?} has escaping bound vars", self.ty);

        // Make sure lifetimes are erased, to avoid generating distinct LLVM
        // types for Rust types that only differ in the choice of lifetimes.
        let normal_ty = cx.tcx.erase_regions(self.ty);

        let mut defer = None;
        let mut field_remapping = None;
        let llty = if self.ty != normal_ty {
            let mut layout = cx.layout_of(normal_ty);
            if let Some(v) = variant_index {
                layout = layout.for_variant(cx, v);
            }
            layout.llvm_type(cx)
        } else {
            uncached_llvm_type(cx, *self, &mut defer, &mut field_remapping)
        };
        debug!("--> mapped {:#?} to llty={:?}", self, llty);

        cx.type_lowering
            .borrow_mut()
            .insert((self.ty, variant_index), TypeLowering { lltype: llty, field_remapping });

        if let Some((llty, layout)) = defer {
            let (llfields, packed, new_field_remapping) = struct_llfields(cx, layout);
            cx.set_struct_body(llty, &llfields, packed);
            cx.type_lowering
                .borrow_mut()
                .get_mut(&(self.ty, variant_index))
                .unwrap()
                .field_remapping = new_field_remapping;
        }
        llty
    }

    fn immediate_llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> &'a Type {
        match self.abi {
            Abi::Scalar(scalar) => {
                if scalar.is_bool() {
                    return cx.type_i1();
                }
            }
            Abi::ScalarPair(..) => {
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
            F16 => cx.type_f16(),
            F32 => cx.type_f32(),
            F64 => cx.type_f64(),
            F128 => cx.type_f128(),
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
        let Abi::ScalarPair(a, b) = self.abi else {
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

    fn scalar_copy_llvm_type<'a>(&self, cx: &CodegenCx<'a, 'tcx>) -> Option<&'a Type> {
        debug_assert!(self.is_sized());

        // FIXME: this is a fairly arbitrary choice, but 128 bits on WASM
        // (matching the 128-bit SIMD types proposal) and 256 bits on x64
        // (like AVX2 registers) seems at least like a tolerable starting point.
        let threshold = cx.data_layout().pointer_size * 4;
        if self.layout.size() > threshold {
            return None;
        }

        // Vectors, even for non-power-of-two sizes, have the same layout as
        // arrays but don't count as aggregate types
        // While LLVM theoretically supports non-power-of-two sizes, and they
        // often work fine, sometimes x86-isel deals with them horribly
        // (see #115212) so for now only use power-of-two ones.
        if let FieldsShape::Array { count, .. } = self.layout.fields()
            && count.is_power_of_two()
            && let element = self.field(cx, 0)
            && element.ty.is_integral()
        {
            // `cx.type_ix(bits)` is tempting here, but while that works great
            // for things that *stay* as memory-to-memory copies, it also ends
            // up suppressing vectorization as it introduces shifts when it
            // extracts all the individual values.

            let ety = element.llvm_type(cx);
            if *count == 1 {
                // Emitting `<1 x T>` would be silly; just use the scalar.
                return Some(ety);
            } else {
                return Some(cx.type_vector(ety, *count));
            }
        }

        // FIXME: The above only handled integer arrays; surely more things
        // would also be possible. Be careful about provenance, though!
        None
    }
}
