use std::fmt::Write;

use gccjit::{Struct, Type};
use crate::rustc_codegen_ssa::traits::{BaseTypeMethods, DerivedTypeMethods, LayoutTypeMethods};
use rustc_middle::bug;
use rustc_middle::ty::{self, Ty, TypeFoldable};
use rustc_middle::ty::layout::{FnAbiOf, LayoutOf, TyAndLayout};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_target::abi::{self, Abi, F32, F64, FieldsShape, Int, Integer, Pointer, PointeeInfo, Size, TyAbiInterface, Variants};
use rustc_target::abi::call::{CastTarget, FnAbi, Reg};

use crate::abi::{FnAbiGccExt, GccType};
use crate::context::CodegenCx;
use crate::type_::struct_fields;

impl<'gcc, 'tcx> CodegenCx<'gcc, 'tcx> {
    fn type_from_unsigned_integer(&self, i: Integer) -> Type<'gcc> {
        use Integer::*;
        match i {
            I8 => self.type_u8(),
            I16 => self.type_u16(),
            I32 => self.type_u32(),
            I64 => self.type_u64(),
            I128 => self.type_u128(),
        }
    }
}

pub fn uncached_gcc_type<'gcc, 'tcx>(cx: &CodegenCx<'gcc, 'tcx>, layout: TyAndLayout<'tcx>, defer: &mut Option<(Struct<'gcc>, TyAndLayout<'tcx>)>) -> Type<'gcc> {
    match layout.abi {
        Abi::Scalar(_) => bug!("handled elsewhere"),
        Abi::Vector { ref element, count } => {
            let element = layout.scalar_gcc_type_at(cx, element, Size::ZERO);
            return cx.context.new_vector_type(element, count);
        },
        Abi::ScalarPair(..) => {
            return cx.type_struct(
                &[
                    layout.scalar_pair_element_gcc_type(cx, 0, false),
                    layout.scalar_pair_element_gcc_type(cx, 1, false),
                ],
                false,
            );
        }
        Abi::Uninhabited | Abi::Aggregate { .. } => {}
    }

    let name = match layout.ty.kind() {
        // FIXME(eddyb) producing readable type names for trait objects can result
        // in problematically distinct types due to HRTB and subtyping (see #47638).
        // ty::Dynamic(..) |
        ty::Adt(..) | ty::Closure(..) | ty::Foreign(..) | ty::Generator(..) | ty::Str
            if !cx.sess().fewer_names() =>
        {
            let mut name = with_no_trimmed_paths(|| layout.ty.to_string());
            if let (&ty::Adt(def, _), &Variants::Single { index }) =
                (layout.ty.kind(), &layout.variants)
            {
                if def.is_enum() && !def.variants.is_empty() {
                    write!(&mut name, "::{}", def.variants[index].ident).unwrap();
                }
            }
            if let (&ty::Generator(_, _, _), &Variants::Single { index }) =
                (layout.ty.kind(), &layout.variants)
            {
                write!(&mut name, "::{}", ty::GeneratorSubsts::variant_name(index)).unwrap();
            }
            Some(name)
        }
        ty::Adt(..) => {
            // If `Some` is returned then a named struct is created in LLVM. Name collisions are
            // avoided by LLVM (with increasing suffixes). If rustc doesn't generate names then that
            // can improve perf.
            // FIXME(antoyo): I don't think that's true for libgccjit.
            Some(String::new())
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
                    let gcc_type = cx.type_named_struct(name);
                    cx.set_struct_body(gcc_type, &[fill], packed);
                    gcc_type.as_type()
                },
            }
        }
        FieldsShape::Array { count, .. } => cx.type_array(layout.field(cx, 0).gcc_type(cx, true), count),
        FieldsShape::Arbitrary { .. } =>
            match name {
                None => {
                    let (gcc_fields, packed) = struct_fields(cx, layout);
                    cx.type_struct(&gcc_fields, packed)
                },
                Some(ref name) => {
                    let gcc_type = cx.type_named_struct(name);
                    *defer = Some((gcc_type, layout));
                    gcc_type.as_type()
                },
            },
    }
}

pub trait LayoutGccExt<'tcx> {
    fn is_gcc_immediate(&self) -> bool;
    fn is_gcc_scalar_pair(&self) -> bool;
    fn gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, 'tcx>, set_fields: bool) -> Type<'gcc>;
    fn immediate_gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc>;
    fn scalar_gcc_type_at<'gcc>(&self, cx: &CodegenCx<'gcc, 'tcx>, scalar: &abi::Scalar, offset: Size) -> Type<'gcc>;
    fn scalar_pair_element_gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, 'tcx>, index: usize, immediate: bool) -> Type<'gcc>;
    fn gcc_field_index(&self, index: usize) -> u64;
    fn pointee_info_at<'gcc>(&self, cx: &CodegenCx<'gcc, 'tcx>, offset: Size) -> Option<PointeeInfo>;
}

impl<'tcx> LayoutGccExt<'tcx> for TyAndLayout<'tcx> {
    fn is_gcc_immediate(&self) -> bool {
        match self.abi {
            Abi::Scalar(_) | Abi::Vector { .. } => true,
            Abi::ScalarPair(..) => false,
            Abi::Uninhabited | Abi::Aggregate { .. } => self.is_zst(),
        }
    }

    fn is_gcc_scalar_pair(&self) -> bool {
        match self.abi {
            Abi::ScalarPair(..) => true,
            Abi::Uninhabited | Abi::Scalar(_) | Abi::Vector { .. } | Abi::Aggregate { .. } => false,
        }
    }

    /// Gets the GCC type corresponding to a Rust type, i.e., `rustc_middle::ty::Ty`.
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
    //TODO(antoyo): do we still need the set_fields parameter?
    fn gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, 'tcx>, set_fields: bool) -> Type<'gcc> {
        if let Abi::Scalar(ref scalar) = self.abi {
            // Use a different cache for scalars because pointers to DSTs
            // can be either fat or thin (data pointers of fat pointers).
            if let Some(&ty) = cx.scalar_types.borrow().get(&self.ty) {
                return ty;
            }
            let ty =
                match *self.ty.kind() {
                    ty::Ref(_, ty, _) | ty::RawPtr(ty::TypeAndMut { ty, .. }) => {
                        cx.type_ptr_to(cx.layout_of(ty).gcc_type(cx, set_fields))
                    }
                    ty::Adt(def, _) if def.is_box() => {
                        cx.type_ptr_to(cx.layout_of(self.ty.boxed_ty()).gcc_type(cx, true))
                    }
                    ty::FnPtr(sig) => cx.fn_ptr_backend_type(&cx.fn_abi_of_fn_ptr(sig, ty::List::empty())),
                    _ => self.scalar_gcc_type_at(cx, scalar, Size::ZERO),
                };
            cx.scalar_types.borrow_mut().insert(self.ty, ty);
            return ty;
        }

        // Check the cache.
        let variant_index =
            match self.variants {
                Variants::Single { index } => Some(index),
                _ => None,
            };
        let cached_type = cx.types.borrow().get(&(self.ty, variant_index)).cloned();
        if let Some(ty) = cached_type {
            let type_to_set_fields = cx.types_with_fields_to_set.borrow_mut().remove(&ty);
            if let Some((struct_type, layout)) = type_to_set_fields {
                // Since we might be trying to generate a type containing another type which is not
                // completely generated yet, we deferred setting the fields until now.
                let (fields, packed) = struct_fields(cx, layout);
                cx.set_struct_body(struct_type, &fields, packed);
            }
            return ty;
        }

        assert!(!self.ty.has_escaping_bound_vars(), "{:?} has escaping bound vars", self.ty);

        // Make sure lifetimes are erased, to avoid generating distinct LLVM
        // types for Rust types that only differ in the choice of lifetimes.
        let normal_ty = cx.tcx.erase_regions(self.ty);

        let mut defer = None;
        let ty =
            if self.ty != normal_ty {
                let mut layout = cx.layout_of(normal_ty);
                if let Some(v) = variant_index {
                    layout = layout.for_variant(cx, v);
                }
                layout.gcc_type(cx, true)
            }
            else {
                uncached_gcc_type(cx, *self, &mut defer)
            };

        cx.types.borrow_mut().insert((self.ty, variant_index), ty);

        if let Some((ty, layout)) = defer {
            let (fields, packed) = struct_fields(cx, layout);
            cx.set_struct_body(ty, &fields, packed);
        }

        ty
    }

    fn immediate_gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc> {
        if let Abi::Scalar(ref scalar) = self.abi {
            if scalar.is_bool() {
                return cx.type_i1();
            }
        }
        self.gcc_type(cx, true)
    }

    fn scalar_gcc_type_at<'gcc>(&self, cx: &CodegenCx<'gcc, 'tcx>, scalar: &abi::Scalar, offset: Size) -> Type<'gcc> {
        match scalar.value {
            Int(i, true) => cx.type_from_integer(i),
            Int(i, false) => cx.type_from_unsigned_integer(i),
            F32 => cx.type_f32(),
            F64 => cx.type_f64(),
            Pointer => {
                // If we know the alignment, pick something better than i8.
                let pointee =
                    if let Some(pointee) = self.pointee_info_at(cx, offset) {
                        cx.type_pointee_for_align(pointee.align)
                    }
                    else {
                        cx.type_i8()
                    };
                cx.type_ptr_to(pointee)
            }
        }
    }

    fn scalar_pair_element_gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, 'tcx>, index: usize, immediate: bool) -> Type<'gcc> {
        // TODO(antoyo): remove llvm hack:
        // HACK(eddyb) special-case fat pointers until LLVM removes
        // pointee types, to avoid bitcasting every `OperandRef::deref`.
        match self.ty.kind() {
            ty::Ref(..) | ty::RawPtr(_) => {
                return self.field(cx, index).gcc_type(cx, true);
            }
            ty::Adt(def, _) if def.is_box() => {
                let ptr_ty = cx.tcx.mk_mut_ptr(self.ty.boxed_ty());
                return cx.layout_of(ptr_ty).scalar_pair_element_gcc_type(cx, index, immediate);
            }
            _ => {}
        }

        let (a, b) = match self.abi {
            Abi::ScalarPair(ref a, ref b) => (a, b),
            _ => bug!("TyAndLayout::scalar_pair_element_llty({:?}): not applicable", self),
        };
        let scalar = [a, b][index];

        // Make sure to return the same type `immediate_gcc_type` would when
        // dealing with an immediate pair.  This means that `(bool, bool)` is
        // effectively represented as `{i8, i8}` in memory and two `i1`s as an
        // immediate, just like `bool` is typically `i8` in memory and only `i1`
        // when immediate.  We need to load/store `bool` as `i8` to avoid
        // crippling LLVM optimizations or triggering other LLVM bugs with `i1`.
        // TODO(antoyo): this bugs certainly don't happen in this case since the bool type is used instead of i1.
        if scalar.is_bool() {
            return cx.type_i1();
        }

        let offset =
            if index == 0 {
                Size::ZERO
            }
            else {
                a.value.size(cx).align_to(b.value.align(cx).abi)
            };
        self.scalar_gcc_type_at(cx, scalar, offset)
    }

    fn gcc_field_index(&self, index: usize) -> u64 {
        match self.abi {
            Abi::Scalar(_) | Abi::ScalarPair(..) => {
                bug!("TyAndLayout::gcc_field_index({:?}): not applicable", self)
            }
            _ => {}
        }
        match self.fields {
            FieldsShape::Primitive | FieldsShape::Union(_) => {
                bug!("TyAndLayout::gcc_field_index({:?}): not applicable", self)
            }

            FieldsShape::Array { .. } => index as u64,

            FieldsShape::Arbitrary { .. } => 1 + (self.fields.memory_index(index) as u64) * 2,
        }
    }

    fn pointee_info_at<'a>(&self, cx: &CodegenCx<'a, 'tcx>, offset: Size) -> Option<PointeeInfo> {
        if let Some(&pointee) = cx.pointee_infos.borrow().get(&(self.ty, offset)) {
            return pointee;
        }

        let result = Ty::ty_and_layout_pointee_info_at(*self, cx, offset);

        cx.pointee_infos.borrow_mut().insert((self.ty, offset), result);
        result
    }
}

impl<'gcc, 'tcx> LayoutTypeMethods<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn backend_type(&self, layout: TyAndLayout<'tcx>) -> Type<'gcc> {
        layout.gcc_type(self, true)
    }

    fn immediate_backend_type(&self, layout: TyAndLayout<'tcx>) -> Type<'gcc> {
        layout.immediate_gcc_type(self)
    }

    fn is_backend_immediate(&self, layout: TyAndLayout<'tcx>) -> bool {
        layout.is_gcc_immediate()
    }

    fn is_backend_scalar_pair(&self, layout: TyAndLayout<'tcx>) -> bool {
        layout.is_gcc_scalar_pair()
    }

    fn backend_field_index(&self, layout: TyAndLayout<'tcx>, index: usize) -> u64 {
        layout.gcc_field_index(index)
    }

    fn scalar_pair_element_backend_type(&self, layout: TyAndLayout<'tcx>, index: usize, immediate: bool) -> Type<'gcc> {
        layout.scalar_pair_element_gcc_type(self, index, immediate)
    }

    fn cast_backend_type(&self, ty: &CastTarget) -> Type<'gcc> {
        ty.gcc_type(self)
    }

    fn fn_ptr_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> Type<'gcc> {
        fn_abi.ptr_to_gcc_type(self)
    }

    fn reg_backend_type(&self, _ty: &Reg) -> Type<'gcc> {
        unimplemented!();
    }

    fn fn_decl_backend_type(&self, _fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> Type<'gcc> {
        // FIXME(antoyo): return correct type.
        self.type_void()
    }
}
