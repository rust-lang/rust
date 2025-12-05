use std::fmt::Write;

use gccjit::{Struct, Type};
use rustc_abi as abi;
use rustc_abi::Primitive::*;
use rustc_abi::{
    BackendRepr, FieldsShape, Integer, PointeeInfo, Reg, Size, TyAbiInterface, Variants,
};
use rustc_codegen_ssa::traits::{
    BaseTypeCodegenMethods, DerivedTypeCodegenMethods, LayoutTypeCodegenMethods,
};
use rustc_middle::bug;
use rustc_middle::ty::layout::{LayoutOf, TyAndLayout};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, CoroutineArgsExt, Ty, TypeVisitableExt};
use rustc_target::callconv::{CastTarget, FnAbi};

use crate::abi::{FnAbiGcc, FnAbiGccExt, GccType};
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

    #[cfg(feature = "master")]
    pub fn type_int_from_ty(&self, t: ty::IntTy) -> Type<'gcc> {
        match t {
            ty::IntTy::Isize => self.type_isize(),
            ty::IntTy::I8 => self.type_i8(),
            ty::IntTy::I16 => self.type_i16(),
            ty::IntTy::I32 => self.type_i32(),
            ty::IntTy::I64 => self.type_i64(),
            ty::IntTy::I128 => self.type_i128(),
        }
    }

    #[cfg(feature = "master")]
    pub fn type_uint_from_ty(&self, t: ty::UintTy) -> Type<'gcc> {
        match t {
            ty::UintTy::Usize => self.type_isize(),
            ty::UintTy::U8 => self.type_i8(),
            ty::UintTy::U16 => self.type_i16(),
            ty::UintTy::U32 => self.type_i32(),
            ty::UintTy::U64 => self.type_i64(),
            ty::UintTy::U128 => self.type_i128(),
        }
    }
}

fn uncached_gcc_type<'gcc, 'tcx>(
    cx: &CodegenCx<'gcc, 'tcx>,
    layout: TyAndLayout<'tcx>,
    defer: &mut Option<(Struct<'gcc>, TyAndLayout<'tcx>)>,
) -> Type<'gcc> {
    match layout.backend_repr {
        BackendRepr::Scalar(_) => bug!("handled elsewhere"),
        BackendRepr::SimdVector { ref element, count } => {
            let element = layout.scalar_gcc_type_at(cx, element, Size::ZERO);
            let element =
                // NOTE: gcc doesn't allow pointer types in vectors.
                if element.get_pointee().is_some() {
                    cx.usize_type
                }
                else {
                    element
                };
            return cx.context.new_vector_type(element, count);
        }
        BackendRepr::ScalarPair(..) => {
            return cx.type_struct(
                &[
                    layout.scalar_pair_element_gcc_type(cx, 0),
                    layout.scalar_pair_element_gcc_type(cx, 1),
                ],
                false,
            );
        }
        BackendRepr::Memory { .. } => {}
    }

    let name = match *layout.ty.kind() {
        // FIXME(eddyb) producing readable type names for trait objects can result
        // in problematically distinct types due to HRTB and subtyping (see #47638).
        // ty::Dynamic(..) |
        ty::Adt(..)
        | ty::Closure(..)
        | ty::CoroutineClosure(..)
        | ty::Foreign(..)
        | ty::Coroutine(..)
        | ty::Str
            if !cx.sess().fewer_names() =>
        {
            let mut name = with_no_trimmed_paths!(layout.ty.to_string());
            if let (&ty::Adt(def, _), &Variants::Single { index }) =
                (layout.ty.kind(), &layout.variants)
                && def.is_enum()
                && !def.variants().is_empty()
            {
                write!(&mut name, "::{}", def.variant(index).name).unwrap();
            }
            if let (&ty::Coroutine(_, _), &Variants::Single { index }) =
                (layout.ty.kind(), &layout.variants)
            {
                write!(&mut name, "::{}", ty::CoroutineArgs::variant_name(index)).unwrap();
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
                }
            }
        }
        FieldsShape::Array { count, .. } => cx.type_array(layout.field(cx, 0).gcc_type(cx), count),
        FieldsShape::Arbitrary { .. } => match name {
            None => {
                let (gcc_fields, packed) = struct_fields(cx, layout);
                cx.type_struct(&gcc_fields, packed)
            }
            Some(ref name) => {
                let gcc_type = cx.type_named_struct(name);
                *defer = Some((gcc_type, layout));
                gcc_type.as_type()
            }
        },
    }
}

pub trait LayoutGccExt<'tcx> {
    fn is_gcc_immediate(&self) -> bool;
    fn is_gcc_scalar_pair(&self) -> bool;
    fn gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc>;
    fn immediate_gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc>;
    fn scalar_gcc_type_at<'gcc>(
        &self,
        cx: &CodegenCx<'gcc, 'tcx>,
        scalar: &abi::Scalar,
        offset: Size,
    ) -> Type<'gcc>;
    fn scalar_pair_element_gcc_type<'gcc>(
        &self,
        cx: &CodegenCx<'gcc, 'tcx>,
        index: usize,
    ) -> Type<'gcc>;
    fn pointee_info_at<'gcc>(
        &self,
        cx: &CodegenCx<'gcc, 'tcx>,
        offset: Size,
    ) -> Option<PointeeInfo>;
}

impl<'tcx> LayoutGccExt<'tcx> for TyAndLayout<'tcx> {
    fn is_gcc_immediate(&self) -> bool {
        match self.backend_repr {
            BackendRepr::Scalar(_) | BackendRepr::SimdVector { .. } => true,
            BackendRepr::ScalarPair(..) | BackendRepr::Memory { .. } => false,
        }
    }

    fn is_gcc_scalar_pair(&self) -> bool {
        match self.backend_repr {
            BackendRepr::ScalarPair(..) => true,
            BackendRepr::Scalar(_)
            | BackendRepr::SimdVector { .. }
            | BackendRepr::Memory { .. } => false,
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
    /// with the innermost trailing unsized field using the "minimal unit"
    /// of that field's type - this is useful for taking the address of
    /// that field and ensuring the struct has the right alignment.
    fn gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc> {
        use rustc_middle::ty::layout::FnAbiOf;
        // This must produce the same result for `repr(transparent)` wrappers as for the inner type!
        // In other words, this should generally not look at the type at all, but only at the
        // layout.
        if let BackendRepr::Scalar(ref scalar) = self.backend_repr {
            // Use a different cache for scalars because pointers to DSTs
            // can be either wide or thin (data pointers of wide pointers).
            if let Some(&ty) = cx.scalar_types.borrow().get(&self.ty) {
                return ty;
            }
            let ty = match *self.ty.kind() {
                // NOTE: we cannot remove this match like in the LLVM codegen because the call
                // to fn_ptr_backend_type handle the on-stack attribute.
                // TODO(antoyo): find a less hackish way to handle the on-stack attribute.
                ty::FnPtr(sig_tys, hdr) => cx
                    .fn_ptr_backend_type(cx.fn_abi_of_fn_ptr(sig_tys.with(hdr), ty::List::empty())),
                _ => self.scalar_gcc_type_at(cx, scalar, Size::ZERO),
            };
            cx.scalar_types.borrow_mut().insert(self.ty, ty);
            return ty;
        }

        // Check the cache.
        let variant_index = match self.variants {
            Variants::Single { index } => Some(index),
            _ => None,
        };
        let cached_type = cx.types.borrow().get(&(self.ty, variant_index)).cloned();
        if let Some(ty) = cached_type {
            return ty;
        }

        assert!(!self.ty.has_escaping_bound_vars(), "{:?} has escaping bound vars", self.ty);

        // Make sure lifetimes are erased, to avoid generating distinct LLVM
        // types for Rust types that only differ in the choice of lifetimes.
        let normal_ty = cx.tcx.erase_and_anonymize_regions(self.ty);

        let mut defer = None;
        let ty = if self.ty != normal_ty {
            let mut layout = cx.layout_of(normal_ty);
            if let Some(v) = variant_index {
                layout = layout.for_variant(cx, v);
            }
            layout.gcc_type(cx)
        } else {
            uncached_gcc_type(cx, *self, &mut defer)
        };

        cx.types.borrow_mut().insert((self.ty, variant_index), ty);

        if let Some((deferred_ty, layout)) = defer {
            let (fields, packed) = struct_fields(cx, layout);
            cx.set_struct_body(deferred_ty, &fields, packed);
        }

        ty
    }

    fn immediate_gcc_type<'gcc>(&self, cx: &CodegenCx<'gcc, 'tcx>) -> Type<'gcc> {
        if let BackendRepr::Scalar(ref scalar) = self.backend_repr
            && scalar.is_bool()
        {
            return cx.type_i1();
        }
        self.gcc_type(cx)
    }

    fn scalar_gcc_type_at<'gcc>(
        &self,
        cx: &CodegenCx<'gcc, 'tcx>,
        scalar: &abi::Scalar,
        offset: Size,
    ) -> Type<'gcc> {
        match scalar.primitive() {
            Int(i, true) => cx.type_from_integer(i),
            Int(i, false) => cx.type_from_unsigned_integer(i),
            Float(f) => cx.type_from_float(f),
            Pointer(address_space) => {
                // If we know the alignment, pick something better than i8.
                let pointee = if let Some(pointee) = self.pointee_info_at(cx, offset) {
                    cx.type_pointee_for_align(pointee.align)
                } else {
                    cx.type_i8()
                };
                cx.type_ptr_to_ext(pointee, address_space)
            }
        }
    }

    fn scalar_pair_element_gcc_type<'gcc>(
        &self,
        cx: &CodegenCx<'gcc, 'tcx>,
        index: usize,
    ) -> Type<'gcc> {
        // This must produce the same result for `repr(transparent)` wrappers as for the inner type!
        // In other words, this should generally not look at the type at all, but only at the
        // layout.
        let (a, b) = match self.backend_repr {
            BackendRepr::ScalarPair(ref a, ref b) => (a, b),
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

        let offset = if index == 0 { Size::ZERO } else { a.size(cx).align_to(b.align(cx).abi) };
        self.scalar_gcc_type_at(cx, scalar, offset)
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

impl<'gcc, 'tcx> LayoutTypeCodegenMethods<'tcx> for CodegenCx<'gcc, 'tcx> {
    fn backend_type(&self, layout: TyAndLayout<'tcx>) -> Type<'gcc> {
        layout.gcc_type(self)
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

    fn scalar_pair_element_backend_type(
        &self,
        layout: TyAndLayout<'tcx>,
        index: usize,
        _immediate: bool,
    ) -> Type<'gcc> {
        layout.scalar_pair_element_gcc_type(self, index)
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

    fn fn_decl_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> Type<'gcc> {
        // FIXME(antoyo): Should we do something with `FnAbiGcc::fn_attributes`?
        let FnAbiGcc { return_type, arguments_type, is_c_variadic, .. } = fn_abi.gcc_type(self);
        self.context.new_function_pointer_type(None, return_type, &arguments_type, is_c_variadic)
    }
}
