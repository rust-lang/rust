use super::misc::MiscMethods;
use super::Backend;
use super::HasCodegen;
use crate::common::{self, TypeKind};
use crate::mir::place::PlaceRef;
use rustc::ty::layout::{self, Align, Size, TyLayout};
use rustc::ty::{self, Ty};
use rustc::util::nodemap::FxHashMap;
use rustc_target::abi::call::{ArgType, CastTarget, FnType, Reg};
use std::cell::RefCell;
use syntax::ast;

// This depends on `Backend` and not `BackendTypes`, because consumers will probably want to use
// `LayoutOf` or `HasTyCtxt`. This way, they don't have to add a constraint on it themselves.
pub trait BaseTypeMethods<'tcx>: Backend<'tcx> {
    fn type_void(&self) -> Self::Type;
    fn type_metadata(&self) -> Self::Type;
    fn type_i1(&self) -> Self::Type;
    fn type_i8(&self) -> Self::Type;
    fn type_i16(&self) -> Self::Type;
    fn type_i32(&self) -> Self::Type;
    fn type_i64(&self) -> Self::Type;
    fn type_i128(&self) -> Self::Type;

    // Creates an integer type with the given number of bits, e.g., i24
    fn type_ix(&self, num_bits: u64) -> Self::Type;
    fn type_isize(&self) -> Self::Type;

    fn type_f32(&self) -> Self::Type;
    fn type_f64(&self) -> Self::Type;
    fn type_x86_mmx(&self) -> Self::Type;

    fn type_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type;
    fn type_variadic_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type;
    fn type_struct(&self, els: &[Self::Type], packed: bool) -> Self::Type;
    fn type_array(&self, ty: Self::Type, len: u64) -> Self::Type;
    fn type_vector(&self, ty: Self::Type, len: u64) -> Self::Type;
    fn type_kind(&self, ty: Self::Type) -> TypeKind;
    fn type_ptr_to(&self, ty: Self::Type) -> Self::Type;
    fn element_type(&self, ty: Self::Type) -> Self::Type;

    /// Return the number of elements in `self` if it is a LLVM vector type.
    fn vector_length(&self, ty: Self::Type) -> usize;

    fn func_params_types(&self, ty: Self::Type) -> Vec<Self::Type>;
    fn float_width(&self, ty: Self::Type) -> usize;

    /// Retrieve the bit width of the integer type `self`.
    fn int_width(&self, ty: Self::Type) -> u64;

    fn val_ty(&self, v: Self::Value) -> Self::Type;
    fn scalar_lltypes(&self) -> &RefCell<FxHashMap<Ty<'tcx>, Self::Type>>;
}

pub trait DerivedTypeMethods<'tcx>: BaseTypeMethods<'tcx> + MiscMethods<'tcx> {
    fn type_bool(&self) -> Self::Type {
        self.type_i8()
    }

    fn type_i8p(&self) -> Self::Type {
        self.type_ptr_to(self.type_i8())
    }

    fn type_int(&self) -> Self::Type {
        match &self.sess().target.target.target_c_int_width[..] {
            "16" => self.type_i16(),
            "32" => self.type_i32(),
            "64" => self.type_i64(),
            width => bug!("Unsupported target_c_int_width: {}", width),
        }
    }

    fn type_int_from_ty(&self, t: ast::IntTy) -> Self::Type {
        match t {
            ast::IntTy::Isize => self.type_isize(),
            ast::IntTy::I8 => self.type_i8(),
            ast::IntTy::I16 => self.type_i16(),
            ast::IntTy::I32 => self.type_i32(),
            ast::IntTy::I64 => self.type_i64(),
            ast::IntTy::I128 => self.type_i128(),
        }
    }

    fn type_uint_from_ty(&self, t: ast::UintTy) -> Self::Type {
        match t {
            ast::UintTy::Usize => self.type_isize(),
            ast::UintTy::U8 => self.type_i8(),
            ast::UintTy::U16 => self.type_i16(),
            ast::UintTy::U32 => self.type_i32(),
            ast::UintTy::U64 => self.type_i64(),
            ast::UintTy::U128 => self.type_i128(),
        }
    }

    fn type_float_from_ty(&self, t: ast::FloatTy) -> Self::Type {
        match t {
            ast::FloatTy::F32 => self.type_f32(),
            ast::FloatTy::F64 => self.type_f64(),
        }
    }

    fn type_from_integer(&self, i: layout::Integer) -> Self::Type {
        use rustc::ty::layout::Integer::*;
        match i {
            I8 => self.type_i8(),
            I16 => self.type_i16(),
            I32 => self.type_i32(),
            I64 => self.type_i64(),
            I128 => self.type_i128(),
        }
    }

    fn type_pointee_for_align(&self, align: Align) -> Self::Type {
        // FIXME(eddyb) We could find a better approximation if ity.align < align.
        let ity = layout::Integer::approximate_align(self, align);
        self.type_from_integer(ity)
    }

    /// Return a LLVM type that has at most the required alignment,
    /// and exactly the required size, as a best-effort padding array.
    fn type_padding_filler(&self, size: Size, align: Align) -> Self::Type {
        let unit = layout::Integer::approximate_align(self, align);
        let size = size.bytes();
        let unit_size = unit.size().bytes();
        assert_eq!(size % unit_size, 0);
        self.type_array(self.type_from_integer(unit), size / unit_size)
    }

    fn type_needs_drop(&self, ty: Ty<'tcx>) -> bool {
        common::type_needs_drop(self.tcx(), ty)
    }

    fn type_is_sized(&self, ty: Ty<'tcx>) -> bool {
        common::type_is_sized(self.tcx(), ty)
    }

    fn type_is_freeze(&self, ty: Ty<'tcx>) -> bool {
        common::type_is_freeze(self.tcx(), ty)
    }

    fn type_has_metadata(&self, ty: Ty<'tcx>) -> bool {
        use syntax_pos::DUMMY_SP;
        if ty.is_sized(self.tcx().at(DUMMY_SP), ty::ParamEnv::reveal_all()) {
            return false;
        }

        let tail = self.tcx().struct_tail(ty);
        match tail.sty {
            ty::Foreign(..) => false,
            ty::Str | ty::Slice(..) | ty::Dynamic(..) => true,
            _ => bug!("unexpected unsized tail: {:?}", tail.sty),
        }
    }
}

impl<T> DerivedTypeMethods<'tcx> for T where Self: BaseTypeMethods<'tcx> + MiscMethods<'tcx> {}

pub trait LayoutTypeMethods<'tcx>: Backend<'tcx> {
    fn backend_type(&self, layout: TyLayout<'tcx>) -> Self::Type;
    fn cast_backend_type(&self, ty: &CastTarget) -> Self::Type;
    fn fn_backend_type(&self, ty: &FnType<'tcx, Ty<'tcx>>) -> Self::Type;
    fn fn_ptr_backend_type(&self, ty: &FnType<'tcx, Ty<'tcx>>) -> Self::Type;
    fn reg_backend_type(&self, ty: &Reg) -> Self::Type;
    fn immediate_backend_type(&self, layout: TyLayout<'tcx>) -> Self::Type;
    fn is_backend_immediate(&self, layout: TyLayout<'tcx>) -> bool;
    fn is_backend_scalar_pair(&self, layout: TyLayout<'tcx>) -> bool;
    fn backend_field_index(&self, layout: TyLayout<'tcx>, index: usize) -> u64;
    fn scalar_pair_element_backend_type<'a>(
        &self,
        layout: TyLayout<'tcx>,
        index: usize,
        immediate: bool,
    ) -> Self::Type;
}

pub trait ArgTypeMethods<'tcx>: HasCodegen<'tcx> {
    fn store_fn_arg(
        &mut self,
        ty: &ArgType<'tcx, Ty<'tcx>>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, Self::Value>,
    );
    fn store_arg_ty(
        &mut self,
        ty: &ArgType<'tcx, Ty<'tcx>>,
        val: Self::Value,
        dst: PlaceRef<'tcx, Self::Value>,
    );
    fn memory_ty(&self, ty: &ArgType<'tcx, Ty<'tcx>>) -> Self::Type;
}

pub trait TypeMethods<'tcx>: DerivedTypeMethods<'tcx> + LayoutTypeMethods<'tcx> {}

impl<T> TypeMethods<'tcx> for T where Self: DerivedTypeMethods<'tcx> + LayoutTypeMethods<'tcx> {}
