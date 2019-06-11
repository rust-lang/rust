use super::misc::MiscMethods;
use super::Backend;
use super::HasCodegen;
use crate::common::TypeKind;
use crate::mir::place::PlaceRef;
use rustc::ty::{self, Ty};
use rustc::ty::layout::{self, TyLayout};
use rustc_target::abi::call::{ArgType, CastTarget, FnType, Reg};
use syntax_pos::DUMMY_SP;

// This depends on `Backend` and not `BackendTypes`, because consumers will probably want to use
// `LayoutOf` or `HasTyCtxt`. This way, they don't have to add a constraint on it themselves.
pub trait BaseTypeMethods<'tcx>: Backend<'tcx> {
    fn type_i1(&self) -> Self::Type;
    fn type_i8(&self) -> Self::Type;
    fn type_i16(&self) -> Self::Type;
    fn type_i32(&self) -> Self::Type;
    fn type_i64(&self) -> Self::Type;
    fn type_i128(&self) -> Self::Type;
    fn type_isize(&self) -> Self::Type;

    fn type_f32(&self) -> Self::Type;
    fn type_f64(&self) -> Self::Type;

    fn type_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type;
    fn type_struct(&self, els: &[Self::Type], packed: bool) -> Self::Type;
    fn type_kind(&self, ty: Self::Type) -> TypeKind;
    fn type_ptr_to(&self, ty: Self::Type) -> Self::Type;
    fn element_type(&self, ty: Self::Type) -> Self::Type;

    /// Returns the number of elements in `self` if it is a LLVM vector type.
    fn vector_length(&self, ty: Self::Type) -> usize;

    fn float_width(&self, ty: Self::Type) -> usize;

    /// Retrieves the bit width of the integer type `self`.
    fn int_width(&self, ty: Self::Type) -> u64;

    fn val_ty(&self, v: Self::Value) -> Self::Type;
}

pub trait DerivedTypeMethods<'tcx>: BaseTypeMethods<'tcx> + MiscMethods<'tcx> {
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

    fn type_needs_drop(&self, ty: Ty<'tcx>) -> bool {
        ty.needs_drop(self.tcx(), ty::ParamEnv::reveal_all())
    }

    fn type_is_sized(&self, ty: Ty<'tcx>) -> bool {
        ty.is_sized(self.tcx().at(DUMMY_SP), ty::ParamEnv::reveal_all())
    }

    fn type_is_freeze(&self, ty: Ty<'tcx>) -> bool {
        ty.is_freeze(self.tcx(), ty::ParamEnv::reveal_all(), DUMMY_SP)
    }

    fn type_has_metadata(&self, ty: Ty<'tcx>) -> bool {
        if ty.is_sized(self.tcx().at(DUMMY_SP), ty::ParamEnv::reveal_all()) {
            return false;
        }

        let tail = self.tcx().struct_tail(ty);
        match tail.sty {
            ty::Foreign(..) => false,
            ty::Str | ty::Slice(..) | ty::Dynamic(..) => true,
            _ => bug!("unexpected unsized tail: {:?}", tail),
        }
    }
}

impl<T> DerivedTypeMethods<'tcx> for T where Self: BaseTypeMethods<'tcx> + MiscMethods<'tcx> {}

pub trait LayoutTypeMethods<'tcx>: Backend<'tcx> {
    fn backend_type(&self, layout: TyLayout<'tcx>) -> Self::Type;
    fn cast_backend_type(&self, ty: &CastTarget) -> Self::Type;
    fn fn_ptr_backend_type(&self, ty: &FnType<'tcx, Ty<'tcx>>) -> Self::Type;
    fn reg_backend_type(&self, ty: &Reg) -> Self::Type;
    fn immediate_backend_type(&self, layout: TyLayout<'tcx>) -> Self::Type;
    fn is_backend_immediate(&self, layout: TyLayout<'tcx>) -> bool;
    fn is_backend_scalar_pair(&self, layout: TyLayout<'tcx>) -> bool;
    fn backend_field_index(&self, layout: TyLayout<'tcx>, index: usize) -> u64;
    fn scalar_pair_element_backend_type(
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
