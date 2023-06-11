use super::misc::MiscMethods;
use super::Backend;
use super::HasCodegen;
use crate::common::TypeKind;
use crate::mir::place::PlaceRef;
use rustc_middle::ty::layout::TyAndLayout;
use rustc_middle::ty::{self, Ty};
use rustc_target::abi::call::{ArgAbi, CastTarget, FnAbi, Reg};
use rustc_target::abi::{AddressSpace, Integer};

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

    fn type_array(&self, ty: Self::Type, len: u64) -> Self::Type;
    fn type_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type;
    fn type_struct(&self, els: &[Self::Type], packed: bool) -> Self::Type;
    fn type_kind(&self, ty: Self::Type) -> TypeKind;
    fn type_ptr_to(&self, ty: Self::Type) -> Self::Type;
    fn type_ptr_to_ext(&self, ty: Self::Type, address_space: AddressSpace) -> Self::Type;
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
        self.type_i8p_ext(AddressSpace::DATA)
    }

    fn type_i8p_ext(&self, address_space: AddressSpace) -> Self::Type {
        self.type_ptr_to_ext(self.type_i8(), address_space)
    }

    fn type_int(&self) -> Self::Type {
        match &self.sess().target.c_int_width[..] {
            "16" => self.type_i16(),
            "32" => self.type_i32(),
            "64" => self.type_i64(),
            width => bug!("Unsupported c_int_width: {}", width),
        }
    }

    fn type_from_integer(&self, i: Integer) -> Self::Type {
        use Integer::*;
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
        ty.is_sized(self.tcx(), ty::ParamEnv::reveal_all())
    }

    fn type_is_freeze(&self, ty: Ty<'tcx>) -> bool {
        ty.is_freeze(self.tcx(), ty::ParamEnv::reveal_all())
    }

    fn type_has_metadata(&self, ty: Ty<'tcx>) -> bool {
        let param_env = ty::ParamEnv::reveal_all();
        if ty.is_sized(self.tcx(), param_env) {
            return false;
        }

        let tail = self.tcx().struct_tail_erasing_lifetimes(ty, param_env);
        match tail.kind() {
            ty::Foreign(..) => false,
            ty::Str | ty::Slice(..) | ty::Dynamic(..) => true,
            _ => bug!("unexpected unsized tail: {:?}", tail),
        }
    }
}

impl<'tcx, T> DerivedTypeMethods<'tcx> for T where Self: BaseTypeMethods<'tcx> + MiscMethods<'tcx> {}

pub trait LayoutTypeMethods<'tcx>: Backend<'tcx> {
    /// The backend type used for a rust type when it's in memory,
    /// such as when it's stack-allocated or when it's being loaded or stored.
    fn backend_type(&self, layout: TyAndLayout<'tcx>) -> Self::Type;
    fn cast_backend_type(&self, ty: &CastTarget) -> Self::Type;
    fn fn_decl_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> Self::Type;
    fn fn_ptr_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> Self::Type;
    fn reg_backend_type(&self, ty: &Reg) -> Self::Type;
    /// The backend type used for a rust type when it's in an SSA register.
    ///
    /// For nearly all types this is the same as the [`Self::backend_type`], however
    /// `bool` (and other `0`-or-`1` values) are kept as [`BaseTypeMethods::type_i1`]
    /// in registers but as [`BaseTypeMethods::type_i8`] in memory.
    ///
    /// Converting values between the two different backend types is done using
    /// [`from_immediate`](super::BuilderMethods::from_immediate) and
    /// [`to_immediate_scalar`](super::BuilderMethods::to_immediate_scalar).
    fn immediate_backend_type(&self, layout: TyAndLayout<'tcx>) -> Self::Type;
    fn is_backend_immediate(&self, layout: TyAndLayout<'tcx>) -> bool;
    fn is_backend_scalar_pair(&self, layout: TyAndLayout<'tcx>) -> bool;
    fn backend_field_index(&self, layout: TyAndLayout<'tcx>, index: usize) -> u64;
    fn scalar_pair_element_backend_type(
        &self,
        layout: TyAndLayout<'tcx>,
        index: usize,
        immediate: bool,
    ) -> Self::Type;

    /// A type that can be used in a [`super::BuilderMethods::load`] +
    /// [`super::BuilderMethods::store`] pair to implement a *typed* copy,
    /// such as a MIR `*_0 = *_1`.
    ///
    /// It's always legal to return `None` here, as the provided impl does,
    /// in which case callers should use [`super::BuilderMethods::memcpy`]
    /// instead of the `load`+`store` pair.
    ///
    /// This can be helpful for things like arrays, where the LLVM backend type
    /// `[3 x i16]` optimizes to three separate loads and stores, but it can
    /// instead be copied via an `i48` that stays as the single `load`+`store`.
    /// (As of 2023-05 LLVM cannot necessarily optimize away a `memcpy` in these
    /// cases, due to `poison` handling, but in codegen we have more information
    /// about the type invariants, so can emit something better instead.)
    ///
    /// This *should* return `None` for particularly-large types, where leaving
    /// the `memcpy` may well be important to avoid code size explosion.
    fn scalar_copy_backend_type(&self, layout: TyAndLayout<'tcx>) -> Option<Self::Type> {
        let _ = layout;
        None
    }
}

// For backends that support CFI using type membership (i.e., testing whether a given pointer is
// associated with a type identifier).
pub trait TypeMembershipMethods<'tcx>: Backend<'tcx> {
    fn add_type_metadata(&self, _function: Self::Function, _typeid: String) {}
    fn set_type_metadata(&self, _function: Self::Function, _typeid: String) {}
    fn typeid_metadata(&self, _typeid: String) -> Option<Self::Value> {
        None
    }
    fn add_kcfi_type_metadata(&self, _function: Self::Function, _typeid: u32) {}
    fn set_kcfi_type_metadata(&self, _function: Self::Function, _typeid: u32) {}
}

pub trait ArgAbiMethods<'tcx>: HasCodegen<'tcx> {
    fn store_fn_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, Self::Value>,
    );
    fn store_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        val: Self::Value,
        dst: PlaceRef<'tcx, Self::Value>,
    );
    fn arg_memory_ty(&self, arg_abi: &ArgAbi<'tcx, Ty<'tcx>>) -> Self::Type;
}

pub trait TypeMethods<'tcx>:
    DerivedTypeMethods<'tcx> + LayoutTypeMethods<'tcx> + TypeMembershipMethods<'tcx>
{
}

impl<'tcx, T> TypeMethods<'tcx> for T where
    Self: DerivedTypeMethods<'tcx> + LayoutTypeMethods<'tcx> + TypeMembershipMethods<'tcx>
{
}
