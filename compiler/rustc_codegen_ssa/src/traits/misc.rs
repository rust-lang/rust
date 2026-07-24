use std::cell::RefCell;

use rustc_data_structures::fx::FxHashMap;
use rustc_middle::ty::{self, Instance, Ty};
use rustc_session::{PointerAuthSchema, Session};
use rustc_span::Symbol;

use super::BackendTypes;

pub trait MiscCodegenMethods<'tcx>: BackendTypes {
    fn vtables(
        &self,
    ) -> &RefCell<FxHashMap<(Ty<'tcx>, Option<ty::ExistentialTraitRef<'tcx>>), Self::Value>>;
    fn apply_vcall_visibility_metadata(
        &self,
        _ty: Ty<'tcx>,
        _poly_trait_ref: Option<ty::ExistentialTraitRef<'tcx>>,
        _vtable: Self::Value,
    ) {
    }
    fn get_fn(&self, instance: Instance<'tcx>) -> Self::Function;
    fn get_fn_addr(
        &self,
        instance: Instance<'tcx>,
        ptrauth_schema: Option<PointerAuthSchema>,
    ) -> Self::Value;
    fn eh_personality(&self) -> Self::Function;
    fn sess(&self) -> &Session;
    fn set_frame_pointer_type(&self, llfn: Self::Function);
    fn apply_target_cpu_attr(&self, llfn: Self::Function);
    /// Declares the extern "C" main function for the entry point. Returns None if the symbol
    /// already exists.
    fn declare_c_main(&self, fn_type: Self::FunctionSignature) -> Option<Self::Function>;

    /// Whether `codegen_intrinsic_call` expects to always have a `place_value`
    /// when emitting code for the intrinsic `name`.
    ///
    /// This is discouraged, but here for now to simplify migration to using OperandValues
    fn intrinsic_call_expects_place_always(&self, name: Symbol) -> bool;
}
