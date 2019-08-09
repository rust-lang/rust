use super::BackendTypes;
use rustc::hir::def_id::DefId;
use rustc::mir::mono::{Linkage, Visibility};
use rustc::ty::{self, Instance};

pub trait DeclareMethods<'tcx>: BackendTypes {
    /// Declare a global value.
    ///
    /// If there’s a value with the same name already declared, the function will
    /// return its Value instead.
    fn declare_global(&self, name: &str, ty: Self::Type) -> Self::Value;

    /// Declare a C ABI function.
    ///
    /// Only use this for foreign function ABIs and glue. For Rust functions use
    /// `declare_fn` instead.
    ///
    /// If there’s a value with the same name already declared, the function will
    /// update the declaration and return existing Value instead.
    fn declare_cfn(&self, name: &str, fn_type: Self::Type) -> Self::Value;

    /// Declare a Rust function.
    ///
    /// If there’s a value with the same name already declared, the function will
    /// update the declaration and return existing Value instead.
    fn declare_fn(&self, name: &str, sig: ty::PolyFnSig<'tcx>) -> Self::Value;

    /// Declare a global with an intention to define it.
    ///
    /// Use this function when you intend to define a global. This function will
    /// return `None` if the name already has a definition associated with it. In that
    /// case an error should be reported to the user, because it usually happens due
    /// to user’s fault (e.g., misuse of #[no_mangle] or #[export_name] attributes).
    fn define_global(&self, name: &str, ty: Self::Type) -> Option<Self::Value>;

    /// Declare a private global
    ///
    /// Use this function when you intend to define a global without a name.
    fn define_private_global(&self, ty: Self::Type) -> Self::Value;

    /// Declare a Rust function with an intention to define it.
    ///
    /// Use this function when you intend to define a function. This function will
    /// return panic if the name already has a definition associated with it. This
    /// can happen with #[no_mangle] or #[export_name], for example.
    fn define_fn(&self, name: &str, fn_sig: ty::PolyFnSig<'tcx>) -> Self::Value;

    /// Declare a Rust function with an intention to define it.
    ///
    /// Use this function when you intend to define a function. This function will
    /// return panic if the name already has a definition associated with it. This
    /// can happen with #[no_mangle] or #[export_name], for example.
    fn define_internal_fn(&self, name: &str, fn_sig: ty::PolyFnSig<'tcx>) -> Self::Value;

    /// Gets declared value by name.
    fn get_declared_value(&self, name: &str) -> Option<Self::Value>;

    /// Gets defined or externally defined (AvailableExternally linkage) value by
    /// name.
    fn get_defined_value(&self, name: &str) -> Option<Self::Value>;
}

pub trait PreDefineMethods<'tcx>: BackendTypes {
    fn predefine_static(
        &self,
        def_id: DefId,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    );
    fn predefine_fn(
        &self,
        instance: Instance<'tcx>,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    );
}
