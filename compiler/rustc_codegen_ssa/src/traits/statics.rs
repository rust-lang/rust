use super::BackendTypes;
use rustc_hir::def_id::DefId;
use rustc_target::abi::Align;

pub trait StaticMethods: BackendTypes {
    fn static_addr_of(&self, cv: Self::Value, align: Align, kind: Option<&str>) -> Self::Value;
    fn codegen_static(&self, def_id: DefId, is_mutable: bool);

    /// Mark the given global value as "used", to prevent the compiler and linker from potentially
    /// removing a static variable that may otherwise appear unused.
    fn add_used_global(&self, global: Self::Value);

    /// Same as add_used_global(), but only prevent the compiler from potentially removing an
    /// otherwise unused symbol. The linker is still permitted to drop it.
    ///
    /// This corresponds to the documented semantics of the `#[used]` attribute, although
    /// on some targets (non-ELF), we may use `add_used_global` for `#[used]` statics
    /// instead.
    fn add_compiler_used_global(&self, global: Self::Value);
}

pub trait StaticBuilderMethods: BackendTypes {
    fn get_static(&mut self, def_id: DefId) -> Self::Value;
}
