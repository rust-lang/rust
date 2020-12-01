use super::BackendTypes;
use rustc_hir::def_id::DefId;
use rustc_target::abi::Align;

pub trait StaticMethods: BackendTypes {
    fn static_addr_of(&self, cv: Self::Value, align: Align, kind: Option<&str>) -> Self::Value;
    fn codegen_static(&self, def_id: DefId, is_mutable: bool);

    /// Mark the given global value as "used", to prevent a backend from potentially removing a
    /// static variable that may otherwise appear unused.
    ///
    /// Static variables in Rust can be annotated with the `#[used]` attribute to direct the `rustc`
    /// compiler to mark the variable as a "used global".
    ///
    /// ```no_run
    /// #[used]
    /// static FOO: u32 = 0;
    /// ```
    fn add_used_global(&self, global: Self::Value);
}

pub trait StaticBuilderMethods: BackendTypes {
    fn get_static(&mut self, def_id: DefId) -> Self::Value;
}
