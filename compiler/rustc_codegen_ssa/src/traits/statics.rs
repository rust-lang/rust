use rustc_hir::def_id::DefId;
use rustc_middle::mir::interpret::ConstAllocation;

use super::BackendTypes;

pub trait StaticCodegenMethods: BackendTypes {
    fn static_addr_of(&self, alloc: ConstAllocation<'_>, kind: Option<&str>) -> Self::Value;
    /// Set a debuginfo name. The name closure is only invoked if a name actually needs to be
    /// registered, so you can do expensive name calculations in it.
    fn set_value_name(&self, val: Self::Value, gen_name: impl FnOnce() -> String);
    fn codegen_static(&mut self, def_id: DefId);

    /// Prefer calling [StaticBuilderMethods::get_static] as that also performs
    /// addrspace casts and runtime lookups for thread local statics.
    fn get_static(&self, def_id: DefId) -> Self::Value;
}

pub trait StaticBuilderMethods: BackendTypes {
    fn get_static(&mut self, def_id: DefId) -> Self::Value;
}
