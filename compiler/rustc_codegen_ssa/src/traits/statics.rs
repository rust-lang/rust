use rustc_abi::Size;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir::def_id::DefId;
use rustc_middle::mir::interpret::ConstAllocation;

use super::BackendTypes;

pub trait StaticCodegenMethods: BackendTypes {
    fn static_addr_of(
        &self,
        alloc: ConstAllocation<'_>,
        kind: Option<&str>,
        ptrauth_discriminators: Option<&FxHashMap<Size, u64>>,
    ) -> Self::Value;

    fn codegen_static(&mut self, def_id: DefId);
}

pub trait StaticBuilderMethods: BackendTypes {
    fn get_static(&mut self, def_id: DefId) -> Self::Value;
}
