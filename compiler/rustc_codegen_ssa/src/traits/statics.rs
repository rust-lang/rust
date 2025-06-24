use rustc_abi::Align;
use rustc_hir::def_id::DefId;

use super::BackendTypes;

pub trait StaticCodegenMethods: BackendTypes {
    fn static_addr_of(&self, cv: Self::Value, align: Align, kind: Option<&str>) -> Self::Value;
    fn get_value_name(&self, val: Self::Value) -> &[u8];
    fn set_value_name(&self, val: Self::Value, name: &[u8]);
    fn codegen_static(&mut self, def_id: DefId);
    fn get_static(&self, def_id: DefId) -> Self::Value;
}

pub trait StaticBuilderMethods: BackendTypes {
    fn get_static(&mut self, def_id: DefId) -> Self::Value;
}
