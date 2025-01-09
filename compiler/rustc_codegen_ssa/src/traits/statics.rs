use rustc_abi::Align;
use rustc_hir::def_id::DefId;

use super::BackendTypes;

pub trait StaticCodegenMethods: BackendTypes {
    fn static_addr_of(&self, cv: Self::Value, align: Align, kind: Option<&str>) -> Self::Value;
    fn codegen_static(&mut self, def_id: DefId);
}

pub trait StaticBuilderMethods: BackendTypes {
    fn get_static(&mut self, def_id: DefId) -> Self::Value;
}
