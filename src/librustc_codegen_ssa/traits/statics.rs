use super::BackendTypes;
use rustc::ty::layout::Align;
use rustc_hir::def_id::DefId;

pub trait StaticMethods: BackendTypes {
    fn static_addr_of(&self, cv: Self::Value, align: Align, kind: Option<&str>) -> Self::Value;
    fn codegen_static(&self, def_id: DefId, is_mutable: bool);
}

pub trait StaticBuilderMethods: BackendTypes {
    fn get_static(&mut self, def_id: DefId) -> Self::Value;
}
