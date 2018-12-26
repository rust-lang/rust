use super::BackendTypes;
use rustc::hir::def_id::DefId;
use rustc::ty::layout::Align;

pub trait StaticMethods: BackendTypes {
    fn static_addr_of(&self, cv: Self::Value, align: Align, kind: Option<&str>) -> Self::Value;
    fn codegen_static(&self, def_id: DefId, is_mutable: bool);
}

pub trait StaticBuilderMethods<'tcx>: BackendTypes {
    fn get_static(&self, def_id: DefId) -> Self::Value;
}
