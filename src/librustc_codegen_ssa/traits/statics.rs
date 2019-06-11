use super::BackendTypes;
use syntax_pos::symbol::LocalInternedString;
use rustc::hir::def_id::DefId;
use rustc::ty::layout::Align;

pub trait StaticMethods: BackendTypes {
    fn static_addr_of(&self, cv: Self::Value, align: Align, kind: Option<&str>) -> Self::Value;
    fn codegen_static(&self, def_id: DefId, is_mutable: bool);
}

pub trait StaticBuilderMethods: BackendTypes {
    fn get_static(&mut self, def_id: DefId) -> Self::Value;
    fn static_panic_msg(
        &mut self,
        msg: Option<LocalInternedString>,
        filename: LocalInternedString,
        line: Self::Value,
        col: Self::Value,
        kind: &str,
    ) -> Self::Value;
}
