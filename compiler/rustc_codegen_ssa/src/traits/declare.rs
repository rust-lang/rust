use rustc_hir::def_id::DefId;
use rustc_middle::mir::mono::{Linkage, Visibility};
use rustc_middle::ty::Instance;

pub trait PreDefineCodegenMethods<'tcx> {
    fn predefine_static(
        &mut self,
        def_id: DefId,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    );
    fn predefine_fn(
        &mut self,
        instance: Instance<'tcx>,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    );
}
