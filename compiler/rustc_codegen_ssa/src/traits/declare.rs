use rustc_hir::def_id::DefId;
use rustc_middle::mir::mono::{Linkage, Visibility};
use rustc_middle::ty::Instance;

use crate::traits::BackendTypes;

pub trait PreDefineCodegenMethods<'tcx>: BackendTypes {
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

    fn weak_alias(&self, aliasee: Self::Function, aliasee_name: &str, name: &str);
}
