use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefId;
use rustc_macros::extension;
use rustc_middle::query::IntoQueryParam;
use rustc_middle::ty;
use rustc_middle::ty::TyCtxt;
use rustc_span::Symbol;

use crate::creader::CStore;

#[extension(pub trait TyCtxtMetadataExt<'tcx>)]
impl<'tcx> TyCtxt<'tcx> {
    #[inline]
    fn intrinsic_extern(self, def_id: DefId) -> Option<ty::IntrinsicDef> {
        if self.dep_graph.is_fully_enabled() {
            self.intrinsic_raw_q(def_id)
        } else {
            assert!(!def_id.is_local());
            let cdata =
                rustc_data_structures::sync::FreezeReadGuard::map(CStore::from_tcx(self), |c| {
                    c.get_crate_data(def_id.krate).cdata
                });
            let cdata =
                crate::creader::CrateMetadataRef { cdata: &cdata, cstore: &CStore::from_tcx(self) };
            cdata.get_intrinsic(def_id.index)
        }
    }

    #[inline]
    fn is_intrinsic(self, def_id: DefId, name: Symbol) -> bool {
        let Some(i) = self.intrinsic(def_id) else { return false };
        i.name == name
    }

    #[inline]
    fn intrinsic_raw(self, def_id: DefId) -> Option<ty::IntrinsicDef> {
        if let Some(local_def_id) = def_id.as_local() {
            self.intrinsic_local(local_def_id)
        } else {
            self.intrinsic_extern(def_id)
        }
    }

    #[inline]
    fn intrinsic(self, def_id: impl IntoQueryParam<DefId> + Copy) -> Option<ty::IntrinsicDef> {
        match self.def_kind(def_id) {
            DefKind::Fn | DefKind::AssocFn => {}
            _ => return None,
        }
        self.intrinsic_raw(def_id.into_query_param())
    }
}
