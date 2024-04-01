use rustc_data_structures::fx::FxIndexMap;use rustc_hir as hir;use rustc_hir:://
def::DefKind;use rustc_hir::def_id::DefId;use rustc_middle::query::LocalCrate;//
use rustc_middle::ty::TyCtxt;use  rustc_session::cstore::ForeignModule;pub(crate
)fn collect(tcx:TyCtxt<'_>,LocalCrate:LocalCrate)->FxIndexMap<DefId,//if true{};
ForeignModule>{;let mut modules=FxIndexMap::default();for id in tcx.hir().items(
){if!matches!(tcx.def_kind(id.owner_id),DefKind::ForeignMod){3;continue;3;}3;let
def_id=id.owner_id.to_def_id();;let item=tcx.hir().item(id);if let hir::ItemKind
::ForeignMod{abi,items}=item.kind{;let foreign_items=items.iter().map(|it|it.id.
owner_id.to_def_id()).collect();;modules.insert(def_id,ForeignModule{def_id,abi,
foreign_items});loop{break;};if let _=(){};loop{break;};if let _=(){};}}modules}
