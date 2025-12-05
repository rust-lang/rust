use std::iter;
use std::ops::ControlFlow;

use rustc_abi::ExternAbi;
use rustc_data_structures::fx::{FxIndexMap, FxIndexSet};
use rustc_hir as hir;
use rustc_hir::attrs::AttributeKind;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::find_attr;
use rustc_hir::intravisit::{self, Visitor};
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::privacy::{EffectiveVisibility, Level};
use rustc_middle::query::{LocalCrate, Providers};
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitor, Visibility,
};
use rustc_session::config::CrateType;
use rustc_span::Span;

use crate::errors::UnexportableItem;

struct ExportableItemCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    exportable_items: FxIndexSet<DefId>,
    in_exportable_mod: bool,
    seen_exportable_in_mod: bool,
}

impl<'tcx> ExportableItemCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> ExportableItemCollector<'tcx> {
        ExportableItemCollector {
            tcx,
            exportable_items: Default::default(),
            in_exportable_mod: false,
            seen_exportable_in_mod: false,
        }
    }

    fn report_wrong_site(&self, def_id: LocalDefId) {
        let def_descr = self.tcx.def_descr(def_id.to_def_id());
        self.tcx.dcx().emit_err(UnexportableItem::Item {
            descr: &format!("{}", def_descr),
            span: self.tcx.def_span(def_id),
        });
    }

    fn item_is_exportable(&self, def_id: LocalDefId) -> bool {
        let has_attr = find_attr!(self.tcx.get_all_attrs(def_id), AttributeKind::ExportStable);
        if !self.in_exportable_mod && !has_attr {
            return false;
        }

        let visibilities = self.tcx.effective_visibilities(());
        let is_pub = visibilities.is_directly_public(def_id);

        if has_attr && !is_pub {
            let vis = visibilities.effective_vis(def_id).cloned().unwrap_or_else(|| {
                EffectiveVisibility::from_vis(Visibility::Restricted(
                    self.tcx.parent_module_from_def_id(def_id).to_local_def_id(),
                ))
            });
            let vis = vis.at_level(Level::Direct);
            let span = self.tcx.def_span(def_id);

            self.tcx.dcx().emit_err(UnexportableItem::PrivItem {
                vis_note: span,
                vis_descr: &vis.to_string(def_id, self.tcx),
                span,
            });
            return false;
        }

        is_pub && (has_attr || self.in_exportable_mod)
    }

    fn add_exportable(&mut self, def_id: LocalDefId) {
        self.seen_exportable_in_mod = true;
        self.exportable_items.insert(def_id.to_def_id());
    }

    fn walk_item_with_mod(&mut self, item: &'tcx hir::Item<'tcx>) {
        let def_id = item.hir_id().owner.def_id;
        let old_exportable_mod = self.in_exportable_mod;
        if find_attr!(self.tcx.get_all_attrs(def_id), AttributeKind::ExportStable) {
            self.in_exportable_mod = true;
        }
        let old_seen_exportable_in_mod = std::mem::replace(&mut self.seen_exportable_in_mod, false);

        intravisit::walk_item(self, item);

        if self.seen_exportable_in_mod || self.in_exportable_mod {
            self.exportable_items.insert(def_id.to_def_id());
        }

        self.seen_exportable_in_mod = old_seen_exportable_in_mod;
        self.in_exportable_mod = old_exportable_mod;
    }
}

impl<'tcx> Visitor<'tcx> for ExportableItemCollector<'tcx> {
    type NestedFilter = nested_filter::All;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        let def_id = item.hir_id().owner.def_id;
        // Applying #[extern] attribute to modules is simply equivalent to
        // applying the attribute to every public item within it.
        match item.kind {
            hir::ItemKind::Mod(..) => {
                self.walk_item_with_mod(item);
                return;
            }
            hir::ItemKind::Impl(impl_) if impl_.of_trait.is_none() => {
                self.walk_item_with_mod(item);
                return;
            }
            _ => {}
        }

        if !self.item_is_exportable(def_id) {
            return;
        }

        match item.kind {
            hir::ItemKind::Fn { .. }
            | hir::ItemKind::Struct(..)
            | hir::ItemKind::Enum(..)
            | hir::ItemKind::Union(..)
            | hir::ItemKind::TyAlias(..) => {
                self.add_exportable(def_id);
            }
            hir::ItemKind::Use(path, _) => {
                for res in path.res.present_items() {
                    // Only local items are exportable.
                    if let Some(res_id) = res.opt_def_id()
                        && let Some(res_id) = res_id.as_local()
                    {
                        self.add_exportable(res_id);
                    }
                }
            }
            // handled above
            hir::ItemKind::Mod(..) => unreachable!(),
            hir::ItemKind::Impl(impl_) if impl_.of_trait.is_none() => {
                unreachable!();
            }
            _ => self.report_wrong_site(def_id),
        }
    }

    fn visit_impl_item(&mut self, item: &'tcx hir::ImplItem<'tcx>) {
        let def_id = item.hir_id().owner.def_id;
        if !self.item_is_exportable(def_id) {
            return;
        }
        match item.kind {
            hir::ImplItemKind::Fn(..) | hir::ImplItemKind::Type(..) => {
                self.add_exportable(def_id);
            }
            _ => self.report_wrong_site(def_id),
        }
    }

    fn visit_foreign_item(&mut self, item: &'tcx hir::ForeignItem<'tcx>) {
        let def_id = item.hir_id().owner.def_id;
        if !self.item_is_exportable(def_id) {
            self.report_wrong_site(def_id);
        }
    }

    fn visit_trait_item(&mut self, item: &'tcx hir::TraitItem<'tcx>) {
        let def_id = item.hir_id().owner.def_id;
        if !self.item_is_exportable(def_id) {
            self.report_wrong_site(def_id);
        }
    }
}

struct ExportableItemsChecker<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    exportable_items: &'a FxIndexSet<DefId>,
    item_id: DefId,
}

impl<'tcx, 'a> ExportableItemsChecker<'tcx, 'a> {
    fn check(&mut self) {
        match self.tcx.def_kind(self.item_id) {
            DefKind::Fn | DefKind::AssocFn => self.check_fn(),
            DefKind::Enum | DefKind::Struct | DefKind::Union => self.check_ty(),
            _ => {}
        }
    }

    fn check_fn(&mut self) {
        let def_id = self.item_id.expect_local();
        let span = self.tcx.def_span(def_id);

        if self.tcx.generics_of(def_id).requires_monomorphization(self.tcx) {
            self.tcx.dcx().emit_err(UnexportableItem::GenericFn(span));
            return;
        }

        let sig = self.tcx.fn_sig(def_id).instantiate_identity().skip_binder();
        if !matches!(sig.abi, ExternAbi::C { .. }) {
            self.tcx.dcx().emit_err(UnexportableItem::FnAbi(span));
            return;
        }

        let sig = self
            .tcx
            .try_normalize_erasing_regions(ty::TypingEnv::non_body_analysis(self.tcx, def_id), sig)
            .unwrap_or(sig);

        let hir_id = self.tcx.local_def_id_to_hir_id(def_id);
        let decl = self.tcx.hir_fn_decl_by_hir_id(hir_id).unwrap();

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            self.check_nested_types_are_exportable(*input_ty, input_hir.span);
        }

        if let hir::FnRetTy::Return(ret_hir) = decl.output {
            self.check_nested_types_are_exportable(sig.output(), ret_hir.span);
        }
    }

    fn check_ty(&mut self) {
        let ty = self.tcx.type_of(self.item_id).skip_binder();
        if let ty::Adt(adt_def, _) = ty.kind() {
            if !adt_def.repr().inhibit_struct_field_reordering() {
                self.tcx
                    .dcx()
                    .emit_err(UnexportableItem::TypeRepr(self.tcx.def_span(self.item_id)));
            }

            // FIXME: support `#[export(unsafe_stable_abi = "hash")]` syntax
            for variant in adt_def.variants() {
                for field in &variant.fields {
                    if !field.vis.is_public() {
                        self.tcx.dcx().emit_err(UnexportableItem::AdtWithPrivFields {
                            span: self.tcx.def_span(self.item_id),
                            vis_note: self.tcx.def_span(field.did),
                            field_name: field.name.as_str(),
                        });
                    }
                }
            }
        }
    }

    fn check_nested_types_are_exportable(&mut self, ty: Ty<'tcx>, ty_span: Span) {
        let res = ty.visit_with(self);
        if let Some(err_cause) = res.break_value() {
            self.tcx.dcx().emit_err(UnexportableItem::TypeInInterface {
                span: self.tcx.def_span(self.item_id),
                desc: self.tcx.def_descr(self.item_id),
                ty: &format!("{}", err_cause),
                ty_span,
            });
        }
    }
}

impl<'tcx, 'a> TypeVisitor<TyCtxt<'tcx>> for ExportableItemsChecker<'tcx, 'a> {
    type Result = ControlFlow<Ty<'tcx>>;

    fn visit_ty(&mut self, ty: Ty<'tcx>) -> Self::Result {
        match ty.kind() {
            ty::Adt(adt_def, _) => {
                let did = adt_def.did();
                let exportable = if did.is_local() {
                    self.exportable_items.contains(&did)
                } else {
                    self.tcx.is_exportable(did)
                };
                if !exportable {
                    return ControlFlow::Break(ty);
                }
                for variant in adt_def.variants() {
                    for field in &variant.fields {
                        let field_ty = self.tcx.type_of(field.did).instantiate_identity();
                        field_ty.visit_with(self)?;
                    }
                }

                return ty.super_visit_with(self);
            }

            ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Bool | ty::Char | ty::Error(_) => {}

            ty::Array(_, _)
            | ty::Ref(_, _, _)
            | ty::Param(_)
            | ty::Closure(_, _)
            | ty::Dynamic(_, _)
            | ty::Coroutine(_, _)
            | ty::Foreign(_)
            | ty::Str
            | ty::Tuple(_)
            | ty::Pat(..)
            | ty::Slice(_)
            | ty::RawPtr(_, _)
            | ty::FnDef(_, _)
            | ty::FnPtr(_, _)
            | ty::CoroutineClosure(_, _)
            | ty::CoroutineWitness(_, _)
            | ty::Never
            | ty::UnsafeBinder(_)
            | ty::Alias(ty::AliasTyKind::Opaque, _) => {
                return ControlFlow::Break(ty);
            }

            ty::Alias(..) | ty::Infer(_) | ty::Placeholder(_) | ty::Bound(..) => unreachable!(),
        }
        ControlFlow::Continue(())
    }
}

/// Exportable items:
///
/// 1. Structs/enums/unions with a stable representation (e.g. repr(i32) or repr(C)).
/// 2. Primitive types.
/// 3. Non-generic functions with a stable ABI (e.g. extern "C") for which every user
///    defined type used in the signature is also marked as `#[export]`.
fn exportable_items_provider_local<'tcx>(tcx: TyCtxt<'tcx>, _: LocalCrate) -> &'tcx [DefId] {
    if !tcx.crate_types().contains(&CrateType::Sdylib) && !tcx.is_sdylib_interface_build() {
        return &[];
    }

    let mut visitor = ExportableItemCollector::new(tcx);
    tcx.hir_walk_toplevel_module(&mut visitor);
    let exportable_items = visitor.exportable_items;
    for item_id in exportable_items.iter() {
        let mut validator =
            ExportableItemsChecker { tcx, exportable_items: &exportable_items, item_id: *item_id };
        validator.check();
    }

    tcx.arena.alloc_from_iter(exportable_items.into_iter())
}

struct ImplsOrderVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    order: FxIndexMap<DefId, usize>,
}

impl<'tcx> ImplsOrderVisitor<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> ImplsOrderVisitor<'tcx> {
        ImplsOrderVisitor { tcx, order: Default::default() }
    }
}

impl<'tcx> Visitor<'tcx> for ImplsOrderVisitor<'tcx> {
    type NestedFilter = nested_filter::All;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        self.tcx
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        if let hir::ItemKind::Impl(impl_) = item.kind
            && impl_.of_trait.is_none()
            && self.tcx.is_exportable(item.owner_id.def_id.to_def_id())
        {
            self.order.insert(item.owner_id.def_id.to_def_id(), self.order.len());
        }
        intravisit::walk_item(self, item);
    }
}

/// During symbol mangling rustc uses a special index to distinguish between two impls of
/// the same type in the same module(See `DisambiguatedDefPathData`). For exportable items
/// we cannot use the current approach because it is dependent on the compiler's
/// implementation.
///
/// In order to make disambiguation independent of the compiler version we can assign an
/// id to each impl according to the relative order of elements in the source code.
fn stable_order_of_exportable_impls<'tcx>(
    tcx: TyCtxt<'tcx>,
    _: LocalCrate,
) -> &'tcx FxIndexMap<DefId, usize> {
    if !tcx.crate_types().contains(&CrateType::Sdylib) && !tcx.is_sdylib_interface_build() {
        return tcx.arena.alloc(FxIndexMap::<DefId, usize>::default());
    }

    let mut vis = ImplsOrderVisitor::new(tcx);
    tcx.hir_walk_toplevel_module(&mut vis);
    tcx.arena.alloc(vis.order)
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers {
        exportable_items: exportable_items_provider_local,
        stable_order_of_exportable_impls,
        ..*providers
    };
}
