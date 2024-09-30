use std::iter;
use std::ops::ControlFlow;

use rustc_data_structures::fx::FxIndexSet;
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::{DefId, LocalDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_middle::hir::nested_filter;
use rustc_middle::middle::privacy::{EffectiveVisibility, Level};
use rustc_middle::query::{LocalCrate, Providers};
use rustc_middle::ty::{
    self, Ty, TyCtxt, TypeSuperVisitable, TypeVisitable, TypeVisitor, Visibility,
};
use rustc_session::config::SymbolManglingVersion;
use rustc_span::{Span, sym};
use rustc_target::spec::abi::Abi;

use crate::errors;

struct ExportableItemCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    exportable_items: FxIndexSet<DefId>,
    in_exportable_mod: bool,
}

impl<'tcx> ExportableItemCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> ExportableItemCollector<'tcx> {
        ExportableItemCollector {
            tcx,
            exportable_items: Default::default(),
            in_exportable_mod: false,
        }
    }

    fn report_wrong_site(&self, def_id: LocalDefId) {
        let def_descr = self.tcx.def_descr(def_id.to_def_id());
        self.tcx.dcx().emit_err(errors::UnexportableItem {
            descr: &format!("{}'s are not exportable", def_descr),
            span: self.tcx.def_span(def_id),
        });
    }

    fn item_is_exportable(&self, def_id: LocalDefId) -> bool {
        let has_attr = self.tcx.has_attr(def_id, sym::export);
        if !self.in_exportable_mod && !has_attr {
            return false;
        }

        let visibilities = self.tcx.effective_visibilities(());
        let is_pub = visibilities.is_directly_public(def_id);

        if has_attr && !is_pub {
            let vis = visibilities.effective_vis(def_id).cloned().unwrap_or(
                EffectiveVisibility::from_vis(Visibility::Restricted(
                    self.tcx.parent_module_from_def_id(def_id).to_local_def_id(),
                )),
            );
            let vis = vis.at_level(Level::Direct);
            let span = self.tcx.def_span(def_id);

            self.tcx.dcx().emit_err(errors::UnexportablePrivItem {
                vis_note: span,
                vis_descr: &vis.to_string(def_id, self.tcx),
                span,
            });
            return false;
        }

        is_pub && (has_attr || self.in_exportable_mod)
    }
}

impl<'tcx> Visitor<'tcx> for ExportableItemCollector<'tcx> {
    type NestedFilter = nested_filter::All;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.tcx.hir()
    }

    fn visit_item(&mut self, item: &'tcx hir::Item<'tcx>) {
        let def_id = item.hir_id().owner.def_id;
        // Applying #[extern] attribute to modules is simply equivalent to
        // applying the attribute to every public item within it.
        match item.kind {
            hir::ItemKind::Mod(..) => {
                intravisit::walk_item(self, item);
                return;
            }
            hir::ItemKind::Impl(impl_) if impl_.of_trait.is_none() => {
                let old_exportable_mod = self.in_exportable_mod;
                if self.tcx.get_attr(def_id, sym::export).is_some() {
                    self.in_exportable_mod = true;
                }
                intravisit::walk_item(self, item);
                self.in_exportable_mod = old_exportable_mod;
                return;
            }
            _ => {}
        }

        if !self.item_is_exportable(def_id) {
            return;
        }

        match item.kind {
            hir::ItemKind::Fn(..)
            | hir::ItemKind::Struct(..)
            | hir::ItemKind::Enum(..)
            | hir::ItemKind::Union(..)
            | hir::ItemKind::TyAlias(..) => {
                self.exportable_items.insert(def_id.to_def_id());
            }
            hir::ItemKind::Use(path, _) => {
                for res in &path.res {
                    // Only local items are exportable.
                    if let Some(res_id) = res.opt_def_id()
                        && res_id.is_local()
                    {
                        self.exportable_items.insert(res_id);
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
                self.exportable_items.insert(def_id.to_def_id());
            }
            _ => self.report_wrong_site(def_id),
        }
    }

    fn visit_mod(&mut self, m: &'tcx hir::Mod<'tcx>, _: Span, hir_id: hir::HirId) {
        let old_exportable_mod = self.in_exportable_mod;
        if self.tcx.get_attr(hir_id.owner.def_id, sym::export).is_some() {
            self.in_exportable_mod = true;
        }
        intravisit::walk_mod(self, m, hir_id);
        self.in_exportable_mod = old_exportable_mod;
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

struct ExportableItemsValidator<'tcx, 'a> {
    tcx: TyCtxt<'tcx>,
    exportable_items: &'a FxIndexSet<DefId>,
    item_id: DefId,
}

impl<'tcx, 'a> ExportableItemsValidator<'tcx, 'a> {
    fn check(&mut self) {
        match self.tcx.def_kind(self.item_id) {
            DefKind::Fn | DefKind::AssocFn => self.check_fn(),
            DefKind::Enum
            | DefKind::Struct
            | DefKind::Union
            | DefKind::TyAlias
            | DefKind::AssocTy => self.check_ty(),
            _ => unreachable!(),
        }
        if self.tcx.sess.opts.get_symbol_mangling_version() != SymbolManglingVersion::V0
            && self.tcx.sess.opts.output_types.should_codegen()
        {
            self.tcx.dcx().emit_err(errors::UnexportableItem {
                span: self.tcx.def_span(self.item_id),
                descr: "`#[export]` attribute is only usable with `v0` mangling scheme",
            });
        }
    }

    fn check_fn(&mut self) {
        let def_id = self.item_id.expect_local();
        if !self.check_generics(def_id) {
            return;
        }

        let sig = self.tcx.fn_sig(def_id).instantiate_identity().skip_binder();
        let span = self.tcx.def_span(def_id);
        if !matches!(sig.abi, Abi::C { .. }) {
            self.tcx.dcx().emit_err(errors::UnexportableItem {
                descr: "Only `extern \"C\"` functions are exportable",
                span,
            });
            return;
        }

        let sig = self
            .tcx
            .try_normalize_erasing_regions(ty::TypingEnv::non_body_analysis(self.tcx, def_id), sig)
            .unwrap_or(sig);

        let hir_id = self.tcx.local_def_id_to_hir_id(def_id);
        let decl = self.tcx.hir().fn_decl_by_hir_id(hir_id).unwrap();

        for (input_ty, input_hir) in iter::zip(sig.inputs(), decl.inputs) {
            self.check_nested(*input_ty, input_hir.span);
        }

        match decl.output {
            hir::FnRetTy::Return(ret_hir) => self.check_nested(sig.output(), ret_hir.span),
            hir::FnRetTy::DefaultReturn(span) => {
                self.tcx.dcx().emit_err(errors::UnexportableTypeInInterface {
                    span: self.tcx.def_span(self.item_id),
                    desc: self.tcx.def_descr(self.item_id),
                    ty: &format!("{}", self.tcx.types.unit),
                    ty_span: span,
                });
            }
        }
    }

    fn check_ty(&mut self) {
        let ty = self.tcx.type_of(self.item_id).skip_binder();
        if let ty::Adt(adt_def, _) = ty.kind() {
            let mut is_err = false;
            if !adt_def.repr().inhibit_struct_field_reordering() {
                self.tcx.dcx().emit_err(errors::UnexportableItem {
                    descr: "types with unstable layout are not exportable",
                    span: self.tcx.def_span(self.item_id),
                });
                is_err = true;
            }

            for variant in adt_def.variants() {
                for field in &variant.fields {
                    if !field.vis.is_public() {
                        self.tcx.dcx().emit_err(errors::UnexportableAdtWithPrivFields {
                            span: self.tcx.def_span(self.item_id),
                            vis_note: self.tcx.def_span(field.did),
                            field_name: field.name.as_str(),
                        });
                        is_err = true;
                    }
                }
            }
            if is_err {
                return;
            }
        }
        let item_span = self.tcx.def_span(self.item_id);
        self.check_nested(ty, item_span);
    }

    fn check_generics(&self, def_id: LocalDefId) -> bool {
        let span = self.tcx.def_span(def_id);
        let def_descr = self.tcx.def_descr(def_id.to_def_id());

        let generics = self.tcx.generics_of(def_id);
        if generics.requires_monomorphization(self.tcx) {
            self.tcx.dcx().emit_err(errors::UnexportableItem {
                descr: &format!("generic {}'s are not exportable", def_descr),
                span,
            });
            return false;
        }
        true
    }

    fn check_nested(&mut self, ty: Ty<'tcx>, ty_span: Span) {
        let res = ty.visit_with(self);
        if let Some(err_cause) = res.break_value() {
            self.tcx.dcx().emit_err(errors::UnexportableTypeInInterface {
                span: self.tcx.def_span(self.item_id),
                desc: self.tcx.def_descr(self.item_id),
                ty: &format!("{}", err_cause),
                ty_span,
            });
        }
    }
}

impl<'tcx, 'a> TypeVisitor<TyCtxt<'tcx>> for ExportableItemsValidator<'tcx, 'a> {
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

            ty::Int(_) | ty::Uint(_) | ty::Float(_) | ty::Bool | ty::Char => {}

            // Allow the use of type parameters. Generic functions have already been rejected.
            ty::Param(_) => {}

            ty::Array(_, _)
            | ty::Ref(_, _, _)
            | ty::Closure(_, _)
            | ty::Dynamic(_, _, _)
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
            | ty::Never => {
                return ControlFlow::Break(ty);
            }

            ty::Error(_) => {}

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
///    defined type used in the signature is also marked as #[export].
fn exportable_items_provider_local<'tcx>(tcx: TyCtxt<'tcx>, _: LocalCrate) -> &'tcx [DefId] {
    let mut visitor = ExportableItemCollector::new(tcx);
    tcx.hir().walk_toplevel_module(&mut visitor);
    let exportable_items = visitor.exportable_items;
    for item_id in exportable_items.iter() {
        let mut validator = ExportableItemsValidator {
            tcx,
            exportable_items: &exportable_items,
            item_id: *item_id,
        };
        validator.check();
    }

    tcx.arena.alloc_from_iter(exportable_items.into_iter())
}

pub(crate) fn provide(providers: &mut Providers) {
    *providers = Providers { exportable_items: exportable_items_provider_local, ..*providers };
}
