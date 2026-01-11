//! Walks the crate looking for items/impl-items/trait-items that have
//! either a `rustc_symbol_name` or `rustc_def_path` attribute and
//! generates an error giving, respectively, the symbol name or
//! def-path. This is used for unit testing the code that generates
//! paths etc in all kinds of annoying scenarios.

use rustc_hir::def_id::LocalDefId;
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{GenericArgs, Instance, TyCtxt};
use rustc_span::Span;

use crate::errors::{Kind, TestOutput};

#[inline(always)]
pub fn process_symbol_name_attr<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, attr_span: Span) {
    let def_id = def_id.to_def_id();
    let instance = Instance::new_raw(
        def_id,
        tcx.erase_and_anonymize_regions(GenericArgs::identity_for_item(tcx, def_id)),
    );
    let mangled = tcx.symbol_name(instance);
    tcx.dcx().emit_err(TestOutput {
        span: attr_span,
        kind: Kind::SymbolName,
        content: format!("{mangled}"),
    });
    if let Ok(demangling) = rustc_demangle::try_demangle(mangled.name) {
        tcx.dcx().emit_err(TestOutput {
            span: attr_span,
            kind: Kind::Demangling,
            content: format!("{demangling}"),
        });
        tcx.dcx().emit_err(TestOutput {
            span: attr_span,
            kind: Kind::DemanglingAlt,
            content: format!("{demangling:#}"),
        });
    }
}

#[inline(always)]
pub fn process_def_path_attr<'tcx>(tcx: TyCtxt<'tcx>, def_id: LocalDefId, attr_span: Span) {
    tcx.dcx().emit_err(TestOutput {
        span: attr_span,
        kind: Kind::DefPath,
        content: with_no_trimmed_paths!(tcx.def_path_str(def_id)),
    });
}
