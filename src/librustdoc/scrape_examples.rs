//! This module analyzes provided crates to find examples of uses for items in the
//! current crate being documented.

use crate::clean;
use crate::clean::types::Span;
use crate::config;
use crate::formats;
use crate::formats::renderer::FormatRenderer;
use crate::html::render::Context;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{
    self as hir,
    intravisit::{self, Visitor},
};
use rustc_interface::interface;
use rustc_middle::hir::map::Map;
use rustc_middle::ty::{self, TyCtxt};
use rustc_span::{def_id::DefId, FileName};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Debug, Clone)]
crate struct CallData {
    crate locations: Vec<(usize, usize)>,
    crate url: String,
    crate display_name: String,
}
crate type DefIdCallKey = String;
crate type FnCallLocations = FxHashMap<PathBuf, CallData>;
crate type AllCallLocations = FxHashMap<DefIdCallKey, FnCallLocations>;

/// Visitor for traversing a crate and finding instances of function calls.
struct FindCalls<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    map: Map<'tcx>,
    cx: Context<'tcx>,
    calls: &'a mut AllCallLocations,
}

crate fn def_id_call_key(tcx: TyCtxt<'_>, def_id: DefId) -> DefIdCallKey {
    format!(
        "{}{}",
        tcx.crate_name(def_id.krate).to_ident_string(),
        tcx.def_path(def_id).to_string_no_crate_verbose()
    )
}

impl<'a, 'tcx> Visitor<'tcx> for FindCalls<'a, 'tcx>
where
    'tcx: 'a,
{
    type Map = Map<'tcx>;

    fn nested_visit_map(&mut self) -> intravisit::NestedVisitorMap<Self::Map> {
        intravisit::NestedVisitorMap::OnlyBodies(self.map)
    }

    fn visit_expr(&mut self, ex: &'tcx hir::Expr<'tcx>) {
        intravisit::walk_expr(self, ex);

        // Get type of function if expression is a function call
        let (ty, span) = match ex.kind {
            hir::ExprKind::Call(f, _) => {
                let types = self.tcx.typeck(ex.hir_id.owner);
                (types.node_type(f.hir_id), ex.span)
            }
            hir::ExprKind::MethodCall(_, _, _, span) => {
                let types = self.tcx.typeck(ex.hir_id.owner);
                let def_id = types.type_dependent_def_id(ex.hir_id).unwrap();
                (self.tcx.type_of(def_id), span)
            }
            _ => {
                return;
            }
        };

        // Save call site if the function resolves to a concrete definition
        if let ty::FnDef(def_id, _) = ty.kind() {
            let fn_key = def_id_call_key(self.tcx, *def_id);
            let entries = self.calls.entry(fn_key).or_insert_with(FxHashMap::default);
            let filename = self.tcx.sess.source_map().span_to_filename(span);
            let file_path = match filename {
                FileName::Real(real_filename) => real_filename.into_local_path(),
                _ => None,
            };

            if let Some(file_path) = file_path {
                let abs_path = fs::canonicalize(file_path.clone()).unwrap();
                let cx = &self.cx;
                entries
                    .entry(abs_path)
                    .or_insert_with(|| {
                        let url = cx.src_href(Span::from_rustc_span(span), false).unwrap();
                        let display_name = file_path.display().to_string();
                        CallData { locations: Vec::new(), url, display_name }
                    })
                    .locations
                    .push((span.lo().0 as usize, span.hi().0 as usize));
            }
        }
    }
}

crate fn run(
    krate: clean::Crate,
    renderopts: config::RenderOptions,
    cache: formats::cache::Cache,
    tcx: TyCtxt<'tcx>,
    example_path: PathBuf,
) -> interface::Result<()> {
    let inner = move || {
        // Generates source files for examples
        let (cx, _) = Context::init(krate, renderopts, cache, tcx).map_err(|e| format!("{}", e))?;

        // Run call-finder on all items
        let mut calls = FxHashMap::default();
        let mut finder = FindCalls { calls: &mut calls, tcx, map: tcx.hir(), cx };
        tcx.hir().krate().visit_all_item_likes(&mut finder.as_deep_visitor());

        // Save output JSON to provided path
        let calls_json = serde_json::to_string(&calls).map_err(|e| format!("{}", e))?;
        fs::write(example_path, &calls_json).map_err(|e| format!("{}", e))?;

        Ok(())
    };

    inner().map_err(|e: String| {
        eprintln!("{}", e);
        rustc_errors::ErrorReported
    })
}
