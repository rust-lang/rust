//! This module analyzes crates to find examples of uses for items in the
//! current crate being documented.

use crate::clean;
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
crate struct CallLocation {
    crate call_span: (usize, usize),
    crate enclosing_item_span: (usize, usize),
    crate enclosing_item_lines: (usize, usize),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
crate struct CallData {
    crate locations: Vec<CallLocation>,
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

        if span.from_expansion() {
            return;
        }

        // Save call site if the function resolves to a concrete definition
        if let ty::FnDef(def_id, _) = ty.kind() {
            let fn_key = def_id_call_key(self.tcx, *def_id);
            let entries = self.calls.entry(fn_key).or_insert_with(FxHashMap::default);
            let file = self.tcx.sess.source_map().lookup_char_pos(span.lo()).file;
            let file_path = match file.name.clone() {
                FileName::Real(real_filename) => real_filename.into_local_path(),
                _ => None,
            };

            let get_pos =
                |bytepos: rustc_span::BytePos| file.original_relative_byte_pos(bytepos).0 as usize;
            let get_range = |span: rustc_span::Span| (get_pos(span.lo()), get_pos(span.hi()));
            let get_line = |bytepos: rustc_span::BytePos| file.lookup_line(bytepos).unwrap();
            let get_lines = |span: rustc_span::Span| (get_line(span.lo()), get_line(span.hi()));

            if let Some(file_path) = file_path {
                let abs_path = fs::canonicalize(file_path.clone()).unwrap();
                let cx = &self.cx;
                let enclosing_item_span =
                    self.tcx.hir().span_with_body(self.tcx.hir().get_parent_item(ex.hir_id));
                assert!(enclosing_item_span.contains(span));

                let location = CallLocation {
                    call_span: get_range(span),
                    enclosing_item_span: get_range(enclosing_item_span),
                    enclosing_item_lines: get_lines(enclosing_item_span),
                };

                entries
                    .entry(abs_path)
                    .or_insert_with(|| {
                        let clean_span = crate::clean::types::Span::new(span);
                        let url = cx.href_from_span(clean_span).unwrap();
                        let display_name = file_path.display().to_string();
                        CallData { locations: Vec::new(), url, display_name }
                    })
                    .locations
                    .push(location);
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

crate fn load_call_locations(
    with_examples: Vec<String>,
    diag: &rustc_errors::Handler,
) -> Result<Option<AllCallLocations>, i32> {
    let each_call_locations = with_examples
        .into_iter()
        .map(|path| {
            let bytes = fs::read(&path).map_err(|e| format!("{} (for path {})", e, path))?;
            let calls: AllCallLocations =
                serde_json::from_slice(&bytes).map_err(|e| format!("{}", e))?;
            Ok(calls)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e: String| {
            diag.err(&format!("failed to load examples with error: {}", e));
            1
        })?;

    Ok((each_call_locations.len() > 0).then(|| {
        each_call_locations.into_iter().fold(FxHashMap::default(), |mut acc, map| {
            for (function, calls) in map.into_iter() {
                acc.entry(function).or_insert_with(FxHashMap::default).extend(calls.into_iter());
            }
            acc
        })
    }))
}
