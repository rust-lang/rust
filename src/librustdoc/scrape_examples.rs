//! This module analyzes crates to find call sites that can serve as examples in the documentation.

use crate::clean;
use crate::config;
use crate::formats;
use crate::formats::renderer::FormatRenderer;
use crate::html::render::Context;

use rustc_data_structures::fx::FxHashMap;
use rustc_hir::{
    self as hir,
    intravisit::{self, Visitor},
    HirId,
};
use rustc_interface::interface;
use rustc_macros::{Decodable, Encodable};
use rustc_middle::hir::map::Map;
use rustc_middle::ty::{self, TyCtxt};
use rustc_serialize::{
    opaque::{Decoder, FileEncoder},
    Decodable, Encodable,
};
use rustc_span::{def_id::DefId, edition::Edition, BytePos, FileName, SourceFile};

use std::fs;
use std::path::PathBuf;

#[derive(Encodable, Decodable, Debug, Clone)]
crate struct SyntaxRange {
    crate byte_span: (u32, u32),
    crate line_span: (usize, usize),
}

impl SyntaxRange {
    fn new(span: rustc_span::Span, file: &SourceFile) -> Self {
        let get_pos = |bytepos: BytePos| file.original_relative_byte_pos(bytepos).0;
        let get_line = |bytepos: BytePos| file.lookup_line(bytepos).unwrap();

        SyntaxRange {
            byte_span: (get_pos(span.lo()), get_pos(span.hi())),
            line_span: (get_line(span.lo()), get_line(span.hi())),
        }
    }
}

#[derive(Encodable, Decodable, Debug, Clone)]
crate struct CallLocation {
    crate call_expr: SyntaxRange,
    crate enclosing_item: SyntaxRange,
}

impl CallLocation {
    fn new(
        tcx: TyCtxt<'_>,
        expr_span: rustc_span::Span,
        expr_id: HirId,
        source_file: &SourceFile,
    ) -> Self {
        let enclosing_item_span =
            tcx.hir().span_with_body(tcx.hir().get_parent_item(expr_id)).source_callsite();
        assert!(enclosing_item_span.contains(expr_span));

        CallLocation {
            call_expr: SyntaxRange::new(expr_span, source_file),
            enclosing_item: SyntaxRange::new(enclosing_item_span, source_file),
        }
    }
}

#[derive(Encodable, Decodable, Debug, Clone)]
crate struct CallData {
    crate locations: Vec<CallLocation>,
    crate url: String,
    crate display_name: String,
    crate edition: Edition,
}

crate type FnCallLocations = FxHashMap<PathBuf, CallData>;
crate type AllCallLocations = FxHashMap<String, FnCallLocations>;

/// Visitor for traversing a crate and finding instances of function calls.
struct FindCalls<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    map: Map<'tcx>,
    cx: Context<'tcx>,
    calls: &'a mut AllCallLocations,
}

crate fn def_id_call_key(tcx: TyCtxt<'_>, def_id: DefId) -> String {
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
        let tcx = self.tcx;
        let (ty, span) = match ex.kind {
            hir::ExprKind::Call(f, _) => {
                let types = tcx.typeck(ex.hir_id.owner);
                (types.node_type(f.hir_id), ex.span)
            }
            hir::ExprKind::MethodCall(_, _, _, span) => {
                let types = tcx.typeck(ex.hir_id.owner);
                let def_id = types.type_dependent_def_id(ex.hir_id).unwrap();
                (tcx.type_of(def_id), span)
            }
            _ => {
                return;
            }
        };

        // We need to get the file the example originates in. If the call is contained
        // in a macro, then trace the span back to the macro source (rather than macro definition).
        let span = span.source_callsite();

        // Save call site if the function resolves to a concrete definition
        if let ty::FnDef(def_id, _) = ty.kind() {
            let file = tcx.sess.source_map().lookup_char_pos(span.lo()).file;
            let file_path = match file.name.clone() {
                FileName::Real(real_filename) => real_filename.into_local_path(),
                _ => None,
            };

            if let Some(file_path) = file_path {
                let abs_path = fs::canonicalize(file_path.clone()).unwrap();
                let cx = &self.cx;
                let mk_call_data = || {
                    let clean_span = crate::clean::types::Span::new(span);
                    let url = cx.href_from_span(clean_span).unwrap();
                    let display_name = file_path.display().to_string();
                    let edition = tcx.sess.edition();
                    CallData { locations: Vec::new(), url, display_name, edition }
                };

                let fn_key = def_id_call_key(tcx, *def_id);
                let fn_entries = self.calls.entry(fn_key).or_default();

                let location = CallLocation::new(tcx, span, ex.hir_id, &file);
                fn_entries.entry(abs_path).or_insert_with(mk_call_data).locations.push(location);
            }
        }
    }
}

crate fn run(
    krate: clean::Crate,
    renderopts: config::RenderOptions,
    cache: formats::cache::Cache,
    tcx: TyCtxt<'_>,
    example_path: PathBuf,
) -> interface::Result<()> {
    let inner = move || -> Result<(), String> {
        // Generates source files for examples
        let (cx, _) = Context::init(krate, renderopts, cache, tcx).map_err(|e| e.to_string())?;

        // Run call-finder on all items
        let mut calls = FxHashMap::default();
        let mut finder = FindCalls { calls: &mut calls, tcx, map: tcx.hir(), cx };
        tcx.hir().krate().visit_all_item_likes(&mut finder.as_deep_visitor());

        // Save output to provided path
        let mut encoder = FileEncoder::new(example_path).map_err(|e| e.to_string())?;
        calls.encode(&mut encoder).map_err(|e| e.to_string())?;
        encoder.flush().map_err(|e| e.to_string())?;

        Ok(())
    };

    if let Err(e) = inner() {
        tcx.sess.fatal(&e);
    }

    Ok(())
}

crate fn load_call_locations(
    with_examples: Vec<String>,
    diag: &rustc_errors::Handler,
) -> Result<AllCallLocations, i32> {
    let inner = || {
        let mut all_calls: AllCallLocations = FxHashMap::default();
        for path in with_examples {
            let bytes = fs::read(&path).map_err(|e| format!("{} (for path {})", e, path))?;
            let mut decoder = Decoder::new(&bytes, 0);
            let calls = AllCallLocations::decode(&mut decoder)?;

            for (function, fn_calls) in calls.into_iter() {
                all_calls.entry(function).or_default().extend(fn_calls.into_iter());
            }
        }

        Ok(all_calls)
    };

    inner().map_err(|e: String| {
        diag.err(&format!("failed to load examples: {}", e));
        1
    })
}
