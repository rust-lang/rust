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
};
use rustc_interface::interface;
use rustc_macros::{Decodable, Encodable};
use rustc_middle::hir::map::Map;
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, TyCtxt};
use rustc_serialize::{
    opaque::{FileEncoder, MemDecoder},
    Decodable, Encodable,
};
use rustc_session::getopts;
use rustc_span::{
    def_id::{CrateNum, DefPathHash, LOCAL_CRATE},
    edition::Edition,
    BytePos, FileName, SourceFile,
};

use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub(crate) struct ScrapeExamplesOptions {
    output_path: PathBuf,
    target_crates: Vec<String>,
    pub(crate) scrape_tests: bool,
}

impl ScrapeExamplesOptions {
    pub(crate) fn new(
        matches: &getopts::Matches,
        diag: &rustc_errors::Handler,
    ) -> Result<Option<Self>, i32> {
        let output_path = matches.opt_str("scrape-examples-output-path");
        let target_crates = matches.opt_strs("scrape-examples-target-crate");
        let scrape_tests = matches.opt_present("scrape-tests");
        match (output_path, !target_crates.is_empty(), scrape_tests) {
            (Some(output_path), true, _) => Ok(Some(ScrapeExamplesOptions {
                output_path: PathBuf::from(output_path),
                target_crates,
                scrape_tests,
            })),
            (Some(_), false, _) | (None, true, _) => {
                diag.err("must use --scrape-examples-output-path and --scrape-examples-target-crate together");
                Err(1)
            }
            (None, false, true) => {
                diag.err("must use --scrape-examples-output-path and --scrape-examples-target-crate with --scrape-tests");
                Err(1)
            }
            (None, false, false) => Ok(None),
        }
    }
}

#[derive(Encodable, Decodable, Debug, Clone)]
pub(crate) struct SyntaxRange {
    pub(crate) byte_span: (u32, u32),
    pub(crate) line_span: (usize, usize),
}

impl SyntaxRange {
    fn new(span: rustc_span::Span, file: &SourceFile) -> Option<Self> {
        let get_pos = |bytepos: BytePos| file.original_relative_byte_pos(bytepos).0;
        let get_line = |bytepos: BytePos| file.lookup_line(bytepos);

        Some(SyntaxRange {
            byte_span: (get_pos(span.lo()), get_pos(span.hi())),
            line_span: (get_line(span.lo())?, get_line(span.hi())?),
        })
    }
}

#[derive(Encodable, Decodable, Debug, Clone)]
pub(crate) struct CallLocation {
    pub(crate) call_expr: SyntaxRange,
    pub(crate) call_ident: SyntaxRange,
    pub(crate) enclosing_item: SyntaxRange,
}

impl CallLocation {
    fn new(
        expr_span: rustc_span::Span,
        ident_span: rustc_span::Span,
        enclosing_item_span: rustc_span::Span,
        source_file: &SourceFile,
    ) -> Option<Self> {
        Some(CallLocation {
            call_expr: SyntaxRange::new(expr_span, source_file)?,
            call_ident: SyntaxRange::new(ident_span, source_file)?,
            enclosing_item: SyntaxRange::new(enclosing_item_span, source_file)?,
        })
    }
}

#[derive(Encodable, Decodable, Debug, Clone)]
pub(crate) struct CallData {
    pub(crate) locations: Vec<CallLocation>,
    pub(crate) url: String,
    pub(crate) display_name: String,
    pub(crate) edition: Edition,
    pub(crate) is_bin: bool,
}

pub(crate) type FnCallLocations = FxHashMap<PathBuf, CallData>;
pub(crate) type AllCallLocations = FxHashMap<DefPathHash, FnCallLocations>;

/// Visitor for traversing a crate and finding instances of function calls.
struct FindCalls<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    map: Map<'tcx>,
    cx: Context<'tcx>,
    target_crates: Vec<CrateNum>,
    calls: &'a mut AllCallLocations,
    bin_crate: bool,
}

impl<'a, 'tcx> Visitor<'tcx> for FindCalls<'a, 'tcx>
where
    'tcx: 'a,
{
    type NestedFilter = nested_filter::OnlyBodies;

    fn nested_visit_map(&mut self) -> Self::Map {
        self.map
    }

    fn visit_expr(&mut self, ex: &'tcx hir::Expr<'tcx>) {
        intravisit::walk_expr(self, ex);

        let tcx = self.tcx;

        // If we visit an item that contains an expression outside a function body,
        // then we need to exit before calling typeck (which will panic). See
        // test/run-make/rustdoc-scrape-examples-invalid-expr for an example.
        let hir = tcx.hir();
        if hir.maybe_body_owned_by(ex.hir_id.owner.def_id).is_none() {
            return;
        }

        // Get type of function if expression is a function call
        let (ty, call_span, ident_span) = match ex.kind {
            hir::ExprKind::Call(f, _) => {
                let types = tcx.typeck(ex.hir_id.owner.def_id);

                if let Some(ty) = types.node_type_opt(f.hir_id) {
                    (ty, ex.span, f.span)
                } else {
                    trace!("node_type_opt({}) = None", f.hir_id);
                    return;
                }
            }
            hir::ExprKind::MethodCall(path, _, _, call_span) => {
                let types = tcx.typeck(ex.hir_id.owner.def_id);
                let Some(def_id) = types.type_dependent_def_id(ex.hir_id) else {
                    trace!("type_dependent_def_id({}) = None", ex.hir_id);
                    return;
                };

                let ident_span = path.ident.span;
                (tcx.type_of(def_id).instantiate_identity(), call_span, ident_span)
            }
            _ => {
                return;
            }
        };

        // If this span comes from a macro expansion, then the source code may not actually show
        // a use of the given item, so it would be a poor example. Hence, we skip all uses in macros.
        if call_span.from_expansion() {
            trace!("Rejecting expr from macro: {call_span:?}");
            return;
        }

        // If the enclosing item has a span coming from a proc macro, then we also don't want to include
        // the example.
        let enclosing_item_span =
            tcx.hir().span_with_body(tcx.hir().get_parent_item(ex.hir_id).into());
        if enclosing_item_span.from_expansion() {
            trace!("Rejecting expr ({call_span:?}) from macro item: {enclosing_item_span:?}");
            return;
        }

        // If the enclosing item doesn't actually enclose the call, this means we probably have a weird
        // macro issue even though the spans aren't tagged as being from an expansion.
        if !enclosing_item_span.contains(call_span) {
            warn!(
                "Attempted to scrape call at [{call_span:?}] whose enclosing item [{enclosing_item_span:?}] doesn't contain the span of the call."
            );
            return;
        }

        // Similarly for the call w/ the function ident.
        if !call_span.contains(ident_span) {
            warn!(
                "Attempted to scrape call at [{call_span:?}] whose identifier [{ident_span:?}] was not contained in the span of the call."
            );
            return;
        }

        // Save call site if the function resolves to a concrete definition
        if let ty::FnDef(def_id, _) = ty.kind() {
            if self.target_crates.iter().all(|krate| *krate != def_id.krate) {
                trace!("Rejecting expr from crate not being documented: {call_span:?}");
                return;
            }

            let source_map = tcx.sess.source_map();
            let file = source_map.lookup_char_pos(call_span.lo()).file;
            let file_path = match file.name.clone() {
                FileName::Real(real_filename) => real_filename.into_local_path(),
                _ => None,
            };

            if let Some(file_path) = file_path {
                let abs_path = match fs::canonicalize(file_path.clone()) {
                    Ok(abs_path) => abs_path,
                    Err(_) => {
                        trace!("Could not canonicalize file path: {}", file_path.display());
                        return;
                    }
                };

                let cx = &self.cx;
                let clean_span = crate::clean::types::Span::new(call_span);
                let url = match cx.href_from_span(clean_span, false) {
                    Some(url) => url,
                    None => {
                        trace!(
                            "Rejecting expr ({call_span:?}) whose clean span ({clean_span:?}) cannot be turned into a link"
                        );
                        return;
                    }
                };

                let mk_call_data = || {
                    let display_name = file_path.display().to_string();
                    let edition = call_span.edition();
                    let is_bin = self.bin_crate;

                    CallData { locations: Vec::new(), url, display_name, edition, is_bin }
                };

                let fn_key = tcx.def_path_hash(*def_id);
                let fn_entries = self.calls.entry(fn_key).or_default();

                trace!("Including expr: {:?}", call_span);
                let enclosing_item_span =
                    source_map.span_extend_to_prev_char(enclosing_item_span, '\n', false);
                let location =
                    match CallLocation::new(call_span, ident_span, enclosing_item_span, &file) {
                        Some(location) => location,
                        None => {
                            trace!("Could not get serializable call location for {call_span:?}");
                            return;
                        }
                    };
                fn_entries.entry(abs_path).or_insert_with(mk_call_data).locations.push(location);
            }
        }
    }
}

pub(crate) fn run(
    krate: clean::Crate,
    mut renderopts: config::RenderOptions,
    cache: formats::cache::Cache,
    tcx: TyCtxt<'_>,
    options: ScrapeExamplesOptions,
    bin_crate: bool,
) -> interface::Result<()> {
    let inner = move || -> Result<(), String> {
        // Generates source files for examples
        renderopts.no_emit_shared = true;
        let (cx, _) = Context::init(krate, renderopts, cache, tcx).map_err(|e| e.to_string())?;

        // Collect CrateIds corresponding to provided target crates
        // If two different versions of the crate in the dependency tree, then examples will be collected from both.
        let all_crates = tcx
            .crates(())
            .iter()
            .chain([&LOCAL_CRATE])
            .map(|crate_num| (crate_num, tcx.crate_name(*crate_num)))
            .collect::<Vec<_>>();
        let target_crates = options
            .target_crates
            .into_iter()
            .flat_map(|target| all_crates.iter().filter(move |(_, name)| name.as_str() == target))
            .map(|(crate_num, _)| **crate_num)
            .collect::<Vec<_>>();

        debug!("All crates in TyCtxt: {all_crates:?}");
        debug!("Scrape examples target_crates: {target_crates:?}");

        // Run call-finder on all items
        let mut calls = FxHashMap::default();
        let mut finder =
            FindCalls { calls: &mut calls, tcx, map: tcx.hir(), cx, target_crates, bin_crate };
        tcx.hir().visit_all_item_likes_in_crate(&mut finder);

        // The visitor might have found a type error, which we need to
        // promote to a fatal error
        if tcx.sess.diagnostic().has_errors_or_lint_errors().is_some() {
            return Err(String::from("Compilation failed, aborting rustdoc"));
        }

        // Sort call locations within a given file in document order
        for fn_calls in calls.values_mut() {
            for file_calls in fn_calls.values_mut() {
                file_calls.locations.sort_by_key(|loc| loc.call_expr.byte_span.0);
            }
        }

        // Save output to provided path
        let mut encoder = FileEncoder::new(options.output_path).map_err(|e| e.to_string())?;
        calls.encode(&mut encoder);
        encoder.finish().map_err(|e| e.to_string())?;

        Ok(())
    };

    if let Err(e) = inner() {
        tcx.sess.fatal(e);
    }

    Ok(())
}

// Note: the Handler must be passed in explicitly because sess isn't available while parsing options
pub(crate) fn load_call_locations(
    with_examples: Vec<String>,
    diag: &rustc_errors::Handler,
) -> Result<AllCallLocations, i32> {
    let inner = || {
        let mut all_calls: AllCallLocations = FxHashMap::default();
        for path in with_examples {
            let bytes = fs::read(&path).map_err(|e| format!("{} (for path {})", e, path))?;
            let mut decoder = MemDecoder::new(&bytes, 0);
            let calls = AllCallLocations::decode(&mut decoder);

            for (function, fn_calls) in calls.into_iter() {
                all_calls.entry(function).or_default().extend(fn_calls.into_iter());
            }
        }

        Ok(all_calls)
    };

    inner().map_err(|e: String| {
        diag.err(format!("failed to load examples: {}", e));
        1
    })
}
