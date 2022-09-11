//! The various pretty-printing routines.

use crate::session_diagnostics::UnprettyDumpFail;
use rustc_ast as ast;
use rustc_ast_pretty::pprust;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir_pretty as pprust_hir;
use rustc_middle::hir::map as hir_map;
use rustc_middle::mir::{write_mir_graphviz, write_mir_pretty};
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::config::{Input, PpAstTreeMode, PpHirMode, PpMode, PpSourceMode};
use rustc_session::Session;
use rustc_span::symbol::Ident;
use rustc_span::FileName;

use std::cell::Cell;
use std::fmt::Write;
use std::path::Path;

pub use self::PpMode::*;
pub use self::PpSourceMode::*;
use crate::abort_on_err;

// This slightly awkward construction is to allow for each PpMode to
// choose whether it needs to do analyses (which can consume the
// Session) and then pass through the session (now attached to the
// analysis results) on to the chosen pretty-printer, along with the
// `&PpAnn` object.
//
// Note that since the `&PrinterSupport` is freshly constructed on each
// call, it would not make sense to try to attach the lifetime of `self`
// to the lifetime of the `&PrinterObject`.

/// Constructs a `PrinterSupport` object and passes it to `f`.
fn call_with_pp_support<'tcx, A, F>(
    ppmode: &PpSourceMode,
    sess: &'tcx Session,
    tcx: Option<TyCtxt<'tcx>>,
    f: F,
) -> A
where
    F: FnOnce(&dyn PrinterSupport) -> A,
{
    match *ppmode {
        Normal | Expanded => {
            let annotation = NoAnn { sess, tcx };
            f(&annotation)
        }

        Identified | ExpandedIdentified => {
            let annotation = IdentifiedAnnotation { sess, tcx };
            f(&annotation)
        }
        ExpandedHygiene => {
            let annotation = HygieneAnnotation { sess };
            f(&annotation)
        }
    }
}
fn call_with_pp_support_hir<A, F>(ppmode: &PpHirMode, tcx: TyCtxt<'_>, f: F) -> A
where
    F: FnOnce(&dyn HirPrinterSupport<'_>, hir_map::Map<'_>) -> A,
{
    match *ppmode {
        PpHirMode::Normal => {
            let annotation = NoAnn { sess: tcx.sess, tcx: Some(tcx) };
            f(&annotation, tcx.hir())
        }

        PpHirMode::Identified => {
            let annotation = IdentifiedAnnotation { sess: tcx.sess, tcx: Some(tcx) };
            f(&annotation, tcx.hir())
        }
        PpHirMode::Typed => {
            abort_on_err(tcx.analysis(()), tcx.sess);

            let annotation = TypedAnnotation { tcx, maybe_typeck_results: Cell::new(None) };
            tcx.dep_graph.with_ignore(|| f(&annotation, tcx.hir()))
        }
    }
}

trait PrinterSupport: pprust::PpAnn {
    /// Provides a uniform interface for re-extracting a reference to a
    /// `Session` from a value that now owns it.
    fn sess(&self) -> &Session;

    /// Produces the pretty-print annotation object.
    ///
    /// (Rust does not yet support upcasting from a trait object to
    /// an object for one of its supertraits.)
    fn pp_ann(&self) -> &dyn pprust::PpAnn;
}

trait HirPrinterSupport<'hir>: pprust_hir::PpAnn {
    /// Provides a uniform interface for re-extracting a reference to a
    /// `Session` from a value that now owns it.
    fn sess(&self) -> &Session;

    /// Provides a uniform interface for re-extracting a reference to an
    /// `hir_map::Map` from a value that now owns it.
    fn hir_map(&self) -> Option<hir_map::Map<'hir>>;

    /// Produces the pretty-print annotation object.
    ///
    /// (Rust does not yet support upcasting from a trait object to
    /// an object for one of its supertraits.)
    fn pp_ann(&self) -> &dyn pprust_hir::PpAnn;
}

struct NoAnn<'hir> {
    sess: &'hir Session,
    tcx: Option<TyCtxt<'hir>>,
}

impl<'hir> PrinterSupport for NoAnn<'hir> {
    fn sess(&self) -> &Session {
        self.sess
    }

    fn pp_ann(&self) -> &dyn pprust::PpAnn {
        self
    }
}

impl<'hir> HirPrinterSupport<'hir> for NoAnn<'hir> {
    fn sess(&self) -> &Session {
        self.sess
    }

    fn hir_map(&self) -> Option<hir_map::Map<'hir>> {
        self.tcx.map(|tcx| tcx.hir())
    }

    fn pp_ann(&self) -> &dyn pprust_hir::PpAnn {
        self
    }
}

impl<'hir> pprust::PpAnn for NoAnn<'hir> {}
impl<'hir> pprust_hir::PpAnn for NoAnn<'hir> {
    fn nested(&self, state: &mut pprust_hir::State<'_>, nested: pprust_hir::Nested) {
        if let Some(tcx) = self.tcx {
            pprust_hir::PpAnn::nested(&(&tcx.hir() as &dyn hir::intravisit::Map<'_>), state, nested)
        }
    }
}

struct IdentifiedAnnotation<'hir> {
    sess: &'hir Session,
    tcx: Option<TyCtxt<'hir>>,
}

impl<'hir> PrinterSupport for IdentifiedAnnotation<'hir> {
    fn sess(&self) -> &Session {
        self.sess
    }

    fn pp_ann(&self) -> &dyn pprust::PpAnn {
        self
    }
}

impl<'hir> pprust::PpAnn for IdentifiedAnnotation<'hir> {
    fn pre(&self, s: &mut pprust::State<'_>, node: pprust::AnnNode<'_>) {
        if let pprust::AnnNode::Expr(_) = node {
            s.popen();
        }
    }
    fn post(&self, s: &mut pprust::State<'_>, node: pprust::AnnNode<'_>) {
        match node {
            pprust::AnnNode::Crate(_) | pprust::AnnNode::Ident(_) | pprust::AnnNode::Name(_) => {}

            pprust::AnnNode::Item(item) => {
                s.s.space();
                s.synth_comment(item.id.to_string())
            }
            pprust::AnnNode::SubItem(id) => {
                s.s.space();
                s.synth_comment(id.to_string())
            }
            pprust::AnnNode::Block(blk) => {
                s.s.space();
                s.synth_comment(format!("block {}", blk.id))
            }
            pprust::AnnNode::Expr(expr) => {
                s.s.space();
                s.synth_comment(expr.id.to_string());
                s.pclose()
            }
            pprust::AnnNode::Pat(pat) => {
                s.s.space();
                s.synth_comment(format!("pat {}", pat.id));
            }
        }
    }
}

impl<'hir> HirPrinterSupport<'hir> for IdentifiedAnnotation<'hir> {
    fn sess(&self) -> &Session {
        self.sess
    }

    fn hir_map(&self) -> Option<hir_map::Map<'hir>> {
        self.tcx.map(|tcx| tcx.hir())
    }

    fn pp_ann(&self) -> &dyn pprust_hir::PpAnn {
        self
    }
}

impl<'hir> pprust_hir::PpAnn for IdentifiedAnnotation<'hir> {
    fn nested(&self, state: &mut pprust_hir::State<'_>, nested: pprust_hir::Nested) {
        if let Some(ref tcx) = self.tcx {
            pprust_hir::PpAnn::nested(&(&tcx.hir() as &dyn hir::intravisit::Map<'_>), state, nested)
        }
    }
    fn pre(&self, s: &mut pprust_hir::State<'_>, node: pprust_hir::AnnNode<'_>) {
        if let pprust_hir::AnnNode::Expr(_) = node {
            s.popen();
        }
    }
    fn post(&self, s: &mut pprust_hir::State<'_>, node: pprust_hir::AnnNode<'_>) {
        match node {
            pprust_hir::AnnNode::Name(_) => {}
            pprust_hir::AnnNode::Item(item) => {
                s.s.space();
                s.synth_comment(format!("hir_id: {}", item.hir_id()));
            }
            pprust_hir::AnnNode::SubItem(id) => {
                s.s.space();
                s.synth_comment(id.to_string());
            }
            pprust_hir::AnnNode::Block(blk) => {
                s.s.space();
                s.synth_comment(format!("block hir_id: {}", blk.hir_id));
            }
            pprust_hir::AnnNode::Expr(expr) => {
                s.s.space();
                s.synth_comment(format!("expr hir_id: {}", expr.hir_id));
                s.pclose();
            }
            pprust_hir::AnnNode::Pat(pat) => {
                s.s.space();
                s.synth_comment(format!("pat hir_id: {}", pat.hir_id));
            }
            pprust_hir::AnnNode::Arm(arm) => {
                s.s.space();
                s.synth_comment(format!("arm hir_id: {}", arm.hir_id));
            }
        }
    }
}

struct HygieneAnnotation<'a> {
    sess: &'a Session,
}

impl<'a> PrinterSupport for HygieneAnnotation<'a> {
    fn sess(&self) -> &Session {
        self.sess
    }

    fn pp_ann(&self) -> &dyn pprust::PpAnn {
        self
    }
}

impl<'a> pprust::PpAnn for HygieneAnnotation<'a> {
    fn post(&self, s: &mut pprust::State<'_>, node: pprust::AnnNode<'_>) {
        match node {
            pprust::AnnNode::Ident(&Ident { name, span }) => {
                s.s.space();
                s.synth_comment(format!("{}{:?}", name.as_u32(), span.ctxt()))
            }
            pprust::AnnNode::Name(&name) => {
                s.s.space();
                s.synth_comment(name.as_u32().to_string())
            }
            pprust::AnnNode::Crate(_) => {
                s.s.hardbreak();
                let verbose = self.sess.verbose();
                s.synth_comment(rustc_span::hygiene::debug_hygiene_data(verbose));
                s.s.hardbreak_if_not_bol();
            }
            _ => {}
        }
    }
}

struct TypedAnnotation<'tcx> {
    tcx: TyCtxt<'tcx>,
    maybe_typeck_results: Cell<Option<&'tcx ty::TypeckResults<'tcx>>>,
}

impl<'tcx> HirPrinterSupport<'tcx> for TypedAnnotation<'tcx> {
    fn sess(&self) -> &Session {
        self.tcx.sess
    }

    fn hir_map(&self) -> Option<hir_map::Map<'tcx>> {
        Some(self.tcx.hir())
    }

    fn pp_ann(&self) -> &dyn pprust_hir::PpAnn {
        self
    }
}

impl<'tcx> pprust_hir::PpAnn for TypedAnnotation<'tcx> {
    fn nested(&self, state: &mut pprust_hir::State<'_>, nested: pprust_hir::Nested) {
        let old_maybe_typeck_results = self.maybe_typeck_results.get();
        if let pprust_hir::Nested::Body(id) = nested {
            self.maybe_typeck_results.set(Some(self.tcx.typeck_body(id)));
        }
        let pp_ann = &(&self.tcx.hir() as &dyn hir::intravisit::Map<'_>);
        pprust_hir::PpAnn::nested(pp_ann, state, nested);
        self.maybe_typeck_results.set(old_maybe_typeck_results);
    }
    fn pre(&self, s: &mut pprust_hir::State<'_>, node: pprust_hir::AnnNode<'_>) {
        if let pprust_hir::AnnNode::Expr(_) = node {
            s.popen();
        }
    }
    fn post(&self, s: &mut pprust_hir::State<'_>, node: pprust_hir::AnnNode<'_>) {
        if let pprust_hir::AnnNode::Expr(expr) = node {
            let typeck_results = self.maybe_typeck_results.get().or_else(|| {
                self.tcx
                    .hir()
                    .maybe_body_owned_by(expr.hir_id.owner)
                    .map(|body_id| self.tcx.typeck_body(body_id))
            });

            if let Some(typeck_results) = typeck_results {
                s.s.space();
                s.s.word("as");
                s.s.space();
                s.s.word(typeck_results.expr_ty(expr).to_string());
            }

            s.pclose();
        }
    }
}

fn get_source(input: &Input, sess: &Session) -> (String, FileName) {
    let src_name = input.source_name();
    let src = String::clone(
        sess.source_map()
            .get_source_file(&src_name)
            .expect("get_source_file")
            .src
            .as_ref()
            .expect("src"),
    );
    (src, src_name)
}

fn write_or_print(out: &str, ofile: Option<&Path>, sess: &Session) {
    match ofile {
        None => print!("{}", out),
        Some(p) => {
            if let Err(e) = std::fs::write(p, out) {
                sess.emit_fatal(UnprettyDumpFail {
                    path: p.display().to_string(),
                    err: e.to_string(),
                });
            }
        }
    }
}

pub fn print_after_parsing(
    sess: &Session,
    input: &Input,
    krate: &ast::Crate,
    ppm: PpMode,
    ofile: Option<&Path>,
) {
    let (src, src_name) = get_source(input, sess);

    let out = match ppm {
        Source(s) => {
            // Silently ignores an identified node.
            call_with_pp_support(&s, sess, None, move |annotation| {
                debug!("pretty printing source code {:?}", s);
                let sess = annotation.sess();
                let parse = &sess.parse_sess;
                pprust::print_crate(
                    sess.source_map(),
                    krate,
                    src_name,
                    src,
                    annotation.pp_ann(),
                    false,
                    parse.edition,
                )
            })
        }
        AstTree(PpAstTreeMode::Normal) => {
            debug!("pretty printing AST tree");
            format!("{:#?}", krate)
        }
        _ => unreachable!(),
    };

    write_or_print(&out, ofile, sess);
}

pub fn print_after_hir_lowering<'tcx>(
    tcx: TyCtxt<'tcx>,
    input: &Input,
    krate: &ast::Crate,
    ppm: PpMode,
    ofile: Option<&Path>,
) {
    if ppm.needs_analysis() {
        abort_on_err(print_with_analysis(tcx, ppm, ofile), tcx.sess);
        return;
    }

    let (src, src_name) = get_source(input, tcx.sess);

    let out = match ppm {
        Source(s) => {
            // Silently ignores an identified node.
            call_with_pp_support(&s, tcx.sess, Some(tcx), move |annotation| {
                debug!("pretty printing source code {:?}", s);
                let sess = annotation.sess();
                let parse = &sess.parse_sess;
                pprust::print_crate(
                    sess.source_map(),
                    krate,
                    src_name,
                    src,
                    annotation.pp_ann(),
                    true,
                    parse.edition,
                )
            })
        }

        AstTree(PpAstTreeMode::Expanded) => {
            debug!("pretty-printing expanded AST");
            format!("{:#?}", krate)
        }

        Hir(s) => call_with_pp_support_hir(&s, tcx, move |annotation, hir_map| {
            debug!("pretty printing HIR {:?}", s);
            let sess = annotation.sess();
            let sm = sess.source_map();
            let attrs = |id| hir_map.attrs(id);
            pprust_hir::print_crate(
                sm,
                hir_map.root_module(),
                src_name,
                src,
                &attrs,
                annotation.pp_ann(),
            )
        }),

        HirTree => {
            call_with_pp_support_hir(&PpHirMode::Normal, tcx, move |_annotation, hir_map| {
                debug!("pretty printing HIR tree");
                format!("{:#?}", hir_map.krate())
            })
        }

        _ => unreachable!(),
    };

    write_or_print(&out, ofile, tcx.sess);
}

// In an ideal world, this would be a public function called by the driver after
// analysis is performed. However, we want to call `phase_3_run_analysis_passes`
// with a different callback than the standard driver, so that isn't easy.
// Instead, we call that function ourselves.
fn print_with_analysis(
    tcx: TyCtxt<'_>,
    ppm: PpMode,
    ofile: Option<&Path>,
) -> Result<(), ErrorGuaranteed> {
    tcx.analysis(())?;
    let out = match ppm {
        Mir => {
            let mut out = Vec::new();
            write_mir_pretty(tcx, None, &mut out).unwrap();
            String::from_utf8(out).unwrap()
        }

        MirCFG => {
            let mut out = Vec::new();
            write_mir_graphviz(tcx, None, &mut out).unwrap();
            String::from_utf8(out).unwrap()
        }

        ThirTree => {
            let mut out = String::new();
            abort_on_err(rustc_typeck::check_crate(tcx), tcx.sess);
            debug!("pretty printing THIR tree");
            for did in tcx.hir().body_owners() {
                let _ = writeln!(
                    out,
                    "{:?}:\n{}\n",
                    did,
                    tcx.thir_tree(ty::WithOptConstParam::unknown(did))
                );
            }
            out
        }

        _ => unreachable!(),
    };

    write_or_print(&out, ofile, tcx.sess);

    Ok(())
}
