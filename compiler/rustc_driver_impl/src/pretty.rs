//! The various pretty-printing routines.

use rustc_ast as ast;
use rustc_ast_pretty::pprust as pprust_ast;
use rustc_hir as hir;
use rustc_hir_pretty as pprust_hir;
use rustc_middle::bug;
use rustc_middle::mir::{write_mir_graphviz, write_mir_pretty};
use rustc_middle::ty::{self, TyCtxt};
use rustc_session::config::{OutFileName, PpHirMode, PpMode, PpSourceMode};
use rustc_session::Session;
use rustc_span::symbol::Ident;
use rustc_span::FileName;

use std::cell::Cell;
use std::fmt::Write;

pub use self::PpMode::*;
pub use self::PpSourceMode::*;
use crate::abort_on_err;

struct NoAnn<'hir> {
    tcx: Option<TyCtxt<'hir>>,
}

impl<'hir> pprust_ast::PpAnn for NoAnn<'hir> {}
impl<'hir> pprust_hir::PpAnn for NoAnn<'hir> {
    fn nested(&self, state: &mut pprust_hir::State<'_>, nested: pprust_hir::Nested) {
        if let Some(tcx) = self.tcx {
            pprust_hir::PpAnn::nested(&(&tcx.hir() as &dyn hir::intravisit::Map<'_>), state, nested)
        }
    }
}

struct IdentifiedAnnotation<'hir> {
    tcx: Option<TyCtxt<'hir>>,
}

impl<'hir> pprust_ast::PpAnn for IdentifiedAnnotation<'hir> {
    fn pre(&self, s: &mut pprust_ast::State<'_>, node: pprust_ast::AnnNode<'_>) {
        if let pprust_ast::AnnNode::Expr(_) = node {
            s.popen();
        }
    }

    fn post(&self, s: &mut pprust_ast::State<'_>, node: pprust_ast::AnnNode<'_>) {
        match node {
            pprust_ast::AnnNode::Crate(_)
            | pprust_ast::AnnNode::Ident(_)
            | pprust_ast::AnnNode::Name(_) => {}

            pprust_ast::AnnNode::Item(item) => {
                s.s.space();
                s.synth_comment(item.id.to_string())
            }
            pprust_ast::AnnNode::SubItem(id) => {
                s.s.space();
                s.synth_comment(id.to_string())
            }
            pprust_ast::AnnNode::Block(blk) => {
                s.s.space();
                s.synth_comment(format!("block {}", blk.id))
            }
            pprust_ast::AnnNode::Expr(expr) => {
                s.s.space();
                s.synth_comment(expr.id.to_string());
                s.pclose()
            }
            pprust_ast::AnnNode::Pat(pat) => {
                s.s.space();
                s.synth_comment(format!("pat {}", pat.id));
            }
        }
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

impl<'a> pprust_ast::PpAnn for HygieneAnnotation<'a> {
    fn post(&self, s: &mut pprust_ast::State<'_>, node: pprust_ast::AnnNode<'_>) {
        match node {
            pprust_ast::AnnNode::Ident(&Ident { name, span }) => {
                s.s.space();
                s.synth_comment(format!("{}{:?}", name.as_u32(), span.ctxt()))
            }
            pprust_ast::AnnNode::Name(&name) => {
                s.s.space();
                s.synth_comment(name.as_u32().to_string())
            }
            pprust_ast::AnnNode::Crate(_) => {
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
                    .maybe_body_owned_by(expr.hir_id.owner.def_id)
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

fn get_source(sess: &Session) -> (String, FileName) {
    let src_name = sess.io.input.source_name();
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

fn write_or_print(out: &str, sess: &Session) {
    sess.io.output_file.as_ref().unwrap_or(&OutFileName::Stdout).overwrite(out, sess);
}

// Extra data for pretty-printing, the form of which depends on what kind of
// pretty-printing we are doing.
pub enum PrintExtra<'tcx> {
    AfterParsing { krate: ast::Crate },
    NeedsAstMap { tcx: TyCtxt<'tcx> },
}

impl<'tcx> PrintExtra<'tcx> {
    fn with_krate<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&ast::Crate) -> R,
    {
        match self {
            PrintExtra::AfterParsing { krate, .. } => f(krate),
            PrintExtra::NeedsAstMap { tcx } => f(&tcx.resolver_for_lowering(()).borrow().1),
        }
    }

    fn tcx(&self) -> TyCtxt<'tcx> {
        match self {
            PrintExtra::AfterParsing { .. } => bug!("PrintExtra::tcx"),
            PrintExtra::NeedsAstMap { tcx } => *tcx,
        }
    }
}

pub fn print<'tcx>(sess: &Session, ppm: PpMode, ex: PrintExtra<'tcx>) {
    if ppm.needs_analysis() {
        abort_on_err(ex.tcx().analysis(()), sess);
    }

    let (src, src_name) = get_source(sess);

    let out = match ppm {
        Source(s) => {
            debug!("pretty printing source code {:?}", s);
            let annotation: Box<dyn pprust_ast::PpAnn> = match s {
                Normal => Box::new(NoAnn { tcx: None }),
                Expanded => Box::new(NoAnn { tcx: Some(ex.tcx()) }),
                Identified => Box::new(IdentifiedAnnotation { tcx: None }),
                ExpandedIdentified => Box::new(IdentifiedAnnotation { tcx: Some(ex.tcx()) }),
                ExpandedHygiene => Box::new(HygieneAnnotation { sess }),
            };
            let parse = &sess.parse_sess;
            let is_expanded = ppm.needs_ast_map();
            ex.with_krate(|krate| {
                pprust_ast::print_crate(
                    sess.source_map(),
                    krate,
                    src_name,
                    src,
                    &*annotation,
                    is_expanded,
                    parse.edition,
                    &sess.parse_sess.attr_id_generator,
                )
            })
        }
        AstTree => {
            debug!("pretty printing AST tree");
            ex.with_krate(|krate| format!("{krate:#?}"))
        }
        AstTreeExpanded => {
            debug!("pretty-printing expanded AST");
            format!("{:#?}", ex.tcx().resolver_for_lowering(()).borrow().1)
        }
        Hir(s) => {
            debug!("pretty printing HIR {:?}", s);
            let tcx = ex.tcx();
            let f = |annotation: &dyn pprust_hir::PpAnn| {
                let sm = sess.source_map();
                let hir_map = tcx.hir();
                let attrs = |id| hir_map.attrs(id);
                pprust_hir::print_crate(
                    sm,
                    hir_map.root_module(),
                    src_name,
                    src,
                    &attrs,
                    annotation,
                )
            };
            match s {
                PpHirMode::Normal => {
                    let annotation = NoAnn { tcx: Some(tcx) };
                    f(&annotation)
                }
                PpHirMode::Identified => {
                    let annotation = IdentifiedAnnotation { tcx: Some(tcx) };
                    f(&annotation)
                }
                PpHirMode::Typed => {
                    abort_on_err(tcx.analysis(()), tcx.sess);
                    let annotation = TypedAnnotation { tcx, maybe_typeck_results: Cell::new(None) };
                    tcx.dep_graph.with_ignore(|| f(&annotation))
                }
            }
        }
        HirTree => {
            debug!("pretty printing HIR tree");
            format!("{:#?}", ex.tcx().hir().krate())
        }
        Mir => {
            let mut out = Vec::new();
            write_mir_pretty(ex.tcx(), None, &mut out).unwrap();
            String::from_utf8(out).unwrap()
        }
        MirCFG => {
            let mut out = Vec::new();
            write_mir_graphviz(ex.tcx(), None, &mut out).unwrap();
            String::from_utf8(out).unwrap()
        }
        ThirTree => {
            let tcx = ex.tcx();
            let mut out = String::new();
            abort_on_err(rustc_hir_analysis::check_crate(tcx), tcx.sess);
            debug!("pretty printing THIR tree");
            for did in tcx.hir().body_owners() {
                let _ = writeln!(out, "{:?}:\n{}\n", did, tcx.thir_tree(did));
            }
            out
        }
        ThirFlat => {
            let tcx = ex.tcx();
            let mut out = String::new();
            abort_on_err(rustc_hir_analysis::check_crate(tcx), tcx.sess);
            debug!("pretty printing THIR flat");
            for did in tcx.hir().body_owners() {
                let _ = writeln!(out, "{:?}:\n{}\n", did, tcx.thir_flat(did));
            }
            out
        }
    };

    write_or_print(&out, sess);
}
