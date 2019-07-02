//! The various pretty-printing routines.

use rustc::cfg;
use rustc::cfg::graphviz::LabelledCFG;
use rustc::hir;
use rustc::hir::map as hir_map;
use rustc::hir::map::blocks;
use rustc::hir::print as pprust_hir;
use rustc::hir::def_id::LOCAL_CRATE;
use rustc::session::Session;
use rustc::session::config::Input;
use rustc::ty::{self, TyCtxt};
use rustc::util::common::ErrorReported;
use rustc_interface::util::ReplaceBodyWithLoop;
use rustc_borrowck as borrowck;
use rustc_borrowck::graphviz as borrowck_dot;
use rustc_mir::util::{write_mir_pretty, write_mir_graphviz};

use syntax::ast;
use syntax::mut_visit::MutVisitor;
use syntax::print::{pprust};
use syntax::print::pprust::PrintState;
use syntax_pos::FileName;

use graphviz as dot;

use std::cell::Cell;
use std::fs::File;
use std::io::{self, Write};
use std::option;
use std::path::Path;
use std::str::FromStr;

pub use self::UserIdentifiedItem::*;
pub use self::PpSourceMode::*;
pub use self::PpMode::*;
use self::NodesMatchingUII::*;
use crate::abort_on_err;

use crate::source_name;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PpSourceMode {
    PpmNormal,
    PpmEveryBodyLoops,
    PpmExpanded,
    PpmIdentified,
    PpmExpandedIdentified,
    PpmExpandedHygiene,
    PpmTyped,
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PpFlowGraphMode {
    Default,
    /// Drops the labels from the edges in the flowgraph output. This
    /// is mostly for use in the -Z unpretty flowgraph run-make tests,
    /// since the labels are largely uninteresting in those cases and
    /// have become a pain to maintain.
    UnlabelledEdges,
}
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PpMode {
    PpmSource(PpSourceMode),
    PpmHir(PpSourceMode),
    PpmHirTree(PpSourceMode),
    PpmFlowGraph(PpFlowGraphMode),
    PpmMir,
    PpmMirCFG,
}

impl PpMode {
    pub fn needs_ast_map(&self, opt_uii: &Option<UserIdentifiedItem>) -> bool {
        match *self {
            PpmSource(PpmNormal) |
            PpmSource(PpmEveryBodyLoops) |
            PpmSource(PpmIdentified) => opt_uii.is_some(),

            PpmSource(PpmExpanded) |
            PpmSource(PpmExpandedIdentified) |
            PpmSource(PpmExpandedHygiene) |
            PpmHir(_) |
            PpmHirTree(_) |
            PpmMir |
            PpmMirCFG |
            PpmFlowGraph(_) => true,
            PpmSource(PpmTyped) => panic!("invalid state"),
        }
    }

    pub fn needs_analysis(&self) -> bool {
        match *self {
            PpmMir | PpmMirCFG | PpmFlowGraph(_) => true,
            _ => false,
        }
    }
}

pub fn parse_pretty(sess: &Session,
                    name: &str,
                    extended: bool)
                    -> (PpMode, Option<UserIdentifiedItem>) {
    let mut split = name.splitn(2, '=');
    let first = split.next().unwrap();
    let opt_second = split.next();
    let first = match (first, extended) {
        ("normal", _) => PpmSource(PpmNormal),
        ("identified", _) => PpmSource(PpmIdentified),
        ("everybody_loops", true) => PpmSource(PpmEveryBodyLoops),
        ("expanded", _) => PpmSource(PpmExpanded),
        ("expanded,identified", _) => PpmSource(PpmExpandedIdentified),
        ("expanded,hygiene", _) => PpmSource(PpmExpandedHygiene),
        ("hir", true) => PpmHir(PpmNormal),
        ("hir,identified", true) => PpmHir(PpmIdentified),
        ("hir,typed", true) => PpmHir(PpmTyped),
        ("hir-tree", true) => PpmHirTree(PpmNormal),
        ("mir", true) => PpmMir,
        ("mir-cfg", true) => PpmMirCFG,
        ("flowgraph", true) => PpmFlowGraph(PpFlowGraphMode::Default),
        ("flowgraph,unlabelled", true) => PpmFlowGraph(PpFlowGraphMode::UnlabelledEdges),
        _ => {
            if extended {
                sess.fatal(&format!("argument to `unpretty` must be one of `normal`, \
                                     `expanded`, `flowgraph[,unlabelled]=<nodeid>`, \
                                     `identified`, `expanded,identified`, `everybody_loops`, \
                                     `hir`, `hir,identified`, `hir,typed`, `hir-tree`, \
                                     `mir` or `mir-cfg`; got {}",
                                    name));
            } else {
                sess.fatal(&format!("argument to `pretty` must be one of `normal`, `expanded`, \
                                     `identified`, or `expanded,identified`; got {}",
                                    name));
            }
        }
    };
    let opt_second = opt_second.and_then(|s| s.parse::<UserIdentifiedItem>().ok());
    (first, opt_second)
}



// This slightly awkward construction is to allow for each PpMode to
// choose whether it needs to do analyses (which can consume the
// Session) and then pass through the session (now attached to the
// analysis results) on to the chosen pretty-printer, along with the
// `&PpAnn` object.
//
// Note that since the `&PrinterSupport` is freshly constructed on each
// call, it would not make sense to try to attach the lifetime of `self`
// to the lifetime of the `&PrinterObject`.
//
// (The `use_once_payload` is working around the current lack of once
// functions in the compiler.)

impl PpSourceMode {
    /// Constructs a `PrinterSupport` object and passes it to `f`.
    fn call_with_pp_support<'tcx, A, F>(
        &self,
        sess: &'tcx Session,
        tcx: Option<TyCtxt<'tcx>>,
        f: F,
    ) -> A
    where
        F: FnOnce(&dyn PrinterSupport) -> A,
    {
        match *self {
            PpmNormal | PpmEveryBodyLoops | PpmExpanded => {
                let annotation = NoAnn {
                    sess,
                    tcx,
                };
                f(&annotation)
            }

            PpmIdentified | PpmExpandedIdentified => {
                let annotation = IdentifiedAnnotation {
                    sess,
                    tcx,
                };
                f(&annotation)
            }
            PpmExpandedHygiene => {
                let annotation = HygieneAnnotation {
                    sess,
                };
                f(&annotation)
            }
            _ => panic!("Should use call_with_pp_support_hir"),
        }
    }
    fn call_with_pp_support_hir<A, F>(&self, tcx: TyCtxt<'_>, f: F) -> A
    where
        F: FnOnce(&dyn HirPrinterSupport<'_>, &hir::Crate) -> A,
    {
        match *self {
            PpmNormal => {
                let annotation = NoAnn {
                    sess: tcx.sess,
                    tcx: Some(tcx),
                };
                f(&annotation, tcx.hir().forest.krate())
            }

            PpmIdentified => {
                let annotation = IdentifiedAnnotation {
                    sess: tcx.sess,
                    tcx: Some(tcx),
                };
                f(&annotation, tcx.hir().forest.krate())
            }
            PpmTyped => {
                abort_on_err(tcx.analysis(LOCAL_CRATE), tcx.sess);

                let empty_tables = ty::TypeckTables::empty(None);
                let annotation = TypedAnnotation {
                    tcx,
                    tables: Cell::new(&empty_tables)
                };
                tcx.dep_graph.with_ignore(|| {
                    f(&annotation, tcx.hir().forest.krate())
                })
            }
            _ => panic!("Should use call_with_pp_support"),
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
    /// an object for one of its super-traits.)
    fn pp_ann<'a>(&'a self) -> &'a dyn pprust::PpAnn;
}

trait HirPrinterSupport<'hir>: pprust_hir::PpAnn {
    /// Provides a uniform interface for re-extracting a reference to a
    /// `Session` from a value that now owns it.
    fn sess(&self) -> &Session;

    /// Provides a uniform interface for re-extracting a reference to an
    /// `hir_map::Map` from a value that now owns it.
    fn hir_map<'a>(&'a self) -> Option<&'a hir_map::Map<'hir>>;

    /// Produces the pretty-print annotation object.
    ///
    /// (Rust does not yet support upcasting from a trait object to
    /// an object for one of its super-traits.)
    fn pp_ann<'a>(&'a self) -> &'a dyn pprust_hir::PpAnn;

    /// Computes an user-readable representation of a path, if possible.
    fn node_path(&self, id: hir::HirId) -> Option<String> {
        self.hir_map().and_then(|map| {
            map.def_path_from_hir_id(id)
        }).map(|path| {
            path.data
                .into_iter()
                .map(|elem| elem.data.to_string())
                .collect::<Vec<_>>()
                .join("::")
        })
    }
}

struct NoAnn<'hir> {
    sess: &'hir Session,
    tcx: Option<TyCtxt<'hir>>,
}

impl<'hir> PrinterSupport for NoAnn<'hir> {
    fn sess(&self) -> &Session {
        self.sess
    }

    fn pp_ann<'a>(&'a self) -> &'a dyn pprust::PpAnn {
        self
    }
}

impl<'hir> HirPrinterSupport<'hir> for NoAnn<'hir> {
    fn sess(&self) -> &Session {
        self.sess
    }

    fn hir_map<'a>(&'a self) -> Option<&'a hir_map::Map<'hir>> {
        self.tcx.map(|tcx| tcx.hir())
    }

    fn pp_ann<'a>(&'a self) -> &'a dyn pprust_hir::PpAnn {
        self
    }
}

impl<'hir> pprust::PpAnn for NoAnn<'hir> {}
impl<'hir> pprust_hir::PpAnn for NoAnn<'hir> {
    fn nested(&self, state: &mut pprust_hir::State<'_>, nested: pprust_hir::Nested) {
        if let Some(tcx) = self.tcx {
            pprust_hir::PpAnn::nested(tcx.hir(), state, nested)
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

    fn pp_ann<'a>(&'a self) -> &'a dyn pprust::PpAnn {
        self
    }
}

impl<'hir> pprust::PpAnn for IdentifiedAnnotation<'hir> {
    fn pre(&self, s: &mut pprust::State<'_>, node: pprust::AnnNode<'_>) {
        match node {
            pprust::AnnNode::Expr(_) => s.popen(),
            _ => {}
        }
    }
    fn post(&self, s: &mut pprust::State<'_>, node: pprust::AnnNode<'_>) {
        match node {
            pprust::AnnNode::Ident(_) |
            pprust::AnnNode::Name(_) => {},

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

    fn hir_map<'a>(&'a self) -> Option<&'a hir_map::Map<'hir>> {
        self.tcx.map(|tcx| tcx.hir())
    }

    fn pp_ann<'a>(&'a self) -> &'a dyn pprust_hir::PpAnn {
        self
    }
}

impl<'hir> pprust_hir::PpAnn for IdentifiedAnnotation<'hir> {
    fn nested(&self, state: &mut pprust_hir::State<'_>, nested: pprust_hir::Nested) {
        if let Some(ref tcx) = self.tcx {
            pprust_hir::PpAnn::nested(tcx.hir(), state, nested)
        }
    }
    fn pre(&self, s: &mut pprust_hir::State<'_>, node: pprust_hir::AnnNode<'_>) {
        match node {
            pprust_hir::AnnNode::Expr(_) => s.popen(),
            _ => {}
        }
    }
    fn post(&self, s: &mut pprust_hir::State<'_>, node: pprust_hir::AnnNode<'_>) {
        match node {
            pprust_hir::AnnNode::Name(_) => {},
            pprust_hir::AnnNode::Item(item) => {
                s.s.space();
                s.synth_comment(format!("hir_id: {} hir local_id: {}",
                                        item.hir_id, item.hir_id.local_id.as_u32()))
            }
            pprust_hir::AnnNode::SubItem(id) => {
                s.s.space();
                s.synth_comment(id.to_string())
            }
            pprust_hir::AnnNode::Block(blk) => {
                s.s.space();
                s.synth_comment(format!("block hir_id: {} hir local_id: {}",
                                        blk.hir_id, blk.hir_id.local_id.as_u32()))
            }
            pprust_hir::AnnNode::Expr(expr) => {
                s.s.space();
                s.synth_comment(format!("expr hir_id: {} hir local_id: {}",
                                        expr.hir_id, expr.hir_id.local_id.as_u32()));
                s.pclose()
            }
            pprust_hir::AnnNode::Pat(pat) => {
                s.s.space();
                s.synth_comment(format!("pat hir_id: {} hir local_id: {}",
                                        pat.hir_id, pat.hir_id.local_id.as_u32()))
            }
        }
    }
}

struct HygieneAnnotation<'a> {
    sess: &'a Session
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
            pprust::AnnNode::Ident(&ast::Ident { name, span }) => {
                s.s.space();
                // FIXME #16420: this doesn't display the connections
                // between syntax contexts
                s.synth_comment(format!("{}{:?}", name.as_u32(), span.ctxt()))
            }
            pprust::AnnNode::Name(&name) => {
                s.s.space();
                s.synth_comment(name.as_u32().to_string())
            }
            _ => {}
        }
    }
}

struct TypedAnnotation<'a, 'tcx> {
    tcx: TyCtxt<'tcx>,
    tables: Cell<&'a ty::TypeckTables<'tcx>>,
}

impl<'b, 'tcx> HirPrinterSupport<'tcx> for TypedAnnotation<'b, 'tcx> {
    fn sess(&self) -> &Session {
        &self.tcx.sess
    }

    fn hir_map<'a>(&'a self) -> Option<&'a hir_map::Map<'tcx>> {
        Some(&self.tcx.hir())
    }

    fn pp_ann<'a>(&'a self) -> &'a dyn pprust_hir::PpAnn {
        self
    }

    fn node_path(&self, id: hir::HirId) -> Option<String> {
        Some(self.tcx.def_path_str(self.tcx.hir().local_def_id_from_hir_id(id)))
    }
}

impl<'a, 'tcx> pprust_hir::PpAnn for TypedAnnotation<'a, 'tcx> {
    fn nested(&self, state: &mut pprust_hir::State<'_>, nested: pprust_hir::Nested) {
        let old_tables = self.tables.get();
        if let pprust_hir::Nested::Body(id) = nested {
            self.tables.set(self.tcx.body_tables(id));
        }
        pprust_hir::PpAnn::nested(self.tcx.hir(), state, nested);
        self.tables.set(old_tables);
    }
    fn pre(&self, s: &mut pprust_hir::State<'_>, node: pprust_hir::AnnNode<'_>) {
        match node {
            pprust_hir::AnnNode::Expr(_) => s.popen(),
            _ => {}
        }
    }
    fn post(&self, s: &mut pprust_hir::State<'_>, node: pprust_hir::AnnNode<'_>) {
        match node {
            pprust_hir::AnnNode::Expr(expr) => {
                s.s.space();
                s.s.word("as");
                s.s.space();
                s.s.word(self.tables.get().expr_ty(expr).to_string());
                s.pclose();
            }
            _ => {},
        }
    }
}

fn gather_flowgraph_variants(sess: &Session) -> Vec<borrowck_dot::Variant> {
    let print_loans = sess.opts.debugging_opts.flowgraph_print_loans;
    let print_moves = sess.opts.debugging_opts.flowgraph_print_moves;
    let print_assigns = sess.opts.debugging_opts.flowgraph_print_assigns;
    let print_all = sess.opts.debugging_opts.flowgraph_print_all;
    let mut variants = Vec::new();
    if print_all || print_loans {
        variants.push(borrowck_dot::Loans);
    }
    if print_all || print_moves {
        variants.push(borrowck_dot::Moves);
    }
    if print_all || print_assigns {
        variants.push(borrowck_dot::Assigns);
    }
    variants
}

#[derive(Clone, Debug)]
pub enum UserIdentifiedItem {
    ItemViaNode(ast::NodeId),
    ItemViaPath(Vec<String>),
}

impl FromStr for UserIdentifiedItem {
    type Err = ();
    fn from_str(s: &str) -> Result<UserIdentifiedItem, ()> {
        Ok(s.parse()
            .map(ast::NodeId::from_u32)
            .map(ItemViaNode)
            .unwrap_or_else(|_| ItemViaPath(s.split("::").map(|s| s.to_string()).collect())))
    }
}

enum NodesMatchingUII<'a> {
    NodesMatchingDirect(option::IntoIter<ast::NodeId>),
    NodesMatchingSuffix(Box<dyn Iterator<Item = ast::NodeId> + 'a>),
}

impl<'a> Iterator for NodesMatchingUII<'a> {
    type Item = ast::NodeId;

    fn next(&mut self) -> Option<ast::NodeId> {
        match self {
            &mut NodesMatchingDirect(ref mut iter) => iter.next(),
            &mut NodesMatchingSuffix(ref mut iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            &NodesMatchingDirect(ref iter) => iter.size_hint(),
            &NodesMatchingSuffix(ref iter) => iter.size_hint(),
        }
    }
}

impl UserIdentifiedItem {
    fn reconstructed_input(&self) -> String {
        match *self {
            ItemViaNode(node_id) => node_id.to_string(),
            ItemViaPath(ref parts) => parts.join("::"),
        }
    }

    fn all_matching_node_ids<'a, 'hir>(&'a self,
                                       map: &'a hir_map::Map<'hir>)
                                       -> NodesMatchingUII<'a> {
        match *self {
            ItemViaNode(node_id) => NodesMatchingDirect(Some(node_id).into_iter()),
            ItemViaPath(ref parts) => {
                NodesMatchingSuffix(Box::new(map.nodes_matching_suffix(&parts)))
            }
        }
    }

    fn to_one_node_id(self,
                      user_option: &str,
                      sess: &Session,
                      map: &hir_map::Map<'_>)
                      -> ast::NodeId {
        let fail_because = |is_wrong_because| -> ast::NodeId {
            let message = format!("{} needs NodeId (int) or unique path suffix (b::c::d); got \
                                   {}, which {}",
                                  user_option,
                                  self.reconstructed_input(),
                                  is_wrong_because);
            sess.fatal(&message)
        };

        let mut saw_node = ast::DUMMY_NODE_ID;
        let mut seen = 0;
        for node in self.all_matching_node_ids(map) {
            saw_node = node;
            seen += 1;
            if seen > 1 {
                fail_because("does not resolve uniquely");
            }
        }
        if seen == 0 {
            fail_because("does not resolve to any item");
        }

        assert!(seen == 1);
        return saw_node;
    }
}

fn print_flowgraph<'tcx, W: Write>(
    variants: Vec<borrowck_dot::Variant>,
    tcx: TyCtxt<'tcx>,
    code: blocks::Code<'tcx>,
    mode: PpFlowGraphMode,
    mut out: W,
) -> io::Result<()> {
    let body_id = match code {
        blocks::Code::Expr(expr) => {
            // Find the function this expression is from.
            let mut hir_id = expr.hir_id;
            loop {
                let node = tcx.hir().get(hir_id);
                if let Some(n) = hir::map::blocks::FnLikeNode::from_node(node) {
                    break n.body();
                }
                let parent = tcx.hir().get_parent_node(hir_id);
                assert_ne!(hir_id, parent);
                hir_id = parent;
            }
        }
        blocks::Code::FnLike(fn_like) => fn_like.body(),
    };
    let body = tcx.hir().body(body_id);
    let cfg = cfg::CFG::new(tcx, &body);
    let labelled_edges = mode != PpFlowGraphMode::UnlabelledEdges;
    let hir_id = code.id();
    // We have to disassemble the hir_id because name must be ASCII
    // alphanumeric. This does not appear in the rendered graph, so it does not
    // have to be user friendly.
    let name = format!(
        "hir_id_{}_{}",
        hir_id.owner.index(),
        hir_id.local_id.index(),
    );
    let lcfg = LabelledCFG {
        tcx,
        cfg: &cfg,
        name,
        labelled_edges,
    };

    match code {
        _ if variants.is_empty() => {
            let r = dot::render(&lcfg, &mut out);
            return expand_err_details(r);
        }
        blocks::Code::Expr(_) => {
            tcx.sess.err("--pretty flowgraph with -Z flowgraph-print annotations requires \
                          fn-like node id.");
            return Ok(());
        }
        blocks::Code::FnLike(fn_like) => {
            let (bccx, analysis_data) =
                borrowck::build_borrowck_dataflow_data_for_fn(tcx, fn_like.body(), &cfg);

            let lcfg = borrowck_dot::DataflowLabeller {
                inner: lcfg,
                variants,
                borrowck_ctxt: &bccx,
                analysis_data: &analysis_data,
            };
            let r = dot::render(&lcfg, &mut out);
            return expand_err_details(r);
        }
    }

    fn expand_err_details(r: io::Result<()>) -> io::Result<()> {
        r.map_err(|ioerr| {
            io::Error::new(io::ErrorKind::Other,
                           format!("graphviz::render failed: {}", ioerr))
        })
    }
}

pub fn visit_crate(sess: &Session, krate: &mut ast::Crate, ppm: PpMode) {
    if let PpmSource(PpmEveryBodyLoops) = ppm {
        ReplaceBodyWithLoop::new(sess).visit_crate(krate);
    }
}

fn get_source(input: &Input, sess: &Session) -> (Vec<u8>, FileName) {
    let src_name = source_name(input);
    let src = sess.source_map()
        .get_source_file(&src_name)
        .unwrap()
        .src
        .as_ref()
        .unwrap()
        .as_bytes()
        .to_vec();
    (src, src_name)
}

fn write_output(out: Vec<u8>, ofile: Option<&Path>) {
    match ofile {
        None => print!("{}", String::from_utf8(out).unwrap()),
        Some(p) => {
            match File::create(p) {
                Ok(mut w) => w.write_all(&out).unwrap(),
                Err(e) => panic!("print-print failed to open {} due to {}", p.display(), e),
            }
        }
    }
}

pub fn print_after_parsing(sess: &Session,
                           input: &Input,
                           krate: &ast::Crate,
                           ppm: PpMode,
                           ofile: Option<&Path>) {
    let (src, src_name) = get_source(input, sess);

    let mut rdr = &*src;
    let mut out = String::new();

    if let PpmSource(s) = ppm {
        // Silently ignores an identified node.
        let out = &mut out;
        s.call_with_pp_support(sess, None, move |annotation| {
            debug!("pretty printing source code {:?}", s);
            let sess = annotation.sess();
            pprust::print_crate(sess.source_map(),
                                &sess.parse_sess,
                                krate,
                                src_name,
                                &mut rdr,
                                out,
                                annotation.pp_ann(),
                                false)
        })
    } else {
        unreachable!();
    };

    write_output(out.into_bytes(), ofile);
}

pub fn print_after_hir_lowering<'tcx>(
    tcx: TyCtxt<'tcx>,
    input: &Input,
    krate: &ast::Crate,
    ppm: PpMode,
    opt_uii: Option<UserIdentifiedItem>,
    ofile: Option<&Path>,
) {
    if ppm.needs_analysis() {
        abort_on_err(print_with_analysis(
            tcx,
            ppm,
            opt_uii,
            ofile
        ), tcx.sess);
        return;
    }

    let (src, src_name) = get_source(input, tcx.sess);

    let mut rdr = &src[..];
    let mut out = String::new();

    match (ppm, opt_uii) {
            (PpmSource(s), _) => {
                // Silently ignores an identified node.
                let out = &mut out;
                s.call_with_pp_support(tcx.sess, Some(tcx), move |annotation| {
                    debug!("pretty printing source code {:?}", s);
                    let sess = annotation.sess();
                    pprust::print_crate(sess.source_map(),
                                        &sess.parse_sess,
                                        krate,
                                        src_name,
                                        &mut rdr,
                                        out,
                                        annotation.pp_ann(),
                                        true)
                })
            }

            (PpmHir(s), None) => {
                let out = &mut out;
                s.call_with_pp_support_hir(tcx, move |annotation, krate| {
                    debug!("pretty printing source code {:?}", s);
                    let sess = annotation.sess();
                    pprust_hir::print_crate(sess.source_map(),
                                            &sess.parse_sess,
                                            krate,
                                            src_name,
                                            &mut rdr,
                                            out,
                                            annotation.pp_ann())
                })
            }

            (PpmHirTree(s), None) => {
                let out = &mut out;
                s.call_with_pp_support_hir(tcx, move |_annotation, krate| {
                    debug!("pretty printing source code {:?}", s);
                    *out = format!("{:#?}", krate);
                });
            }

            (PpmHir(s), Some(uii)) => {
                let out = &mut out;
                s.call_with_pp_support_hir(tcx, move |annotation, _| {
                    debug!("pretty printing source code {:?}", s);
                    let sess = annotation.sess();
                    let hir_map = annotation.hir_map().expect("-Z unpretty missing HIR map");
                    let mut pp_state = pprust_hir::State::new_from_input(sess.source_map(),
                                                                         &sess.parse_sess,
                                                                         src_name,
                                                                         &mut rdr,
                                                                         out,
                                                                         annotation.pp_ann());
                    for node_id in uii.all_matching_node_ids(hir_map) {
                        let hir_id = tcx.hir().node_to_hir_id(node_id);
                        let node = hir_map.get(hir_id);
                        pp_state.print_node(node);
                        pp_state.s.space();
                        let path = annotation.node_path(hir_id)
                            .expect("-Z unpretty missing node paths");
                        pp_state.synth_comment(path);
                        pp_state.s.hardbreak();
                    }
                    pp_state.s.eof();
                })
            }

            (PpmHirTree(s), Some(uii)) => {
                let out = &mut out;
                s.call_with_pp_support_hir(tcx, move |_annotation, _krate| {
                    debug!("pretty printing source code {:?}", s);
                    for node_id in uii.all_matching_node_ids(tcx.hir()) {
                        let hir_id = tcx.hir().node_to_hir_id(node_id);
                        let node = tcx.hir().get(hir_id);
                        out.push_str(&format!("{:#?}", node));
                    }
                })
            }

            _ => unreachable!(),
        }

    write_output(out.into_bytes(), ofile);
}

// In an ideal world, this would be a public function called by the driver after
// analysis is performed. However, we want to call `phase_3_run_analysis_passes`
// with a different callback than the standard driver, so that isn't easy.
// Instead, we call that function ourselves.
fn print_with_analysis(
    tcx: TyCtxt<'_>,
    ppm: PpMode,
    uii: Option<UserIdentifiedItem>,
    ofile: Option<&Path>,
) -> Result<(), ErrorReported> {
    let nodeid = if let Some(uii) = uii {
        debug!("pretty printing for {:?}", uii);
        Some(uii.to_one_node_id("-Z unpretty", tcx.sess, tcx.hir()))
    } else {
        debug!("pretty printing for whole crate");
        None
    };

    let mut out = Vec::new();

    tcx.analysis(LOCAL_CRATE)?;

    let mut print = || match ppm {
        PpmMir | PpmMirCFG => {
            if let Some(nodeid) = nodeid {
                let def_id = tcx.hir().local_def_id(nodeid);
                match ppm {
                    PpmMir => write_mir_pretty(tcx, Some(def_id), &mut out),
                    PpmMirCFG => write_mir_graphviz(tcx, Some(def_id), &mut out),
                    _ => unreachable!(),
                }?;
            } else {
                match ppm {
                    PpmMir => write_mir_pretty(tcx, None, &mut out),
                    PpmMirCFG => write_mir_graphviz(tcx, None, &mut out),
                    _ => unreachable!(),
                }?;
            }
            Ok(())
        }
        PpmFlowGraph(mode) => {
            let nodeid =
                nodeid.expect("`pretty flowgraph=..` needs NodeId (int) or unique path \
                                suffix (b::c::d)");
            let hir_id = tcx.hir().node_to_hir_id(nodeid);
            let node = tcx.hir().find(hir_id).unwrap_or_else(|| {
                tcx.sess.fatal(&format!("--pretty flowgraph couldn't find id: {}", nodeid))
            });

            match blocks::Code::from_node(&tcx.hir(), hir_id) {
                Some(code) => {
                    let variants = gather_flowgraph_variants(tcx.sess);

                    let out: &mut dyn Write = &mut out;

                    print_flowgraph(variants, tcx, code, mode, out)
                }
                None => {
                    let message = format!("--pretty=flowgraph needs block, fn, or method; \
                                            got {:?}",
                                            node);

                    let hir_id = tcx.hir().node_to_hir_id(nodeid);
                    tcx.sess.span_fatal(tcx.hir().span(hir_id), &message)
                }
            }
        }
        _ => unreachable!(),
    };

    print().unwrap();

    write_output(out, ofile);

    Ok(())
}
