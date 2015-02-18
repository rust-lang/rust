// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! The various pretty print routines.

pub use self::UserIdentifiedItem::*;
pub use self::PpSourceMode::*;
pub use self::PpMode::*;
use self::NodesMatchingUII::*;

use rustc_trans::back::link;

use driver;

use rustc::middle::ty;
use rustc::middle::cfg;
use rustc::middle::cfg::graphviz::LabelledCFG;
use rustc::session::Session;
use rustc::session::config::Input;
use rustc::util::ppaux;
use rustc_borrowck as borrowck;
use rustc_borrowck::graphviz as borrowck_dot;
use rustc_resolve as resolve;

use syntax::ast;
use syntax::ast_map::{self, blocks, NodePrinter};
use syntax::codemap;
use syntax::fold::{self, Folder};
use syntax::print::{pp, pprust};
use syntax::ptr::P;

use graphviz as dot;

use std::old_io::{self, MemReader};
use std::option;
use std::str::FromStr;

#[derive(Copy, PartialEq, Debug)]
pub enum PpSourceMode {
    PpmNormal,
    PpmEveryBodyLoops,
    PpmExpanded,
    PpmTyped,
    PpmIdentified,
    PpmExpandedIdentified,
    PpmExpandedHygiene,
}


#[derive(Copy, PartialEq, Debug)]
pub enum PpFlowGraphMode {
    Default,
    /// Drops the labels from the edges in the flowgraph output. This
    /// is mostly for use in the --xpretty flowgraph run-make tests,
    /// since the labels are largely uninteresting in those cases and
    /// have become a pain to maintain.
    UnlabelledEdges,
}
#[derive(Copy, PartialEq, Debug)]
pub enum PpMode {
    PpmSource(PpSourceMode),
    PpmFlowGraph(PpFlowGraphMode),
}

pub fn parse_pretty(sess: &Session,
                    name: &str,
                    extended: bool) -> (PpMode, Option<UserIdentifiedItem>) {
    let mut split = name.splitn(1, '=');
    let first = split.next().unwrap();
    let opt_second = split.next();
    let first = match (first, extended) {
        ("normal", _)       => PpmSource(PpmNormal),
        ("everybody_loops", true) => PpmSource(PpmEveryBodyLoops),
        ("expanded", _)     => PpmSource(PpmExpanded),
        ("typed", _)        => PpmSource(PpmTyped),
        ("expanded,identified", _) => PpmSource(PpmExpandedIdentified),
        ("expanded,hygiene", _) => PpmSource(PpmExpandedHygiene),
        ("identified", _)   => PpmSource(PpmIdentified),
        ("flowgraph", true)    => PpmFlowGraph(PpFlowGraphMode::Default),
        ("flowgraph,unlabelled", true)    => PpmFlowGraph(PpFlowGraphMode::UnlabelledEdges),
        _ => {
            if extended {
                sess.fatal(&format!(
                    "argument to `xpretty` must be one of `normal`, \
                     `expanded`, `flowgraph[,unlabelled]=<nodeid>`, `typed`, `identified`, \
                     `expanded,identified`, or `everybody_loops`; got {}", name));
            } else {
                sess.fatal(&format!(
                    "argument to `pretty` must be one of `normal`, \
                     `expanded`, `typed`, `identified`, \
                     or `expanded,identified`; got {}", name));
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
    fn call_with_pp_support<'tcx, A, B, F>(&self,
                                           sess: Session,
                                           ast_map: Option<ast_map::Map<'tcx>>,
                                           arenas: &'tcx ty::CtxtArenas<'tcx>,
                                           id: String,
                                           payload: B,
                                           f: F) -> A where
        F: FnOnce(&PrinterSupport, B) -> A,
    {
        match *self {
            PpmNormal | PpmEveryBodyLoops | PpmExpanded => {
                let annotation = NoAnn { sess: sess, ast_map: ast_map };
                f(&annotation, payload)
            }

            PpmIdentified | PpmExpandedIdentified => {
                let annotation = IdentifiedAnnotation { sess: sess, ast_map: ast_map };
                f(&annotation, payload)
            }
            PpmExpandedHygiene => {
                let annotation = HygieneAnnotation { sess: sess, ast_map: ast_map };
                f(&annotation, payload)
            }
            PpmTyped => {
                let ast_map = ast_map.expect("--pretty=typed missing ast_map");
                let analysis = driver::phase_3_run_analysis_passes(sess,
                                                                   ast_map,
                                                                   arenas,
                                                                   id,
                                                                   resolve::MakeGlobMap::No);
                let annotation = TypedAnnotation { analysis: analysis };
                f(&annotation, payload)
            }
        }
    }
}

trait PrinterSupport<'ast>: pprust::PpAnn {
    /// Provides a uniform interface for re-extracting a reference to a
    /// `Session` from a value that now owns it.
    fn sess<'a>(&'a self) -> &'a Session;

    /// Provides a uniform interface for re-extracting a reference to an
    /// `ast_map::Map` from a value that now owns it.
    fn ast_map<'a>(&'a self) -> Option<&'a ast_map::Map<'ast>>;

    /// Produces the pretty-print annotation object.
    ///
    /// (Rust does not yet support upcasting from a trait object to
    /// an object for one of its super-traits.)
    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn;
}

struct NoAnn<'ast> {
    sess: Session,
    ast_map: Option<ast_map::Map<'ast>>
}

impl<'ast> PrinterSupport<'ast> for NoAnn<'ast> {
    fn sess<'a>(&'a self) -> &'a Session { &self.sess }

    fn ast_map<'a>(&'a self) -> Option<&'a ast_map::Map<'ast>> {
        self.ast_map.as_ref()
    }

    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn { self }
}

impl<'ast> pprust::PpAnn for NoAnn<'ast> {}

struct IdentifiedAnnotation<'ast> {
    sess: Session,
    ast_map: Option<ast_map::Map<'ast>>,
}

impl<'ast> PrinterSupport<'ast> for IdentifiedAnnotation<'ast> {
    fn sess<'a>(&'a self) -> &'a Session { &self.sess }

    fn ast_map<'a>(&'a self) -> Option<&'a ast_map::Map<'ast>> {
        self.ast_map.as_ref()
    }

    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn { self }
}

impl<'ast> pprust::PpAnn for IdentifiedAnnotation<'ast> {
    fn pre(&self,
           s: &mut pprust::State,
           node: pprust::AnnNode) -> old_io::IoResult<()> {
        match node {
            pprust::NodeExpr(_) => s.popen(),
            _ => Ok(())
        }
    }
    fn post(&self,
            s: &mut pprust::State,
            node: pprust::AnnNode) -> old_io::IoResult<()> {
        match node {
            pprust::NodeIdent(_) | pprust::NodeName(_) => Ok(()),

            pprust::NodeItem(item) => {
                try!(pp::space(&mut s.s));
                s.synth_comment(item.id.to_string())
            }
            pprust::NodeBlock(blk) => {
                try!(pp::space(&mut s.s));
                s.synth_comment(format!("block {}", blk.id))
            }
            pprust::NodeExpr(expr) => {
                try!(pp::space(&mut s.s));
                try!(s.synth_comment(expr.id.to_string()));
                s.pclose()
            }
            pprust::NodePat(pat) => {
                try!(pp::space(&mut s.s));
                s.synth_comment(format!("pat {}", pat.id))
            }
        }
    }
}

struct HygieneAnnotation<'ast> {
    sess: Session,
    ast_map: Option<ast_map::Map<'ast>>,
}

impl<'ast> PrinterSupport<'ast> for HygieneAnnotation<'ast> {
    fn sess<'a>(&'a self) -> &'a Session { &self.sess }

    fn ast_map<'a>(&'a self) -> Option<&'a ast_map::Map<'ast>> {
        self.ast_map.as_ref()
    }

    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn { self }
}

impl<'ast> pprust::PpAnn for HygieneAnnotation<'ast> {
    fn post(&self,
            s: &mut pprust::State,
            node: pprust::AnnNode) -> old_io::IoResult<()> {
        match node {
            pprust::NodeIdent(&ast::Ident { name: ast::Name(nm), ctxt }) => {
                try!(pp::space(&mut s.s));
                // FIXME #16420: this doesn't display the connections
                // between syntax contexts
                s.synth_comment(format!("{}#{}", nm, ctxt))
            }
            pprust::NodeName(&ast::Name(nm)) => {
                try!(pp::space(&mut s.s));
                s.synth_comment(nm.to_string())
            }
            _ => Ok(())
        }
    }
}


struct TypedAnnotation<'tcx> {
    analysis: ty::CrateAnalysis<'tcx>,
}

impl<'tcx> PrinterSupport<'tcx> for TypedAnnotation<'tcx> {
    fn sess<'a>(&'a self) -> &'a Session { &self.analysis.ty_cx.sess }

    fn ast_map<'a>(&'a self) -> Option<&'a ast_map::Map<'tcx>> {
        Some(&self.analysis.ty_cx.map)
    }

    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn { self }
}

impl<'tcx> pprust::PpAnn for TypedAnnotation<'tcx> {
    fn pre(&self,
           s: &mut pprust::State,
           node: pprust::AnnNode) -> old_io::IoResult<()> {
        match node {
            pprust::NodeExpr(_) => s.popen(),
            _ => Ok(())
        }
    }
    fn post(&self,
            s: &mut pprust::State,
            node: pprust::AnnNode) -> old_io::IoResult<()> {
        let tcx = &self.analysis.ty_cx;
        match node {
            pprust::NodeExpr(expr) => {
                try!(pp::space(&mut s.s));
                try!(pp::word(&mut s.s, "as"));
                try!(pp::space(&mut s.s));
                try!(pp::word(&mut s.s,
                              &ppaux::ty_to_string(
                                  tcx,
                                  ty::expr_ty(tcx, expr))[]));
                s.pclose()
            }
            _ => Ok(())
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
        Ok(s.parse().map(ItemViaNode).unwrap_or_else(|_| {
            ItemViaPath(s.split_str("::").map(|s| s.to_string()).collect())
        }))
    }
}

enum NodesMatchingUII<'a, 'ast: 'a> {
    NodesMatchingDirect(option::IntoIter<ast::NodeId>),
    NodesMatchingSuffix(ast_map::NodesMatchingSuffix<'a, 'ast>),
}

impl<'a, 'ast> Iterator for NodesMatchingUII<'a, 'ast> {
    type Item = ast::NodeId;

    fn next(&mut self) -> Option<ast::NodeId> {
        match self {
            &mut NodesMatchingDirect(ref mut iter) => iter.next(),
            &mut NodesMatchingSuffix(ref mut iter) => iter.next(),
        }
    }
}

impl UserIdentifiedItem {
    fn reconstructed_input(&self) -> String {
        match *self {
            ItemViaNode(node_id) => node_id.to_string(),
            ItemViaPath(ref parts) => parts.connect("::"),
        }
    }

    fn all_matching_node_ids<'a, 'ast>(&'a self, map: &'a ast_map::Map<'ast>)
                                       -> NodesMatchingUII<'a, 'ast> {
        match *self {
            ItemViaNode(node_id) =>
                NodesMatchingDirect(Some(node_id).into_iter()),
            ItemViaPath(ref parts) =>
                NodesMatchingSuffix(map.nodes_matching_suffix(&parts[..])),
        }
    }

    fn to_one_node_id(self, user_option: &str, sess: &Session, map: &ast_map::Map) -> ast::NodeId {
        let fail_because = |is_wrong_because| -> ast::NodeId {
            let message =
                format!("{} needs NodeId (int) or unique \
                         path suffix (b::c::d); got {}, which {}",
                        user_option,
                        self.reconstructed_input(),
                        is_wrong_because);
            sess.fatal(&message[..])
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

fn needs_ast_map(ppm: &PpMode, opt_uii: &Option<UserIdentifiedItem>) -> bool {
    match *ppm {
        PpmSource(PpmNormal) |
        PpmSource(PpmEveryBodyLoops) |
        PpmSource(PpmIdentified) => opt_uii.is_some(),

        PpmSource(PpmExpanded) |
        PpmSource(PpmExpandedIdentified) |
        PpmSource(PpmExpandedHygiene) |
        PpmSource(PpmTyped) |
        PpmFlowGraph(_) => true
    }
}

fn needs_expansion(ppm: &PpMode) -> bool {
    match *ppm {
        PpmSource(PpmNormal) |
        PpmSource(PpmEveryBodyLoops) |
        PpmSource(PpmIdentified) => false,

        PpmSource(PpmExpanded) |
        PpmSource(PpmExpandedIdentified) |
        PpmSource(PpmExpandedHygiene) |
        PpmSource(PpmTyped) |
        PpmFlowGraph(_) => true
    }
}

struct ReplaceBodyWithLoop {
    within_static_or_const: bool,
}

impl ReplaceBodyWithLoop {
    fn new() -> ReplaceBodyWithLoop {
        ReplaceBodyWithLoop { within_static_or_const: false }
    }
}

impl fold::Folder for ReplaceBodyWithLoop {
    fn fold_item_underscore(&mut self, i: ast::Item_) -> ast::Item_ {
        match i {
            ast::ItemStatic(..) | ast::ItemConst(..) => {
                self.within_static_or_const = true;
                let ret = fold::noop_fold_item_underscore(i, self);
                self.within_static_or_const = false;
                return ret;
            }
            _ => {
                fold::noop_fold_item_underscore(i, self)
            }
        }
    }


    fn fold_block(&mut self, b: P<ast::Block>) -> P<ast::Block> {
        fn expr_to_block(rules: ast::BlockCheckMode,
                         e: Option<P<ast::Expr>>) -> P<ast::Block> {
            P(ast::Block {
                expr: e,
                stmts: vec![], rules: rules,
                id: ast::DUMMY_NODE_ID, span: codemap::DUMMY_SP,
            })
        }

        if !self.within_static_or_const {

            let empty_block = expr_to_block(ast::DefaultBlock, None);
            let loop_expr = P(ast::Expr {
                node: ast::ExprLoop(empty_block, None),
                id: ast::DUMMY_NODE_ID, span: codemap::DUMMY_SP
            });

            expr_to_block(b.rules, Some(loop_expr))

        } else {
            fold::noop_fold_block(b, self)
        }
    }

    // in general the pretty printer processes unexpanded code, so
    // we override the default `fold_mac` method which panics.
    fn fold_mac(&mut self, mac: ast::Mac) -> ast::Mac {
        fold::noop_fold_mac(mac, self)
    }
}

pub fn pretty_print_input(sess: Session,
                          cfg: ast::CrateConfig,
                          input: &Input,
                          ppm: PpMode,
                          opt_uii: Option<UserIdentifiedItem>,
                          ofile: Option<Path>) {
    let krate = driver::phase_1_parse_input(&sess, cfg, input);

    let krate = if let PpmSource(PpmEveryBodyLoops) = ppm {
        let mut fold = ReplaceBodyWithLoop::new();
        fold.fold_crate(krate)
    } else {
        krate
    };

    let id = link::find_crate_name(Some(&sess), &krate.attrs, input);

    let is_expanded = needs_expansion(&ppm);
    let compute_ast_map = needs_ast_map(&ppm, &opt_uii);
    let krate = if compute_ast_map {
        match driver::phase_2_configure_and_expand(&sess, krate, &id[..], None) {
            None => return,
            Some(k) => k
        }
    } else {
        krate
    };

    let mut forest = ast_map::Forest::new(krate);
    let arenas = ty::CtxtArenas::new();

    let (krate, ast_map) = if compute_ast_map {
        let map = driver::assign_node_ids_and_map(&sess, &mut forest);
        (map.krate(), Some(map))
    } else {
        (forest.krate(), None)
    };

    let src_name = driver::source_name(input);
    let src = sess.codemap().get_filemap(&src_name[..])
                            .src.as_bytes().to_vec();
    let mut rdr = MemReader::new(src);

    let out = match ofile {
        None => box old_io::stdout() as Box<Writer+'static>,
        Some(p) => {
            let r = old_io::File::create(&p);
            match r {
                Ok(w) => box w as Box<Writer+'static>,
                Err(e) => panic!("print-print failed to open {} due to {}",
                                p.display(), e),
            }
        }
    };

    match (ppm, opt_uii) {
        (PpmSource(s), None) =>
            s.call_with_pp_support(
                sess, ast_map, &arenas, id, out, |annotation, out| {
                    debug!("pretty printing source code {:?}", s);
                    let sess = annotation.sess();
                    pprust::print_crate(sess.codemap(),
                                        sess.diagnostic(),
                                        krate,
                                        src_name.to_string(),
                                        &mut rdr,
                                        out,
                                        annotation.pp_ann(),
                                        is_expanded)
                }),

        (PpmSource(s), Some(uii)) =>
            s.call_with_pp_support(
                sess, ast_map, &arenas, id, (out,uii), |annotation, (out,uii)| {
                    debug!("pretty printing source code {:?}", s);
                    let sess = annotation.sess();
                    let ast_map = annotation.ast_map()
                        .expect("--pretty missing ast_map");
                    let mut pp_state =
                        pprust::State::new_from_input(sess.codemap(),
                                                      sess.diagnostic(),
                                                      src_name.to_string(),
                                                      &mut rdr,
                                                      out,
                                                      annotation.pp_ann(),
                                                      is_expanded);
                    for node_id in uii.all_matching_node_ids(ast_map) {
                        let node = ast_map.get(node_id);
                        try!(pp_state.print_node(&node));
                        try!(pp::space(&mut pp_state.s));
                        try!(pp_state.synth_comment(ast_map.path_to_string(node_id)));
                        try!(pp::hardbreak(&mut pp_state.s));
                    }
                    pp::eof(&mut pp_state.s)
                }),

        (PpmFlowGraph(mode), opt_uii) => {
            debug!("pretty printing flow graph for {:?}", opt_uii);
            let uii = opt_uii.unwrap_or_else(|| {
                sess.fatal(&format!("`pretty flowgraph=..` needs NodeId (int) or
                                     unique path suffix (b::c::d)")[])

            });
            let ast_map = ast_map.expect("--pretty flowgraph missing ast_map");
            let nodeid = uii.to_one_node_id("--pretty", &sess, &ast_map);

            let node = ast_map.find(nodeid).unwrap_or_else(|| {
                sess.fatal(&format!("--pretty flowgraph couldn't find id: {}",
                                   nodeid)[])
            });

            let code = blocks::Code::from_node(node);
            match code {
                Some(code) => {
                    let variants = gather_flowgraph_variants(&sess);
                    let analysis = driver::phase_3_run_analysis_passes(sess,
                                                                       ast_map,
                                                                       &arenas,
                                                                       id,
                                                                       resolve::MakeGlobMap::No);
                    print_flowgraph(variants, analysis, code, mode, out)
                }
                None => {
                    let message = format!("--pretty=flowgraph needs \
                                           block, fn, or method; got {:?}",
                                          node);

                    // point to what was found, if there's an
                    // accessible span.
                    match ast_map.opt_span(nodeid) {
                        Some(sp) => sess.span_fatal(sp, &message[..]),
                        None => sess.fatal(&message[..])
                    }
                }
            }
        }
    }.unwrap()
}

fn print_flowgraph<W:old_io::Writer>(variants: Vec<borrowck_dot::Variant>,
                                 analysis: ty::CrateAnalysis,
                                 code: blocks::Code,
                                 mode: PpFlowGraphMode,
                                 mut out: W) -> old_io::IoResult<()> {
    let ty_cx = &analysis.ty_cx;
    let cfg = match code {
        blocks::BlockCode(block) => cfg::CFG::new(ty_cx, &*block),
        blocks::FnLikeCode(fn_like) => cfg::CFG::new(ty_cx, &*fn_like.body()),
    };
    let labelled_edges = mode != PpFlowGraphMode::UnlabelledEdges;
    let lcfg = LabelledCFG {
        ast_map: &ty_cx.map,
        cfg: &cfg,
        name: format!("node_{}", code.id()),
        labelled_edges: labelled_edges,
    };

    match code {
        _ if variants.len() == 0 => {
            let r = dot::render(&lcfg, &mut out);
            return expand_err_details(r);
        }
        blocks::BlockCode(_) => {
            ty_cx.sess.err("--pretty flowgraph with -Z flowgraph-print \
                            annotations requires fn-like node id.");
            return Ok(())
        }
        blocks::FnLikeCode(fn_like) => {
            let fn_parts = borrowck::FnPartsWithCFG::from_fn_like(&fn_like, &cfg);
            let (bccx, analysis_data) =
                borrowck::build_borrowck_dataflow_data_for_fn(ty_cx, fn_parts);

            let lcfg = borrowck_dot::DataflowLabeller {
                inner: lcfg,
                variants: variants,
                borrowck_ctxt: &bccx,
                analysis_data: &analysis_data,
            };
            let r = dot::render(&lcfg, &mut out);
            return expand_err_details(r);
        }
    }

    fn expand_err_details(r: old_io::IoResult<()>) -> old_io::IoResult<()> {
        r.map_err(|ioerr| {
            let orig_detail = ioerr.detail.clone();
            let m = "graphviz::render failed";
            old_io::IoError {
                detail: Some(match orig_detail {
                    None => m.to_string(),
                    Some(d) => format!("{}: {}", m, d)
                }),
                ..ioerr
            }
        })
    }
}
