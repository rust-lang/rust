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

use back::link;

use driver::config;
use driver::driver::{mod, CrateAnalysis};
use driver::session::Session;

use middle::ty;
use middle::borrowck::{mod, FnPartsWithCFG};
use middle::borrowck::graphviz as borrowck_dot;
use middle::cfg;
use middle::cfg::graphviz::LabelledCFG;

use util::ppaux;

use syntax::ast;
use syntax::ast_map::{mod, blocks, NodePrinter};
use syntax::print::{pp, pprust};

use graphviz as dot;

use std::io::{mod, MemReader};
use std::from_str::FromStr;
use std::option;
use arena::TypedArena;

#[deriving(PartialEq, Show)]
pub enum PpSourceMode {
    PpmNormal,
    PpmExpanded,
    PpmTyped,
    PpmIdentified,
    PpmExpandedIdentified,
    PpmExpandedHygiene,
}

#[deriving(PartialEq, Show)]
pub enum PpMode {
    PpmSource(PpSourceMode),
    PpmFlowGraph,
}

pub fn parse_pretty(sess: &Session, name: &str) -> (PpMode, Option<UserIdentifiedItem>) {
    let mut split = name.splitn(1, '=');
    let first = split.next().unwrap();
    let opt_second = split.next();
    let first = match first {
        "normal"       => PpmSource(PpmNormal),
        "expanded"     => PpmSource(PpmExpanded),
        "typed"        => PpmSource(PpmTyped),
        "expanded,identified" => PpmSource(PpmExpandedIdentified),
        "expanded,hygiene" => PpmSource(PpmExpandedHygiene),
        "identified"   => PpmSource(PpmIdentified),
        "flowgraph"    => PpmFlowGraph,
        _ => {
            sess.fatal(format!(
                "argument to `pretty` must be one of `normal`, \
                 `expanded`, `flowgraph=<nodeid>`, `typed`, `identified`, \
                 or `expanded,identified`; got {}", name).as_slice());
        }
    };
    let opt_second = opt_second.and_then::<UserIdentifiedItem>(from_str);
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
    fn call_with_pp_support<A,B>(&self,
                                 sess: Session,
                                 krate: &ast::Crate,
                                 ast_map: Option<ast_map::Map>,
                                 id: String,
                                 payload: B,
                                 f: |&PrinterSupport, B| -> A) -> A {
        match *self {
            PpmNormal | PpmExpanded => {
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
                let type_arena = TypedArena::new();
                let analysis = driver::phase_3_run_analysis_passes(sess, krate, ast_map,
                                                                   &type_arena, id);
                let annotation = TypedAnnotation { analysis: analysis };
                f(&annotation, payload)
            }
        }
    }
}

trait SessionCarrier {
    /// Provides a uniform interface for re-extracting a reference to a
    /// `Session` from a value that now owns it.
    fn sess<'a>(&'a self) -> &'a Session;
}

trait AstMapCarrier {
    /// Provides a uniform interface for re-extracting a reference to an
    /// `ast_map::Map` from a value that now owns it.
    fn ast_map<'a>(&'a self) -> Option<&'a ast_map::Map>;
}

trait PrinterSupport : SessionCarrier + AstMapCarrier {
    /// Produces the pretty-print annotation object.
    ///
    /// Usually implemented via `self as &pprust::PpAnn`.
    ///
    /// (Rust does not yet support upcasting from a trait object to
    /// an object for one of its super-traits.)
    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn;
}

struct NoAnn {
    sess: Session,
    ast_map: Option<ast_map::Map>,
}

impl PrinterSupport for NoAnn {
    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn { self as &pprust::PpAnn }
}

impl SessionCarrier for NoAnn {
    fn sess<'a>(&'a self) -> &'a Session { &self.sess }
}

impl AstMapCarrier for NoAnn {
    fn ast_map<'a>(&'a self) -> Option<&'a ast_map::Map> {
        self.ast_map.as_ref()
    }
}

impl pprust::PpAnn for NoAnn {}

struct IdentifiedAnnotation {
    sess: Session,
    ast_map: Option<ast_map::Map>,
}

impl PrinterSupport for IdentifiedAnnotation {
    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn { self as &pprust::PpAnn }
}

impl SessionCarrier for IdentifiedAnnotation {
    fn sess<'a>(&'a self) -> &'a Session { &self.sess }
}

impl AstMapCarrier for IdentifiedAnnotation {
    fn ast_map<'a>(&'a self) -> Option<&'a ast_map::Map> {
        self.ast_map.as_ref()
    }
}

impl pprust::PpAnn for IdentifiedAnnotation {
    fn pre(&self,
           s: &mut pprust::State,
           node: pprust::AnnNode) -> io::IoResult<()> {
        match node {
            pprust::NodeExpr(_) => s.popen(),
            _ => Ok(())
        }
    }
    fn post(&self,
            s: &mut pprust::State,
            node: pprust::AnnNode) -> io::IoResult<()> {
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

struct HygieneAnnotation {
    sess: Session,
    ast_map: Option<ast_map::Map>,
}

impl PrinterSupport for HygieneAnnotation {
    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn { self as &pprust::PpAnn }
}

impl SessionCarrier for HygieneAnnotation {
    fn sess<'a>(&'a self) -> &'a Session { &self.sess }
}

impl AstMapCarrier for HygieneAnnotation {
    fn ast_map<'a>(&'a self) -> Option<&'a ast_map::Map> {
        self.ast_map.as_ref()
    }
}

impl pprust::PpAnn for HygieneAnnotation {
    fn post(&self,
            s: &mut pprust::State,
            node: pprust::AnnNode) -> io::IoResult<()> {
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
    analysis: CrateAnalysis<'tcx>,
}

impl<'tcx> PrinterSupport for TypedAnnotation<'tcx> {
    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn { self as &pprust::PpAnn }
}

impl<'tcx> SessionCarrier for TypedAnnotation<'tcx> {
    fn sess<'a>(&'a self) -> &'a Session { &self.analysis.ty_cx.sess }
}

impl<'tcx> AstMapCarrier for TypedAnnotation<'tcx> {
    fn ast_map<'a>(&'a self) -> Option<&'a ast_map::Map> {
        Some(&self.analysis.ty_cx.map)
    }
}

impl<'tcx> pprust::PpAnn for TypedAnnotation<'tcx> {
    fn pre(&self,
           s: &mut pprust::State,
           node: pprust::AnnNode) -> io::IoResult<()> {
        match node {
            pprust::NodeExpr(_) => s.popen(),
            _ => Ok(())
        }
    }
    fn post(&self,
            s: &mut pprust::State,
            node: pprust::AnnNode) -> io::IoResult<()> {
        let tcx = &self.analysis.ty_cx;
        match node {
            pprust::NodeExpr(expr) => {
                try!(pp::space(&mut s.s));
                try!(pp::word(&mut s.s, "as"));
                try!(pp::space(&mut s.s));
                try!(pp::word(&mut s.s,
                              ppaux::ty_to_string(
                                  tcx,
                                  ty::expr_ty(tcx, expr)).as_slice()));
                s.pclose()
            }
            _ => Ok(())
        }
    }
}

fn gather_flowgraph_variants(sess: &Session) -> Vec<borrowck_dot::Variant> {
    let print_loans   = config::FLOWGRAPH_PRINT_LOANS;
    let print_moves   = config::FLOWGRAPH_PRINT_MOVES;
    let print_assigns = config::FLOWGRAPH_PRINT_ASSIGNS;
    let print_all     = config::FLOWGRAPH_PRINT_ALL;
    let opt = |print_which| sess.debugging_opt(print_which);
    let mut variants = Vec::new();
    if opt(print_all) || opt(print_loans) {
        variants.push(borrowck_dot::Loans);
    }
    if opt(print_all) || opt(print_moves) {
        variants.push(borrowck_dot::Moves);
    }
    if opt(print_all) || opt(print_assigns) {
        variants.push(borrowck_dot::Assigns);
    }
    variants
}

#[deriving(Clone, Show)]
pub enum UserIdentifiedItem {
    ItemViaNode(ast::NodeId),
    ItemViaPath(Vec<String>),
}

impl FromStr for UserIdentifiedItem {
    fn from_str(s: &str) -> Option<UserIdentifiedItem> {
        let extract_path_parts = || {
            let v : Vec<_> = s.split_str("::")
                .map(|x|x.to_string())
                .collect();
            Some(ItemViaPath(v))
        };

        from_str(s).map(ItemViaNode).or_else(extract_path_parts)
    }
}

enum NodesMatchingUII<'a> {
    NodesMatchingDirect(option::Item<ast::NodeId>),
    NodesMatchingSuffix(ast_map::NodesMatchingSuffix<'a, String>),
}

impl<'a> Iterator<ast::NodeId> for NodesMatchingUII<'a> {
    fn next(&mut self) -> Option<ast::NodeId> {
        match self {
            &NodesMatchingDirect(ref mut iter) => iter.next(),
            &NodesMatchingSuffix(ref mut iter) => iter.next(),
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

    fn all_matching_node_ids<'a>(&'a self, map: &'a ast_map::Map) -> NodesMatchingUII<'a> {
        match *self {
            ItemViaNode(node_id) =>
                NodesMatchingDirect(Some(node_id).move_iter()),
            ItemViaPath(ref parts) =>
                NodesMatchingSuffix(map.nodes_matching_suffix(parts.as_slice())),
        }
    }

    fn to_one_node_id(self, user_option: &str, sess: &Session, map: &ast_map::Map) -> ast::NodeId {
        let fail_because = |is_wrong_because| -> ast::NodeId {
            let message =
                format!("{:s} needs NodeId (int) or unique \
                         path suffix (b::c::d); got {:s}, which {:s}",
                        user_option,
                        self.reconstructed_input(),
                        is_wrong_because);
            sess.fatal(message.as_slice())
        };

        let mut saw_node = ast::DUMMY_NODE_ID;
        let mut seen = 0u;
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
        PpmSource(PpmIdentified) => opt_uii.is_some(),

        PpmSource(PpmExpanded) |
        PpmSource(PpmExpandedIdentified) |
        PpmSource(PpmExpandedHygiene) |
        PpmSource(PpmTyped) |
        PpmFlowGraph => true
    }
}

fn needs_expansion(ppm: &PpMode) -> bool {
    match *ppm {
        PpmSource(PpmNormal) |
        PpmSource(PpmIdentified) => false,

        PpmSource(PpmExpanded) |
        PpmSource(PpmExpandedIdentified) |
        PpmSource(PpmExpandedHygiene) |
        PpmSource(PpmTyped) |
        PpmFlowGraph => true
    }
}

pub fn pretty_print_input(sess: Session,
                          cfg: ast::CrateConfig,
                          input: &driver::Input,
                          ppm: PpMode,
                          opt_uii: Option<UserIdentifiedItem>,
                          ofile: Option<Path>) {
    let krate = driver::phase_1_parse_input(&sess, cfg, input);
    let id = link::find_crate_name(Some(&sess), krate.attrs.as_slice(), input);

    let is_expanded = needs_expansion(&ppm);
    let (krate, ast_map) = if needs_ast_map(&ppm, &opt_uii) {
        let k = driver::phase_2_configure_and_expand(&sess, krate, id.as_slice(), None);
        let (krate, ast_map) = match k {
            None => return,
            Some(p) => p,
        };
        (krate, Some(ast_map))
    } else {
        (krate, None)
    };

    let src_name = driver::source_name(input);
    let src = Vec::from_slice(sess.codemap()
                                  .get_filemap(src_name.as_slice())
                                  .src
                                  .as_bytes());
    let mut rdr = MemReader::new(src);

    let out = match ofile {
        None => box io::stdout() as Box<Writer+'static>,
        Some(p) => {
            let r = io::File::create(&p);
            match r {
                Ok(w) => box w as Box<Writer+'static>,
                Err(e) => fail!("print-print failed to open {} due to {}",
                                p.display(), e),
            }
        }
    };

    match (ppm, opt_uii) {
        (PpmSource(s), None) =>
            s.call_with_pp_support(
                sess, &krate, ast_map, id, out, |annotation, out| {
                    debug!("pretty printing source code {}", s);
                    let sess = annotation.sess();
                    pprust::print_crate(sess.codemap(),
                                        sess.diagnostic(),
                                        &krate,
                                        src_name.to_string(),
                                        &mut rdr,
                                        out,
                                        annotation.pp_ann(),
                                        is_expanded)
                }),

        (PpmSource(s), Some(uii)) =>
            s.call_with_pp_support(
                sess, &krate, ast_map, id, (out,uii), |annotation, (out,uii)| {
                    debug!("pretty printing source code {}", s);
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

        (PpmFlowGraph, opt_uii) => {
            debug!("pretty printing flow graph for {}", opt_uii);
            let uii = opt_uii.unwrap_or_else(|| {
                sess.fatal(format!("`pretty flowgraph=..` needs NodeId (int) or
                                     unique path suffix (b::c::d)").as_slice())

            });
            let ast_map = ast_map.expect("--pretty flowgraph missing ast_map");
            let nodeid = uii.to_one_node_id("--pretty", &sess, &ast_map);

            let node = ast_map.find(nodeid).unwrap_or_else(|| {
                sess.fatal(format!("--pretty flowgraph couldn't find id: {}",
                                   nodeid).as_slice())
            });

            let code = blocks::Code::from_node(node);
            match code {
                Some(code) => {
                    let variants = gather_flowgraph_variants(&sess);
                    let type_arena = TypedArena::new();
                    let analysis = driver::phase_3_run_analysis_passes(sess, &krate,
                                                                       ast_map, &type_arena, id);
                    print_flowgraph(variants, analysis, code, out)
                }
                None => {
                    let message = format!("--pretty=flowgraph needs \
                                           block, fn, or method; got {:?}",
                                          node);

                    // point to what was found, if there's an
                    // accessible span.
                    match ast_map.opt_span(nodeid) {
                        Some(sp) => sess.span_fatal(sp, message.as_slice()),
                        None => sess.fatal(message.as_slice())
                    }
                }
            }
        }
    }.unwrap()
}

fn print_flowgraph<W:io::Writer>(variants: Vec<borrowck_dot::Variant>,
                                 analysis: CrateAnalysis,
                                 code: blocks::Code,
                                 mut out: W) -> io::IoResult<()> {
    let ty_cx = &analysis.ty_cx;
    let cfg = match code {
        blocks::BlockCode(block) => cfg::CFG::new(ty_cx, &*block),
        blocks::FnLikeCode(fn_like) => cfg::CFG::new(ty_cx, &*fn_like.body()),
    };
    debug!("cfg: {:?}", cfg);

    match code {
        _ if variants.len() == 0 => {
            let lcfg = LabelledCFG {
                ast_map: &ty_cx.map,
                cfg: &cfg,
                name: format!("node_{}", code.id()),
            };
            let r = dot::render(&lcfg, &mut out);
            return expand_err_details(r);
        }
        blocks::BlockCode(_) => {
            ty_cx.sess.err("--pretty flowgraph with -Z flowgraph-print \
                            annotations requires fn-like node id.");
            return Ok(())
        }
        blocks::FnLikeCode(fn_like) => {
            let fn_parts = FnPartsWithCFG::from_fn_like(&fn_like, &cfg);
            let (bccx, analysis_data) =
                borrowck::build_borrowck_dataflow_data_for_fn(ty_cx, fn_parts);

            let lcfg = LabelledCFG {
                ast_map: &ty_cx.map,
                cfg: &cfg,
                name: format!("node_{}", code.id()),
            };
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

    fn expand_err_details(r: io::IoResult<()>) -> io::IoResult<()> {
        r.map_err(|ioerr| {
            let orig_detail = ioerr.detail.clone();
            let m = "graphviz::render failed";
            io::IoError {
                detail: Some(match orig_detail {
                    None => m.into_string(),
                    Some(d) => format!("{}: {}", m, d)
                }),
                ..ioerr
            }
        })
    }
}
