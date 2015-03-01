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

use driver;

use rustc::middle::ty;
use rustc::middle::cfg;
use rustc::middle::cfg::graphviz::LabelledCFG;
use rustc::session::Session;
use rustc::util::ppaux;
use rustc_borrowck as borrowck;
use rustc_borrowck::graphviz as borrowck_dot;

use syntax::ast;
use syntax::ast_map::{self, blocks, NodePrinter};
use syntax::codemap;
use syntax::fold::{self, Folder};
use syntax::print::{pp, pprust};
use syntax::ptr::P;

use graphviz as dot;

use std::borrow::{Cow, IntoCow};
use std::env;
use std::old_io::{self, BufReader};
use std::option;
use std::os;
use std::str::FromStr;

#[derive(Copy, PartialEq, Debug)]
pub enum PpSourceMode {
    PpmNormal,
    PpmEveryBodyLoops,
    PpmExpanded,
    PpmTyped,
    PpmTypedUnsuffixedLiterals,
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

pub fn parse_pretty(name: &str, extended: bool)
                    -> Result<(PpMode, Option<UserIdentifiedItem>), String> {
    let mut split = name.splitn(1, '=');
    let first = split.next().unwrap();
    let opt_second = split.next();
    let first = match (first, extended) {
        ("normal", _)       => PpmSource(PpmNormal),
        ("everybody_loops", true) => PpmSource(PpmEveryBodyLoops),
        ("expanded", _)     => PpmSource(PpmExpanded),
        ("typed", _)        => PpmSource(PpmTyped),
        ("typed,unsuffixed_literals", true) => PpmSource(PpmTypedUnsuffixedLiterals),
        ("expanded,identified", _) => PpmSource(PpmExpandedIdentified),
        ("expanded,hygiene", _) => PpmSource(PpmExpandedHygiene),
        ("identified", _)   => PpmSource(PpmIdentified),
        ("flowgraph", true)    => PpmFlowGraph(PpFlowGraphMode::Default),
        ("flowgraph,unlabelled", true)    => PpmFlowGraph(PpFlowGraphMode::UnlabelledEdges),
        _ => return Err({
            if extended {
                format!(
                    "argument to `xpretty` must be one of `normal`, \
                     `expanded`, `flowgraph[,unlabelled]=<nodeid>`, `typed`, \
                     `typed,unsuffixed_literals`, `identified`, \
                     `expanded,identified`, or `everybody_loops`; got {}", name)
            } else {
                format!(
                    "argument to `pretty` must be one of `normal`, \
                     `expanded`, `typed`, `identified`, \
                     or `expanded,identified`; got {}", name)
            }
        })
    };
    let opt_second = opt_second.and_then(|s| s.parse::<UserIdentifiedItem>().ok());
    Ok((first, opt_second))
}

struct NoAnn;
const NO_ANNOTATION: &'static pprust::PpAnn = &NoAnn;
impl pprust::PpAnn for NoAnn {}

struct IdentifiedAnnotation;
const IDENTIFIED_ANNOTATION: &'static pprust::PpAnn = &IdentifiedAnnotation;
impl pprust::PpAnn for IdentifiedAnnotation {
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

struct HygieneAnnotation;
const HYGIENE_ANNOTATION: &'static pprust::PpAnn = &HygieneAnnotation;
impl pprust::PpAnn for HygieneAnnotation {
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

#[derive(Copy)]
struct TypedAnnotation<'a, 'tcx: 'a> {
    tcx: &'a ty::ctxt<'tcx>,
}

impl<'a, 'tcx> pprust::PpAnn for TypedAnnotation<'a, 'tcx> {
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
            pprust::NodeExpr(expr) => {
                try!(s.word_space(":"));
                try!(pp::word(&mut s.s,
                              &ppaux::ty_to_string(
                                  self.tcx,
                                  ty::expr_ty(self.tcx, expr))));
                s.pclose()
            }
            _ => Ok(())
        }
    }
}

struct UnsuffixedLitAnnotation<A> {
    ann: A
}

impl<A: pprust::PpAnn> pprust::PpAnn for UnsuffixedLitAnnotation<A> {
    fn pre(&self,
           s: &mut pprust::State,
           node: pprust::AnnNode) -> old_io::IoResult<()> {
        self.ann.pre(s, node)
    }
    fn post(&self,
            s: &mut pprust::State,
            node: pprust::AnnNode) -> old_io::IoResult<()> {
        self.ann.post(s, node)
    }
    fn literal_to_string<'a>(&self,
                             s: &mut pprust::State,
                             lit: &'a ast::Lit) -> Cow<'a, str> {
        match lit.node {
            ast::LitInt(i, t) => {
                match t {
                    ast::UnsignedIntLit(_) |
                    ast::SignedIntLit(_, ast::Plus) |
                    ast::UnsuffixedIntLit(ast::Plus) => {
                        format!("{}", i)
                    }
                    ast::SignedIntLit(_, ast::Minus) |
                    ast::UnsuffixedIntLit(ast::Minus) => {
                        format!("-{}", i)
                    }
                }.into_cow()
            }
            ast::LitFloat(ref f, _) => f.into_cow(),
            _ => self.ann.literal_to_string(s, lit)
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
            ItemViaPath(s.split("::").map(|s| s.to_string()).collect())
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

    fn to_one_node_id(&self, user_option: &str, sess: &Session, map: &ast_map::Map)
                      -> ast::NodeId {
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

pub struct ReplaceBodyWithLoop {
    within_static_or_const: bool,
}

impl ReplaceBodyWithLoop {
    pub fn new() -> ReplaceBodyWithLoop {
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

pub fn setup_controller(control: &mut driver::CompileController,
                        ppm_and_uui: Option<(PpMode, Option<UserIdentifiedItem>)>,
                        dump_dir: Option<&str>,
                        keep_going: bool) {

    fn mk_absolute(path: &str) -> Path {
        let path = os::getcwd().unwrap().join(path);
        assert!(path.is_absolute());
        path
    }

    let (ppm, opt_uii, dump_dir, keep_going) = match ppm_and_uui {
        Some((ppm, opt_uii)) => {
            (ppm, opt_uii, dump_dir.map(mk_absolute), keep_going)
        }
        None => {
            let decoded = env::var("RUSTC_PRETTY_DUMP").ok().and_then(|s| {
                let mut s = s.split(":");

                s.next().and_then(|mode| {
                    parse_pretty(mode, true).ok()
                }).and_then(|(ppm, opt_uii)| {
                    s.next().map(|dump_dir| {
                        (ppm, opt_uii, Some(mk_absolute(dump_dir)), true)
                    })
                })
            });
            if let Some(parts) = decoded {
                parts
            } else {
                return;
            }
        }
    };

    if ppm == PpmSource(PpmEveryBodyLoops) {
        control.every_body_loops = true;
    }

    let phase = match ppm {
        PpmSource(PpmNormal) |
        PpmSource(PpmEveryBodyLoops) |
        PpmSource(PpmIdentified) => {
            if opt_uii.is_some() {
                &mut control.after_write_deps
            } else {
                &mut control.after_parse
            }
        }

        PpmSource(PpmExpanded) |
        PpmSource(PpmExpandedIdentified) |
        PpmSource(PpmExpandedHygiene) => {
            &mut control.after_write_deps
        }

        PpmSource(PpmTyped) |
        PpmSource(PpmTypedUnsuffixedLiterals) |
        PpmFlowGraph(_) => {
            &mut control.after_analysis
        }
    };

    phase.callback = box move |state| {
        let pretty_output_path;
        let output = if let Some(ref dir) = dump_dir {
            let file_path = if let Some(outputs) = state.output_filenames {
                outputs.with_extension("rs")
            } else {
                state.session.fatal(
                    "-Z pretty-dump-dir cannot be used with --pretty \
                        options that print before expansion");
            };
            let file_path = os::getcwd().unwrap().join(&file_path);
            assert!(file_path.is_absolute());

            // Cheap isomorphism: /foo/bar--bar/baz <-> foo--bar----bar--baz.
            let components: Vec<_> = file_path.components().map(|bytes| {
                String::from_utf8_lossy(bytes).replace("--", "----")
            }).collect();

            pretty_output_path = dir.join(components.connect("--"));
            Some(&pretty_output_path)
        } else {
            state.output
        };
        print_from_phase(state, ppm, opt_uii.as_ref(), output).unwrap();
    };

    if !keep_going {
        phase.stop = ::Compilation::Stop;
    }
}

fn print_from_phase(state: driver::CompileState,
                    ppm: PpMode,
                    opt_uii: Option<&UserIdentifiedItem>,
                    output: Option<&Path>)
                    -> old_io::IoResult<()> {
    let sess = state.session;
    let krate = state.krate.or(state.expanded_crate)
                           .expect("--pretty=typed missing crate");
    let ast_map = state.ast_map.or_else(|| state.tcx.map(|tcx| &tcx.map));

    let out = match output {
        None => box old_io::stdout() as Box<Writer+'static>,
        Some(p) => {
            match old_io::File::create(p) {
                Ok(w) => box w as Box<Writer+'static>,
                Err(e) => panic!("print-print failed to open {} due to {}",
                                 p.display(), e),
            }
        }
    };

    match ppm {
        PpmSource(mode) => {
            debug!("pretty printing source code {:?}", mode);

            let typed_ann = state.tcx.map(|tcx| TypedAnnotation { tcx: tcx });
            let typed_unsuffixed_lit_ann = typed_ann.map(|ann| {
                UnsuffixedLitAnnotation { ann: ann }
            });
            let annotation: &pprust::PpAnn = match mode {
                PpmNormal | PpmEveryBodyLoops | PpmExpanded => {
                    NO_ANNOTATION
                }
                PpmIdentified | PpmExpandedIdentified => {
                    IDENTIFIED_ANNOTATION
                }
                PpmExpandedHygiene => {
                    HYGIENE_ANNOTATION
                }
                PpmTyped => {
                    typed_ann.as_ref().expect("--pretty=typed missing type context")
                }
                PpmTypedUnsuffixedLiterals => {
                    typed_unsuffixed_lit_ann.as_ref().expect(
                        "--xpretty=typed,unsuffixed_literals missing type context")
                }
            };

            let is_expanded = ast_map.is_some();

            let src_name = driver::source_name(state.input);
            let filemap = sess.codemap().get_filemap(&src_name);
            let mut rdr = BufReader::new(filemap.src.as_bytes());

            if let Some(uii) = opt_uii {
                let ast_map = ast_map.expect("--pretty missing ast_map");
                let mut pp_state =
                    pprust::State::new_from_input(sess.codemap(),
                                                  sess.diagnostic(),
                                                  src_name,
                                                  &mut rdr,
                                                  out,
                                                  annotation,
                                                  is_expanded);
                for node_id in uii.all_matching_node_ids(ast_map) {
                    let node = ast_map.get(node_id);
                    try!(pp_state.print_node(&node));
                    try!(pp::space(&mut pp_state.s));
                    try!(pp_state.synth_comment(ast_map.path_to_string(node_id)));
                    try!(pp::hardbreak(&mut pp_state.s));
                }
                pp::eof(&mut pp_state.s)
            } else {
                pprust::print_crate(sess.codemap(),
                                    sess.diagnostic(),
                                    krate,
                                    src_name,
                                    &mut rdr,
                                    out,
                                    annotation,
                                    is_expanded)
            }
        }

        PpmFlowGraph(mode) => {
            debug!("pretty printing flow graph for {:?}", opt_uii);
            let uii = opt_uii.unwrap_or_else(|| {
                sess.fatal(&format!("`pretty flowgraph=..` needs NodeId (int) or
                                     unique path suffix (b::c::d)"))

            });
            let tcx = state.tcx.expect("--pretty flowgraph missing type context");
            let nodeid = uii.to_one_node_id("--pretty", sess, &tcx.map);

            let node = tcx.map.find(nodeid).unwrap_or_else(|| {
                sess.fatal(&format!("--pretty flowgraph couldn't find id: {}",
                                   nodeid))
            });

            let code = blocks::Code::from_node(node);
            match code {
                Some(code) => {
                    let variants = gather_flowgraph_variants(sess);
                    print_flowgraph(variants, tcx, code, mode, out)
                }
                None => {
                    let message = format!("--pretty=flowgraph needs \
                                           block, fn, or method; got {:?}",
                                          node);

                    // point to what was found, if there's an
                    // accessible span.
                    match tcx.map.opt_span(nodeid) {
                        Some(sp) => sess.span_fatal(sp, &message[..]),
                        None => sess.fatal(&message[..])
                    }
                }
            }
        }
    }
}

fn print_flowgraph<W:old_io::Writer>(variants: Vec<borrowck_dot::Variant>,
                                     tcx: &ty::ctxt,
                                     code: blocks::Code,
                                     mode: PpFlowGraphMode,
                                     mut out: W) -> old_io::IoResult<()> {
    let cfg = match code {
        blocks::BlockCode(block) => cfg::CFG::new(tcx, &*block),
        blocks::FnLikeCode(fn_like) => cfg::CFG::new(tcx, &*fn_like.body()),
    };
    let labelled_edges = mode != PpFlowGraphMode::UnlabelledEdges;
    let lcfg = LabelledCFG {
        ast_map: &tcx.map,
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
            tcx.sess.err("--pretty flowgraph with -Z flowgraph-print \
                          annotations requires fn-like node id.");
            return Ok(())
        }
        blocks::FnLikeCode(fn_like) => {
            let fn_parts = borrowck::FnPartsWithCFG::from_fn_like(&fn_like, &cfg);
            let (bccx, analysis_data) =
                borrowck::build_borrowck_dataflow_data_for_fn(tcx, fn_parts);

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
