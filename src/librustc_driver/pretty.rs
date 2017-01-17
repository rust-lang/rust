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

use {abort_on_err, driver};

use rustc::ty::{self, TyCtxt, GlobalArenas, Resolutions};
use rustc::cfg;
use rustc::cfg::graphviz::LabelledCFG;
use rustc::dep_graph::DepGraph;
use rustc::session::Session;
use rustc::session::config::Input;
use rustc_borrowck as borrowck;
use rustc_borrowck::graphviz as borrowck_dot;

use rustc_mir::pretty::write_mir_pretty;
use rustc_mir::graphviz::write_mir_graphviz;

use syntax::ast::{self, BlockCheckMode};
use syntax::fold::{self, Folder};
use syntax::print::{pp, pprust};
use syntax::print::pprust::PrintState;
use syntax::ptr::P;
use syntax::util::small_vector::SmallVector;
use syntax_pos;

use graphviz as dot;

use std::cell::Cell;
use std::fs::File;
use std::io::{self, Write};
use std::iter;
use std::option;
use std::path::Path;
use std::str::FromStr;

use rustc::hir::map as hir_map;
use rustc::hir::map::blocks;
use rustc::hir;
use rustc::hir::print as pprust_hir;

use arena::DroplessArena;

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
    /// is mostly for use in the --unpretty flowgraph run-make tests,
    /// since the labels are largely uninteresting in those cases and
    /// have become a pain to maintain.
    UnlabelledEdges,
}
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum PpMode {
    PpmSource(PpSourceMode),
    PpmHir(PpSourceMode),
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
        ("mir", true) => PpmMir,
        ("mir-cfg", true) => PpmMirCFG,
        ("flowgraph", true) => PpmFlowGraph(PpFlowGraphMode::Default),
        ("flowgraph,unlabelled", true) => PpmFlowGraph(PpFlowGraphMode::UnlabelledEdges),
        _ => {
            if extended {
                sess.fatal(&format!("argument to `unpretty` must be one of `normal`, \
                                     `expanded`, `flowgraph[,unlabelled]=<nodeid>`, \
                                     `identified`, `expanded,identified`, `everybody_loops`, \
                                     `hir`, `hir,identified`, `hir,typed`, or `mir`; got {}",
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
    fn call_with_pp_support<'tcx, A, B, F>(&self,
                                           sess: &'tcx Session,
                                           ast_map: Option<&hir_map::Map<'tcx>>,
                                           payload: B,
                                           f: F)
                                           -> A
        where F: FnOnce(&PrinterSupport, B) -> A
    {
        match *self {
            PpmNormal | PpmEveryBodyLoops | PpmExpanded => {
                let annotation = NoAnn {
                    sess: sess,
                    ast_map: ast_map.map(|m| m.clone()),
                };
                f(&annotation, payload)
            }

            PpmIdentified | PpmExpandedIdentified => {
                let annotation = IdentifiedAnnotation {
                    sess: sess,
                    ast_map: ast_map.map(|m| m.clone()),
                };
                f(&annotation, payload)
            }
            PpmExpandedHygiene => {
                let annotation = HygieneAnnotation {
                    sess: sess,
                    ast_map: ast_map.map(|m| m.clone()),
                };
                f(&annotation, payload)
            }
            _ => panic!("Should use call_with_pp_support_hir"),
        }
    }
    fn call_with_pp_support_hir<'tcx, A, B, F>(&self,
                                               sess: &'tcx Session,
                                               ast_map: &hir_map::Map<'tcx>,
                                               analysis: &ty::CrateAnalysis<'tcx>,
                                               resolutions: &Resolutions,
                                               arena: &'tcx DroplessArena,
                                               arenas: &'tcx GlobalArenas<'tcx>,
                                               id: &str,
                                               payload: B,
                                               f: F)
                                               -> A
        where F: FnOnce(&HirPrinterSupport, B, &hir::Crate) -> A
    {
        match *self {
            PpmNormal => {
                let annotation = NoAnn {
                    sess: sess,
                    ast_map: Some(ast_map.clone()),
                };
                f(&annotation, payload, ast_map.forest.krate())
            }

            PpmIdentified => {
                let annotation = IdentifiedAnnotation {
                    sess: sess,
                    ast_map: Some(ast_map.clone()),
                };
                f(&annotation, payload, ast_map.forest.krate())
            }
            PpmTyped => {
                abort_on_err(driver::phase_3_run_analysis_passes(sess,
                                                                 ast_map.clone(),
                                                                 analysis.clone(),
                                                                 resolutions.clone(),
                                                                 arena,
                                                                 arenas,
                                                                 id,
                                                                 |tcx, _, _, _| {
                    let empty_tables = ty::Tables::empty();
                    let annotation = TypedAnnotation {
                        tcx: tcx,
                        tables: Cell::new(&empty_tables)
                    };
                    let _ignore = tcx.dep_graph.in_ignore();
                    f(&annotation, payload, ast_map.forest.krate())
                }),
                             sess)
            }
            _ => panic!("Should use call_with_pp_support"),
        }
    }
}

trait PrinterSupport<'ast>: pprust::PpAnn {
    /// Provides a uniform interface for re-extracting a reference to a
    /// `Session` from a value that now owns it.
    fn sess<'a>(&'a self) -> &'a Session;

    /// Provides a uniform interface for re-extracting a reference to an
    /// `hir_map::Map` from a value that now owns it.
    fn ast_map<'a>(&'a self) -> Option<&'a hir_map::Map<'ast>>;

    /// Produces the pretty-print annotation object.
    ///
    /// (Rust does not yet support upcasting from a trait object to
    /// an object for one of its super-traits.)
    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn;
}

trait HirPrinterSupport<'ast>: pprust_hir::PpAnn {
    /// Provides a uniform interface for re-extracting a reference to a
    /// `Session` from a value that now owns it.
    fn sess<'a>(&'a self) -> &'a Session;

    /// Provides a uniform interface for re-extracting a reference to an
    /// `hir_map::Map` from a value that now owns it.
    fn ast_map<'a>(&'a self) -> Option<&'a hir_map::Map<'ast>>;

    /// Produces the pretty-print annotation object.
    ///
    /// (Rust does not yet support upcasting from a trait object to
    /// an object for one of its super-traits.)
    fn pp_ann<'a>(&'a self) -> &'a pprust_hir::PpAnn;

    /// Computes an user-readable representation of a path, if possible.
    fn node_path(&self, id: ast::NodeId) -> Option<String> {
        self.ast_map().and_then(|map| map.def_path_from_id(id)).map(|path| {
            path.data
                .into_iter()
                .map(|elem| elem.data.to_string())
                .collect::<Vec<_>>()
                .join("::")
        })
    }
}

struct NoAnn<'ast> {
    sess: &'ast Session,
    ast_map: Option<hir_map::Map<'ast>>,
}

impl<'ast> PrinterSupport<'ast> for NoAnn<'ast> {
    fn sess<'a>(&'a self) -> &'a Session {
        self.sess
    }

    fn ast_map<'a>(&'a self) -> Option<&'a hir_map::Map<'ast>> {
        self.ast_map.as_ref()
    }

    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn {
        self
    }
}

impl<'ast> HirPrinterSupport<'ast> for NoAnn<'ast> {
    fn sess<'a>(&'a self) -> &'a Session {
        self.sess
    }

    fn ast_map<'a>(&'a self) -> Option<&'a hir_map::Map<'ast>> {
        self.ast_map.as_ref()
    }

    fn pp_ann<'a>(&'a self) -> &'a pprust_hir::PpAnn {
        self
    }
}

impl<'ast> pprust::PpAnn for NoAnn<'ast> {}
impl<'ast> pprust_hir::PpAnn for NoAnn<'ast> {
    fn nested(&self, state: &mut pprust_hir::State, nested: pprust_hir::Nested)
              -> io::Result<()> {
        if let Some(ref map) = self.ast_map {
            pprust_hir::PpAnn::nested(map, state, nested)
        } else {
            Ok(())
        }
    }
}

struct IdentifiedAnnotation<'ast> {
    sess: &'ast Session,
    ast_map: Option<hir_map::Map<'ast>>,
}

impl<'ast> PrinterSupport<'ast> for IdentifiedAnnotation<'ast> {
    fn sess<'a>(&'a self) -> &'a Session {
        self.sess
    }

    fn ast_map<'a>(&'a self) -> Option<&'a hir_map::Map<'ast>> {
        self.ast_map.as_ref()
    }

    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn {
        self
    }
}

impl<'ast> pprust::PpAnn for IdentifiedAnnotation<'ast> {
    fn pre(&self, s: &mut pprust::State, node: pprust::AnnNode) -> io::Result<()> {
        match node {
            pprust::NodeExpr(_) => s.popen(),
            _ => Ok(()),
        }
    }
    fn post(&self, s: &mut pprust::State, node: pprust::AnnNode) -> io::Result<()> {
        match node {
            pprust::NodeIdent(_) |
            pprust::NodeName(_) => Ok(()),

            pprust::NodeItem(item) => {
                pp::space(&mut s.s)?;
                s.synth_comment(item.id.to_string())
            }
            pprust::NodeSubItem(id) => {
                pp::space(&mut s.s)?;
                s.synth_comment(id.to_string())
            }
            pprust::NodeBlock(blk) => {
                pp::space(&mut s.s)?;
                s.synth_comment(format!("block {}", blk.id))
            }
            pprust::NodeExpr(expr) => {
                pp::space(&mut s.s)?;
                s.synth_comment(expr.id.to_string())?;
                s.pclose()
            }
            pprust::NodePat(pat) => {
                pp::space(&mut s.s)?;
                s.synth_comment(format!("pat {}", pat.id))
            }
        }
    }
}

impl<'ast> HirPrinterSupport<'ast> for IdentifiedAnnotation<'ast> {
    fn sess<'a>(&'a self) -> &'a Session {
        self.sess
    }

    fn ast_map<'a>(&'a self) -> Option<&'a hir_map::Map<'ast>> {
        self.ast_map.as_ref()
    }

    fn pp_ann<'a>(&'a self) -> &'a pprust_hir::PpAnn {
        self
    }
}

impl<'ast> pprust_hir::PpAnn for IdentifiedAnnotation<'ast> {
    fn nested(&self, state: &mut pprust_hir::State, nested: pprust_hir::Nested)
              -> io::Result<()> {
        if let Some(ref map) = self.ast_map {
            pprust_hir::PpAnn::nested(map, state, nested)
        } else {
            Ok(())
        }
    }
    fn pre(&self, s: &mut pprust_hir::State, node: pprust_hir::AnnNode) -> io::Result<()> {
        match node {
            pprust_hir::NodeExpr(_) => s.popen(),
            _ => Ok(()),
        }
    }
    fn post(&self, s: &mut pprust_hir::State, node: pprust_hir::AnnNode) -> io::Result<()> {
        match node {
            pprust_hir::NodeName(_) => Ok(()),
            pprust_hir::NodeItem(item) => {
                pp::space(&mut s.s)?;
                s.synth_comment(item.id.to_string())
            }
            pprust_hir::NodeSubItem(id) => {
                pp::space(&mut s.s)?;
                s.synth_comment(id.to_string())
            }
            pprust_hir::NodeBlock(blk) => {
                pp::space(&mut s.s)?;
                s.synth_comment(format!("block {}", blk.id))
            }
            pprust_hir::NodeExpr(expr) => {
                pp::space(&mut s.s)?;
                s.synth_comment(expr.id.to_string())?;
                s.pclose()
            }
            pprust_hir::NodePat(pat) => {
                pp::space(&mut s.s)?;
                s.synth_comment(format!("pat {}", pat.id))
            }
        }
    }
}

struct HygieneAnnotation<'ast> {
    sess: &'ast Session,
    ast_map: Option<hir_map::Map<'ast>>,
}

impl<'ast> PrinterSupport<'ast> for HygieneAnnotation<'ast> {
    fn sess<'a>(&'a self) -> &'a Session {
        self.sess
    }

    fn ast_map<'a>(&'a self) -> Option<&'a hir_map::Map<'ast>> {
        self.ast_map.as_ref()
    }

    fn pp_ann<'a>(&'a self) -> &'a pprust::PpAnn {
        self
    }
}

impl<'ast> pprust::PpAnn for HygieneAnnotation<'ast> {
    fn post(&self, s: &mut pprust::State, node: pprust::AnnNode) -> io::Result<()> {
        match node {
            pprust::NodeIdent(&ast::Ident { name, ctxt }) => {
                pp::space(&mut s.s)?;
                // FIXME #16420: this doesn't display the connections
                // between syntax contexts
                s.synth_comment(format!("{}{:?}", name.as_u32(), ctxt))
            }
            pprust::NodeName(&name) => {
                pp::space(&mut s.s)?;
                s.synth_comment(name.as_u32().to_string())
            }
            _ => Ok(()),
        }
    }
}


struct TypedAnnotation<'a, 'tcx: 'a> {
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    tables: Cell<&'a ty::Tables<'tcx>>,
}

impl<'b, 'tcx> HirPrinterSupport<'tcx> for TypedAnnotation<'b, 'tcx> {
    fn sess<'a>(&'a self) -> &'a Session {
        &self.tcx.sess
    }

    fn ast_map<'a>(&'a self) -> Option<&'a hir_map::Map<'tcx>> {
        Some(&self.tcx.map)
    }

    fn pp_ann<'a>(&'a self) -> &'a pprust_hir::PpAnn {
        self
    }

    fn node_path(&self, id: ast::NodeId) -> Option<String> {
        Some(self.tcx.node_path_str(id))
    }
}

impl<'a, 'tcx> pprust_hir::PpAnn for TypedAnnotation<'a, 'tcx> {
    fn nested(&self, state: &mut pprust_hir::State, nested: pprust_hir::Nested)
              -> io::Result<()> {
        let old_tables = self.tables.get();
        if let pprust_hir::Nested::Body(id) = nested {
            self.tables.set(self.tcx.body_tables(id));
        }
        pprust_hir::PpAnn::nested(&self.tcx.map, state, nested)?;
        self.tables.set(old_tables);
        Ok(())
    }
    fn pre(&self, s: &mut pprust_hir::State, node: pprust_hir::AnnNode) -> io::Result<()> {
        match node {
            pprust_hir::NodeExpr(_) => s.popen(),
            _ => Ok(()),
        }
    }
    fn post(&self, s: &mut pprust_hir::State, node: pprust_hir::AnnNode) -> io::Result<()> {
        match node {
            pprust_hir::NodeExpr(expr) => {
                pp::space(&mut s.s)?;
                pp::word(&mut s.s, "as")?;
                pp::space(&mut s.s)?;
                pp::word(&mut s.s, &self.tables.get().expr_ty(expr).to_string())?;
                s.pclose()
            }
            _ => Ok(()),
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
            .map(ast::NodeId::new)
            .map(ItemViaNode)
            .unwrap_or_else(|_| ItemViaPath(s.split("::").map(|s| s.to_string()).collect())))
    }
}

enum NodesMatchingUII<'a, 'ast: 'a> {
    NodesMatchingDirect(option::IntoIter<ast::NodeId>),
    NodesMatchingSuffix(hir_map::NodesMatchingSuffix<'a, 'ast>),
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
            ItemViaPath(ref parts) => parts.join("::"),
        }
    }

    fn all_matching_node_ids<'a, 'ast>(&'a self,
                                       map: &'a hir_map::Map<'ast>)
                                       -> NodesMatchingUII<'a, 'ast> {
        match *self {
            ItemViaNode(node_id) => NodesMatchingDirect(Some(node_id).into_iter()),
            ItemViaPath(ref parts) => NodesMatchingSuffix(map.nodes_matching_suffix(&parts[..])),
        }
    }

    fn to_one_node_id(self, user_option: &str, sess: &Session, map: &hir_map::Map) -> ast::NodeId {
        let fail_because = |is_wrong_because| -> ast::NodeId {
            let message = format!("{} needs NodeId (int) or unique path suffix (b::c::d); got \
                                   {}, which {}",
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

struct ReplaceBodyWithLoop {
    within_static_or_const: bool,
}

impl ReplaceBodyWithLoop {
    fn new() -> ReplaceBodyWithLoop {
        ReplaceBodyWithLoop { within_static_or_const: false }
    }
}

impl fold::Folder for ReplaceBodyWithLoop {
    fn fold_item_kind(&mut self, i: ast::ItemKind) -> ast::ItemKind {
        match i {
            ast::ItemKind::Static(..) |
            ast::ItemKind::Const(..) => {
                self.within_static_or_const = true;
                let ret = fold::noop_fold_item_kind(i, self);
                self.within_static_or_const = false;
                return ret;
            }
            _ => fold::noop_fold_item_kind(i, self),
        }
    }

    fn fold_trait_item(&mut self, i: ast::TraitItem) -> SmallVector<ast::TraitItem> {
        match i.node {
            ast::TraitItemKind::Const(..) => {
                self.within_static_or_const = true;
                let ret = fold::noop_fold_trait_item(i, self);
                self.within_static_or_const = false;
                return ret;
            }
            _ => fold::noop_fold_trait_item(i, self),
        }
    }

    fn fold_impl_item(&mut self, i: ast::ImplItem) -> SmallVector<ast::ImplItem> {
        match i.node {
            ast::ImplItemKind::Const(..) => {
                self.within_static_or_const = true;
                let ret = fold::noop_fold_impl_item(i, self);
                self.within_static_or_const = false;
                return ret;
            }
            _ => fold::noop_fold_impl_item(i, self),
        }
    }

    fn fold_block(&mut self, b: P<ast::Block>) -> P<ast::Block> {
        fn expr_to_block(rules: ast::BlockCheckMode, e: Option<P<ast::Expr>>) -> P<ast::Block> {
            P(ast::Block {
                stmts: e.map(|e| {
                        ast::Stmt {
                            id: ast::DUMMY_NODE_ID,
                            span: e.span,
                            node: ast::StmtKind::Expr(e),
                        }
                    })
                    .into_iter()
                    .collect(),
                rules: rules,
                id: ast::DUMMY_NODE_ID,
                span: syntax_pos::DUMMY_SP,
            })
        }

        if !self.within_static_or_const {

            let empty_block = expr_to_block(BlockCheckMode::Default, None);
            let loop_expr = P(ast::Expr {
                node: ast::ExprKind::Loop(empty_block, None),
                id: ast::DUMMY_NODE_ID,
                span: syntax_pos::DUMMY_SP,
                attrs: ast::ThinVec::new(),
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

fn print_flowgraph<'a, 'tcx, W: Write>(variants: Vec<borrowck_dot::Variant>,
                                       tcx: TyCtxt<'a, 'tcx, 'tcx>,
                                       code: blocks::Code<'tcx>,
                                       mode: PpFlowGraphMode,
                                       mut out: W)
                                       -> io::Result<()> {
    let cfg = match code {
        blocks::Code::Expr(expr) => cfg::CFG::new(tcx, expr),
        blocks::Code::FnLike(fn_like) => {
            let body = tcx.map.body(fn_like.body());
            cfg::CFG::new(tcx, &body.value)
        },
    };
    let labelled_edges = mode != PpFlowGraphMode::UnlabelledEdges;
    let lcfg = LabelledCFG {
        ast_map: &tcx.map,
        cfg: &cfg,
        name: format!("node_{}", code.id()),
        labelled_edges: labelled_edges,
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
                variants: variants,
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
                           &format!("graphviz::render failed: {}", ioerr)[..])
        })
    }
}

pub fn fold_crate(krate: ast::Crate, ppm: PpMode) -> ast::Crate {
    if let PpmSource(PpmEveryBodyLoops) = ppm {
        let mut fold = ReplaceBodyWithLoop::new();
        fold.fold_crate(krate)
    } else {
        krate
    }
}

fn get_source(input: &Input, sess: &Session) -> (Vec<u8>, String) {
    let src_name = driver::source_name(input);
    let src = sess.codemap()
        .get_filemap(&src_name)
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
    let dep_graph = DepGraph::new(false);
    let _ignore = dep_graph.in_ignore();

    let (src, src_name) = get_source(input, sess);

    let mut rdr = &*src;
    let mut out = Vec::new();

    if let PpmSource(s) = ppm {
        // Silently ignores an identified node.
        let out: &mut Write = &mut out;
        s.call_with_pp_support(sess, None, box out, |annotation, out| {
                debug!("pretty printing source code {:?}", s);
                let sess = annotation.sess();
                pprust::print_crate(sess.codemap(),
                                    &sess.parse_sess,
                                    krate,
                                    src_name.to_string(),
                                    &mut rdr,
                                    out,
                                    annotation.pp_ann(),
                                    false)
            })
            .unwrap()
    } else {
        unreachable!();
    };

    write_output(out, ofile);
}

pub fn print_after_hir_lowering<'tcx, 'a: 'tcx>(sess: &'a Session,
                                                ast_map: &hir_map::Map<'tcx>,
                                                analysis: &ty::CrateAnalysis<'tcx>,
                                                resolutions: &Resolutions,
                                                input: &Input,
                                                krate: &ast::Crate,
                                                crate_name: &str,
                                                ppm: PpMode,
                                                arena: &'tcx DroplessArena,
                                                arenas: &'tcx GlobalArenas<'tcx>,
                                                opt_uii: Option<UserIdentifiedItem>,
                                                ofile: Option<&Path>) {
    let dep_graph = DepGraph::new(false);
    let _ignore = dep_graph.in_ignore();

    if ppm.needs_analysis() {
        print_with_analysis(sess,
                            ast_map,
                            analysis,
                            resolutions,
                            crate_name,
                            arena,
                            arenas,
                            ppm,
                            opt_uii,
                            ofile);
        return;
    }

    let (src, src_name) = get_source(input, sess);

    let mut rdr = &src[..];
    let mut out = Vec::new();

    match (ppm, opt_uii) {
            (PpmSource(s), _) => {
                // Silently ignores an identified node.
                let out: &mut Write = &mut out;
                s.call_with_pp_support(sess, Some(ast_map), box out, |annotation, out| {
                    debug!("pretty printing source code {:?}", s);
                    let sess = annotation.sess();
                    pprust::print_crate(sess.codemap(),
                                        &sess.parse_sess,
                                        krate,
                                        src_name.to_string(),
                                        &mut rdr,
                                        out,
                                        annotation.pp_ann(),
                                        true)
                })
            }

            (PpmHir(s), None) => {
                let out: &mut Write = &mut out;
                s.call_with_pp_support_hir(sess,
                                           ast_map,
                                           analysis,
                                           resolutions,
                                           arena,
                                           arenas,
                                           crate_name,
                                           box out,
                                           |annotation, out, krate| {
                    debug!("pretty printing source code {:?}", s);
                    let sess = annotation.sess();
                    pprust_hir::print_crate(sess.codemap(),
                                            &sess.parse_sess,
                                            krate,
                                            src_name.to_string(),
                                            &mut rdr,
                                            out,
                                            annotation.pp_ann(),
                                            true)
                })
            }

            (PpmHir(s), Some(uii)) => {
                let out: &mut Write = &mut out;
                s.call_with_pp_support_hir(sess,
                                           ast_map,
                                           analysis,
                                           resolutions,
                                           arena,
                                           arenas,
                                           crate_name,
                                           (out, uii),
                                           |annotation, (out, uii), _| {
                    debug!("pretty printing source code {:?}", s);
                    let sess = annotation.sess();
                    let ast_map = annotation.ast_map().expect("--unpretty missing HIR map");
                    let mut pp_state = pprust_hir::State::new_from_input(sess.codemap(),
                                                                         &sess.parse_sess,
                                                                         src_name.to_string(),
                                                                         &mut rdr,
                                                                         box out,
                                                                         annotation.pp_ann(),
                                                                         true);
                    for node_id in uii.all_matching_node_ids(ast_map) {
                        let node = ast_map.get(node_id);
                        pp_state.print_node(node)?;
                        pp::space(&mut pp_state.s)?;
                        let path = annotation.node_path(node_id)
                            .expect("--unpretty missing node paths");
                        pp_state.synth_comment(path)?;
                        pp::hardbreak(&mut pp_state.s)?;
                    }
                    pp::eof(&mut pp_state.s)
                })
            }
            _ => unreachable!(),
        }
        .unwrap();

    write_output(out, ofile);
}

// In an ideal world, this would be a public function called by the driver after
// analsysis is performed. However, we want to call `phase_3_run_analysis_passes`
// with a different callback than the standard driver, so that isn't easy.
// Instead, we call that function ourselves.
fn print_with_analysis<'tcx, 'a: 'tcx>(sess: &'a Session,
                                       ast_map: &hir_map::Map<'tcx>,
                                       analysis: &ty::CrateAnalysis<'tcx>,
                                       resolutions: &Resolutions,
                                       crate_name: &str,
                                       arena: &'tcx DroplessArena,
                                       arenas: &'tcx GlobalArenas<'tcx>,
                                       ppm: PpMode,
                                       uii: Option<UserIdentifiedItem>,
                                       ofile: Option<&Path>) {
    let nodeid = if let Some(uii) = uii {
        debug!("pretty printing for {:?}", uii);
        Some(uii.to_one_node_id("--unpretty", sess, &ast_map))
    } else {
        debug!("pretty printing for whole crate");
        None
    };

    let mut out = Vec::new();

    abort_on_err(driver::phase_3_run_analysis_passes(sess,
                                                     ast_map.clone(),
                                                     analysis.clone(),
                                                     resolutions.clone(),
                                                     arena,
                                                     arenas,
                                                     crate_name,
                                                     |tcx, _, _, _| {
        match ppm {
            PpmMir | PpmMirCFG => {
                if let Some(nodeid) = nodeid {
                    let def_id = tcx.map.local_def_id(nodeid);
                    match ppm {
                        PpmMir => write_mir_pretty(tcx, iter::once(def_id), &mut out),
                        PpmMirCFG => write_mir_graphviz(tcx, iter::once(def_id), &mut out),
                        _ => unreachable!(),
                    }?;
                } else {
                    match ppm {
                        PpmMir => {
                            write_mir_pretty(tcx, tcx.mir_map.borrow().keys().into_iter(), &mut out)
                        }
                        PpmMirCFG => {
                            write_mir_graphviz(tcx,
                                               tcx.mir_map.borrow().keys().into_iter(),
                                               &mut out)
                        }
                        _ => unreachable!(),
                    }?;
                }
                Ok(())
            }
            PpmFlowGraph(mode) => {
                let nodeid =
                    nodeid.expect("`pretty flowgraph=..` needs NodeId (int) or unique path \
                                   suffix (b::c::d)");
                let node = tcx.map.find(nodeid).unwrap_or_else(|| {
                    tcx.sess.fatal(&format!("--pretty flowgraph couldn't find id: {}", nodeid))
                });

                match blocks::Code::from_node(&tcx.map, nodeid) {
                    Some(code) => {
                        let variants = gather_flowgraph_variants(tcx.sess);

                        let out: &mut Write = &mut out;

                        print_flowgraph(variants, tcx, code, mode, out)
                    }
                    None => {
                        let message = format!("--pretty=flowgraph needs block, fn, or method; \
                                               got {:?}",
                                              node);

                        tcx.sess.span_fatal(tcx.map.span(nodeid), &message)
                    }
                }
            }
            _ => unreachable!(),
        }
    }),
                 sess)
        .unwrap();

    write_output(out, ofile);
}
