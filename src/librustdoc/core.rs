// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
pub use self::MaybeTyped::*;

use rustc_lint;
use rustc_driver::{driver, target_features, abort_on_err};
use rustc::dep_graph::DepGraph;
use rustc::session::{self, config};
use rustc::hir::def_id::DefId;
use rustc::middle::privacy::AccessLevels;
use rustc::ty::{self, TyCtxt};
use rustc::hir::map as hir_map;
use rustc::lint;
use rustc_trans::back::link;
use rustc_resolve as resolve;
use rustc_metadata::cstore::CStore;
use rustc_metadata::creader::LocalCrateReader;

use syntax::{ast, codemap, errors};
use syntax::errors::emitter::ColorConfig;
use syntax::feature_gate::UnstableFeatures;
use syntax::parse::token;

use std::cell::{RefCell, Cell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use visit_ast::RustdocVisitor;
use clean;
use clean::Clean;
use html::render::RenderInfo;

pub use rustc::session::config::Input;
pub use rustc::session::search_paths::SearchPaths;

/// Are we generating documentation (`Typed`) or tests (`NotTyped`)?
pub enum MaybeTyped<'a, 'tcx: 'a> {
    Typed(&'a TyCtxt<'tcx>),
    NotTyped(&'a session::Session)
}

pub type Externs = HashMap<String, Vec<String>>;
pub type ExternalPaths = HashMap<DefId, (Vec<String>, clean::TypeKind)>;

pub struct DocContext<'a, 'tcx: 'a> {
    pub map: &'a hir_map::Map<'tcx>,
    pub maybe_typed: MaybeTyped<'a, 'tcx>,
    pub input: Input,
    pub populated_crate_impls: RefCell<HashSet<ast::CrateNum>>,
    pub deref_trait_did: Cell<Option<DefId>>,
    // Note that external items for which `doc(hidden)` applies to are shown as
    // non-reachable while local items aren't. This is because we're reusing
    // the access levels from crateanalysis.
    /// Later on moved into `clean::Crate`
    pub access_levels: RefCell<AccessLevels<DefId>>,
    /// Later on moved into `html::render::CACHE_KEY`
    pub renderinfo: RefCell<RenderInfo>,
    /// Later on moved through `clean::Crate` into `html::render::CACHE_KEY`
    pub external_traits: RefCell<HashMap<DefId, clean::Trait>>,
}

impl<'b, 'tcx> DocContext<'b, 'tcx> {
    pub fn sess<'a>(&'a self) -> &'a session::Session {
        match self.maybe_typed {
            Typed(tcx) => &tcx.sess,
            NotTyped(ref sess) => sess
        }
    }

    pub fn tcx_opt<'a>(&'a self) -> Option<&'a TyCtxt<'tcx>> {
        match self.maybe_typed {
            Typed(tcx) => Some(tcx),
            NotTyped(_) => None
        }
    }

    pub fn tcx<'a>(&'a self) -> &'a TyCtxt<'tcx> {
        let tcx_opt = self.tcx_opt();
        tcx_opt.expect("tcx not present")
    }
}

pub trait DocAccessLevels {
    fn is_doc_reachable(&self, DefId) -> bool;
}

impl DocAccessLevels for AccessLevels<DefId> {
    fn is_doc_reachable(&self, did: DefId) -> bool {
        self.is_public(did)
    }
}


pub fn run_core(search_paths: SearchPaths,
                cfgs: Vec<String>,
                externs: Externs,
                input: Input,
                triple: Option<String>) -> (clean::Crate, RenderInfo)
{
    // Parse, resolve, and typecheck the given crate.

    let cpath = match input {
        Input::File(ref p) => Some(p.clone()),
        _ => None
    };

    let warning_lint = lint::builtin::WARNINGS.name_lower();

    let sessopts = config::Options {
        maybe_sysroot: None,
        search_paths: search_paths,
        crate_types: vec!(config::CrateTypeRlib),
        lint_opts: vec!((warning_lint, lint::Allow)),
        lint_cap: Some(lint::Allow),
        externs: externs,
        target_triple: triple.unwrap_or(config::host_triple().to_string()),
        cfg: config::parse_cfgspecs(cfgs),
        // Ensure that rustdoc works even if rustc is feature-staged
        unstable_features: UnstableFeatures::Allow,
        ..config::basic_options().clone()
    };

    let codemap = Rc::new(codemap::CodeMap::new());
    let diagnostic_handler = errors::Handler::with_tty_emitter(ColorConfig::Auto,
                                                               None,
                                                               true,
                                                               false,
                                                               codemap.clone());

    let cstore = Rc::new(CStore::new(token::get_ident_interner()));
    let sess = session::build_session_(sessopts, cpath, diagnostic_handler,
                                       codemap, cstore.clone());
    rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));

    let mut cfg = config::build_configuration(&sess);
    target_features::add_configuration(&mut cfg, &sess);

    let krate = panictry!(driver::phase_1_parse_input(&sess, cfg, &input));

    let name = link::find_crate_name(Some(&sess), &krate.attrs,
                                     &input);

    let krate = driver::phase_2_configure_and_expand(&sess, &cstore, krate, &name, None)
                    .expect("phase_2_configure_and_expand aborted in rustdoc!");

    let krate = driver::assign_node_ids(&sess, krate);
    let dep_graph = DepGraph::new(false);

    let defs = &RefCell::new(hir_map::collect_definitions(&krate));
    LocalCrateReader::new(&sess, &cstore, &defs, &krate, &name).read_crates(&dep_graph);

    // Lower ast -> hir and resolve.
    let (analysis, resolutions, mut hir_forest) = {
        let defs = &mut *defs.borrow_mut();
        driver::lower_and_resolve(&sess, &name, defs, &krate, dep_graph, resolve::MakeGlobMap::No)
    };

    let arenas = ty::CtxtArenas::new();
    let hir_map = hir_map::map_crate(&mut hir_forest, defs);

    abort_on_err(driver::phase_3_run_analysis_passes(&sess,
                                                     hir_map,
                                                     analysis,
                                                     resolutions,
                                                     &arenas,
                                                     &name,
                                                     |tcx, _, analysis, result| {
        // Return if the driver hit an err (in `result`)
        if let Err(_) = result {
            return None
        }

        let _ignore = tcx.dep_graph.in_ignore();
        let ty::CrateAnalysis { access_levels, .. } = analysis;

        // Convert from a NodeId set to a DefId set since we don't always have easy access
        // to the map from defid -> nodeid
        let access_levels = AccessLevels {
            map: access_levels.map.into_iter()
                                  .map(|(k, v)| (tcx.map.local_def_id(k), v))
                                  .collect()
        };

        let ctxt = DocContext {
            map: &tcx.map,
            maybe_typed: Typed(tcx),
            input: input,
            populated_crate_impls: RefCell::new(HashSet::new()),
            deref_trait_did: Cell::new(None),
            access_levels: RefCell::new(access_levels),
            external_traits: RefCell::new(HashMap::new()),
            renderinfo: RefCell::new(Default::default()),
        };
        debug!("crate: {:?}", ctxt.map.krate());

        let krate = {
            let mut v = RustdocVisitor::new(&ctxt);
            v.visit(ctxt.map.krate());
            v.clean(&ctxt)
        };

        Some((krate, ctxt.renderinfo.into_inner()))
    }), &sess).unwrap()
}
