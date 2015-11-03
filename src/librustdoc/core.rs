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
use rustc_driver::{driver, target_features};
use rustc::session::{self, config};
use rustc::middle::def_id::DefId;
use rustc::middle::ty;
use rustc::front::map as hir_map;
use rustc::lint;
use rustc::util::nodemap::DefIdSet;
use rustc_trans::back::link;
use rustc_resolve as resolve;
use rustc_front::lowering::{lower_crate, LoweringContext};

use syntax::{ast, codemap, diagnostic};
use syntax::feature_gate::UnstableFeatures;

use std::cell::{RefCell, Cell};
use std::collections::{HashMap, HashSet};
use std::default::Default;

use visit_ast::RustdocVisitor;
use clean;
use clean::Clean;

pub use rustc::session::config::Input;
pub use rustc::session::search_paths::SearchPaths;

/// Are we generating documentation (`Typed`) or tests (`NotTyped`)?
pub enum MaybeTyped<'a, 'tcx: 'a> {
    Typed(&'a ty::ctxt<'tcx>),
    NotTyped(&'a session::Session)
}

pub type ExternalPaths = RefCell<Option<HashMap<DefId,
                                                (Vec<String>, clean::TypeKind)>>>;

pub struct DocContext<'a, 'tcx: 'a> {
    pub map: &'a hir_map::Map<'tcx>,
    pub maybe_typed: MaybeTyped<'a, 'tcx>,
    pub input: Input,
    pub external_paths: ExternalPaths,
    pub external_traits: RefCell<Option<HashMap<DefId, clean::Trait>>>,
    pub external_typarams: RefCell<Option<HashMap<DefId, String>>>,
    pub inlined: RefCell<Option<HashSet<DefId>>>,
    pub populated_crate_impls: RefCell<HashSet<ast::CrateNum>>,
    pub deref_trait_did: Cell<Option<DefId>>,
}

impl<'b, 'tcx> DocContext<'b, 'tcx> {
    pub fn sess<'a>(&'a self) -> &'a session::Session {
        match self.maybe_typed {
            Typed(tcx) => &tcx.sess,
            NotTyped(ref sess) => sess
        }
    }

    pub fn tcx_opt<'a>(&'a self) -> Option<&'a ty::ctxt<'tcx>> {
        match self.maybe_typed {
            Typed(tcx) => Some(tcx),
            NotTyped(_) => None
        }
    }

    pub fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        let tcx_opt = self.tcx_opt();
        tcx_opt.expect("tcx not present")
    }
}

pub struct CrateAnalysis {
    pub exported_items: DefIdSet,
    pub public_items: DefIdSet,
    pub external_paths: ExternalPaths,
    pub external_typarams: RefCell<Option<HashMap<DefId, String>>>,
    pub inlined: RefCell<Option<HashSet<DefId>>>,
    pub deref_trait_did: Option<DefId>,
}

pub type Externs = HashMap<String, Vec<String>>;

pub fn run_core(search_paths: SearchPaths, cfgs: Vec<String>, externs: Externs,
                input: Input, triple: Option<String>)
                -> (clean::Crate, CrateAnalysis) {

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
        externs: externs,
        target_triple: triple.unwrap_or(config::host_triple().to_string()),
        cfg: config::parse_cfgspecs(cfgs),
        // Ensure that rustdoc works even if rustc is feature-staged
        unstable_features: UnstableFeatures::Allow,
        ..config::basic_options().clone()
    };

    let codemap = codemap::CodeMap::new();
    let diagnostic_handler = diagnostic::Handler::new(Default::default(), None, true);
    let span_diagnostic_handler =
        diagnostic::SpanHandler::new(diagnostic_handler, codemap);

    let sess = session::build_session_(sessopts, cpath,
                                       span_diagnostic_handler);
    rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));

    let mut cfg = config::build_configuration(&sess);
    target_features::add_configuration(&mut cfg, &sess);

    let krate = driver::phase_1_parse_input(&sess, cfg, &input);

    let name = link::find_crate_name(Some(&sess), &krate.attrs,
                                     &input);

    let krate = driver::phase_2_configure_and_expand(&sess, krate, &name, None)
                    .expect("phase_2_configure_and_expand aborted in rustdoc!");

    let krate = driver::assign_node_ids(&sess, krate);
    // Lower ast -> hir.
    let lcx = LoweringContext::new(&sess, Some(&krate));
    let mut hir_forest = hir_map::Forest::new(lower_crate(&lcx, &krate));
    let arenas = ty::CtxtArenas::new();
    let hir_map = driver::make_map(&sess, &mut hir_forest);

    driver::phase_3_run_analysis_passes(&sess,
                                        hir_map,
                                        &arenas,
                                        &name,
                                        resolve::MakeGlobMap::No,
                                        |tcx, analysis| {
        let ty::CrateAnalysis { exported_items, public_items, .. } = analysis;

        // Convert from a NodeId set to a DefId set since we don't always have easy access
        // to the map from defid -> nodeid
        let exported_items: DefIdSet =
            exported_items.into_iter()
                          .map(|n| tcx.map.local_def_id(n))
                          .collect();
        let public_items: DefIdSet =
            public_items.into_iter()
                        .map(|n| tcx.map.local_def_id(n))
                        .collect();

        let ctxt = DocContext {
            map: &tcx.map,
            maybe_typed: Typed(tcx),
            input: input,
            external_traits: RefCell::new(Some(HashMap::new())),
            external_typarams: RefCell::new(Some(HashMap::new())),
            external_paths: RefCell::new(Some(HashMap::new())),
            inlined: RefCell::new(Some(HashSet::new())),
            populated_crate_impls: RefCell::new(HashSet::new()),
            deref_trait_did: Cell::new(None),
        };
        debug!("crate: {:?}", ctxt.map.krate());

        let mut analysis = CrateAnalysis {
            exported_items: exported_items,
            public_items: public_items,
            external_paths: RefCell::new(None),
            external_typarams: RefCell::new(None),
            inlined: RefCell::new(None),
            deref_trait_did: None,
        };

        let krate = {
            let mut v = RustdocVisitor::new(&ctxt, Some(&analysis));
            v.visit(ctxt.map.krate());
            v.clean(&ctxt)
        };

        let external_paths = ctxt.external_paths.borrow_mut().take();
        *analysis.external_paths.borrow_mut() = external_paths;
        let map = ctxt.external_typarams.borrow_mut().take();
        *analysis.external_typarams.borrow_mut() = map;
        let map = ctxt.inlined.borrow_mut().take();
        *analysis.inlined.borrow_mut() = map;
        analysis.deref_trait_did = ctxt.deref_trait_did.get();
        (krate, analysis)
    })
}
