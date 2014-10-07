// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::driver::{config, driver, session};
use rustc::middle::{privacy, ty};
use rustc::lint;
use rustc::back::link;

use syntax::{ast, ast_map, codemap, diagnostic};

use std::cell::RefCell;
use std::os;
use std::collections::{HashMap, HashSet};
use arena::TypedArena;

use visit_ast::RustdocVisitor;
use clean;
use clean::Clean;

/// Are we generating documentation (`Typed`) or tests (`NotTyped`)?
pub enum MaybeTyped<'tcx> {
    Typed(ty::ctxt<'tcx>),
    NotTyped(session::Session)
}

pub type ExternalPaths = RefCell<Option<HashMap<ast::DefId,
                                                (Vec<String>, clean::TypeKind)>>>;

pub struct DocContext<'tcx> {
    pub krate: &'tcx ast::Crate,
    pub maybe_typed: MaybeTyped<'tcx>,
    pub src: Path,
    pub external_paths: ExternalPaths,
    pub external_traits: RefCell<Option<HashMap<ast::DefId, clean::Trait>>>,
    pub external_typarams: RefCell<Option<HashMap<ast::DefId, String>>>,
    pub inlined: RefCell<Option<HashSet<ast::DefId>>>,
    pub populated_crate_impls: RefCell<HashSet<ast::CrateNum>>,
}

impl<'tcx> DocContext<'tcx> {
    pub fn sess<'a>(&'a self) -> &'a session::Session {
        match self.maybe_typed {
            Typed(ref tcx) => &tcx.sess,
            NotTyped(ref sess) => sess
        }
    }

    pub fn tcx_opt<'a>(&'a self) -> Option<&'a ty::ctxt<'tcx>> {
        match self.maybe_typed {
            Typed(ref tcx) => Some(tcx),
            NotTyped(_) => None
        }
    }

    pub fn tcx<'a>(&'a self) -> &'a ty::ctxt<'tcx> {
        let tcx_opt = self.tcx_opt();
        tcx_opt.expect("tcx not present")
    }
}

pub struct CrateAnalysis {
    pub exported_items: privacy::ExportedItems,
    pub public_items: privacy::PublicItems,
    pub external_paths: ExternalPaths,
    pub external_traits: RefCell<Option<HashMap<ast::DefId, clean::Trait>>>,
    pub external_typarams: RefCell<Option<HashMap<ast::DefId, String>>>,
    pub inlined: RefCell<Option<HashSet<ast::DefId>>>,
}

pub type Externs = HashMap<String, Vec<String>>;

pub fn run_core(libs: Vec<Path>, cfgs: Vec<String>, externs: Externs,
                cpath: &Path, triple: Option<String>)
                -> (clean::Crate, CrateAnalysis) {

    // Parse, resolve, and typecheck the given crate.

    let input = driver::FileInput(cpath.clone());

    let warning_lint = lint::builtin::WARNINGS.name_lower();

    let sessopts = config::Options {
        maybe_sysroot: Some(os::self_exe_path().unwrap().dir_path()),
        addl_lib_search_paths: RefCell::new(libs),
        crate_types: vec!(config::CrateTypeRlib),
        lint_opts: vec!((warning_lint, lint::Allow)),
        externs: externs,
        target_triple: triple.unwrap_or(driver::host_triple().to_string()),
        cfg: config::parse_cfgspecs(cfgs),
        ..config::basic_options().clone()
    };


    let codemap = codemap::CodeMap::new();
    let diagnostic_handler = diagnostic::default_handler(diagnostic::Auto, None);
    let span_diagnostic_handler =
        diagnostic::mk_span_handler(diagnostic_handler, codemap);

    let sess = session::build_session_(sessopts,
                                       Some(cpath.clone()),
                                       span_diagnostic_handler);

    let cfg = config::build_configuration(&sess);

    let krate = driver::phase_1_parse_input(&sess, cfg, &input);

    let name = link::find_crate_name(Some(&sess), krate.attrs.as_slice(),
                                     &input);

    let krate = driver::phase_2_configure_and_expand(&sess, krate, name.as_slice(), None)
                    .expect("phase_2_configure_and_expand aborted in rustdoc!");

    let mut forest = ast_map::Forest::new(krate);
    let ast_map = driver::assign_node_ids_and_map(&sess, &mut forest);

    let type_arena = TypedArena::new();
    let driver::CrateAnalysis {
        exported_items, public_items, ty_cx, ..
    } = driver::phase_3_run_analysis_passes(sess, ast_map, &type_arena, name);

    let ctxt = DocContext {
        krate: ty_cx.map.krate(),
        maybe_typed: Typed(ty_cx),
        src: cpath.clone(),
        external_traits: RefCell::new(Some(HashMap::new())),
        external_typarams: RefCell::new(Some(HashMap::new())),
        external_paths: RefCell::new(Some(HashMap::new())),
        inlined: RefCell::new(Some(HashSet::new())),
        populated_crate_impls: RefCell::new(HashSet::new()),
    };
    debug!("crate: {:?}", ctxt.krate);

    let analysis = CrateAnalysis {
        exported_items: exported_items,
        public_items: public_items,
        external_paths: RefCell::new(None),
        external_traits: RefCell::new(None),
        external_typarams: RefCell::new(None),
        inlined: RefCell::new(None),
    };

    let krate = {
        let mut v = RustdocVisitor::new(&ctxt, Some(&analysis));
        v.visit(ctxt.krate);
        v.clean(&ctxt)
    };

    let external_paths = ctxt.external_paths.borrow_mut().take();
    *analysis.external_paths.borrow_mut() = external_paths;
    let map = ctxt.external_traits.borrow_mut().take();
    *analysis.external_traits.borrow_mut() = map;
    let map = ctxt.external_typarams.borrow_mut().take();
    *analysis.external_typarams.borrow_mut() = map;
    let map = ctxt.inlined.borrow_mut().take();
    *analysis.inlined.borrow_mut() = map;
    (krate, analysis)
}
