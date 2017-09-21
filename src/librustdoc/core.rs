// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc_lint;
use rustc_driver::{driver, target_features, abort_on_err};
use rustc_driver::pretty::ReplaceBodyWithLoop;
use rustc::session::{self, config};
use rustc::hir::def_id::DefId;
use rustc::hir::def::Def;
use rustc::middle::privacy::AccessLevels;
use rustc::ty::{self, TyCtxt, GlobalArenas};
use rustc::hir::map as hir_map;
use rustc::lint;
use rustc::util::nodemap::FxHashMap;
use rustc_trans;
use rustc_trans::back::link;
use rustc_resolve as resolve;
use rustc_metadata::cstore::CStore;

use syntax::codemap;
use syntax::feature_gate::UnstableFeatures;
use syntax::fold::Folder;
use errors;
use errors::emitter::ColorConfig;

use std::cell::{RefCell, Cell};
use std::mem;
use std::rc::Rc;
use std::path::PathBuf;

use visit_ast::RustdocVisitor;
use clean;
use clean::Clean;
use html::render::RenderInfo;
use arena::DroplessArena;

pub use rustc::session::config::Input;
pub use rustc::session::search_paths::SearchPaths;

pub type ExternalPaths = FxHashMap<DefId, (Vec<String>, clean::TypeKind)>;

pub struct DocContext<'a, 'tcx: 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub populated_all_crate_impls: Cell<bool>,
    // Note that external items for which `doc(hidden)` applies to are shown as
    // non-reachable while local items aren't. This is because we're reusing
    // the access levels from crateanalysis.
    /// Later on moved into `clean::Crate`
    pub access_levels: RefCell<AccessLevels<DefId>>,
    /// Later on moved into `html::render::CACHE_KEY`
    pub renderinfo: RefCell<RenderInfo>,
    /// Later on moved through `clean::Crate` into `html::render::CACHE_KEY`
    pub external_traits: RefCell<FxHashMap<DefId, clean::Trait>>,

    // The current set of type and lifetime substitutions,
    // for expanding type aliases at the HIR level:

    /// Table type parameter definition -> substituted type
    pub ty_substs: RefCell<FxHashMap<Def, clean::Type>>,
    /// Table node id of lifetime parameter definition -> substituted lifetime
    pub lt_substs: RefCell<FxHashMap<DefId, clean::Lifetime>>,
}

impl<'a, 'tcx> DocContext<'a, 'tcx> {
    pub fn sess(&self) -> &session::Session {
        &self.tcx.sess
    }

    /// Call the closure with the given parameters set as
    /// the substitutions for a type alias' RHS.
    pub fn enter_alias<F, R>(&self,
                             ty_substs: FxHashMap<Def, clean::Type>,
                             lt_substs: FxHashMap<DefId, clean::Lifetime>,
                             f: F) -> R
    where F: FnOnce() -> R {
        let (old_tys, old_lts) =
            (mem::replace(&mut *self.ty_substs.borrow_mut(), ty_substs),
             mem::replace(&mut *self.lt_substs.borrow_mut(), lt_substs));
        let r = f();
        *self.ty_substs.borrow_mut() = old_tys;
        *self.lt_substs.borrow_mut() = old_lts;
        r
    }
}

pub trait DocAccessLevels {
    fn is_doc_reachable(&self, did: DefId) -> bool;
}

impl DocAccessLevels for AccessLevels<DefId> {
    fn is_doc_reachable(&self, did: DefId) -> bool {
        self.is_public(did)
    }
}


pub fn run_core(search_paths: SearchPaths,
                cfgs: Vec<String>,
                externs: config::Externs,
                input: Input,
                triple: Option<String>,
                maybe_sysroot: Option<PathBuf>,
                allow_warnings: bool,
                force_unstable_if_unmarked: bool) -> (clean::Crate, RenderInfo)
{
    // Parse, resolve, and typecheck the given crate.

    let cpath = match input {
        Input::File(ref p) => Some(p.clone()),
        _ => None
    };

    let warning_lint = lint::builtin::WARNINGS.name_lower();

    let sessopts = config::Options {
        maybe_sysroot,
        search_paths,
        crate_types: vec![config::CrateTypeRlib],
        lint_opts: if !allow_warnings { vec![(warning_lint, lint::Allow)] } else { vec![] },
        lint_cap: Some(lint::Allow),
        externs,
        target_triple: triple.unwrap_or(config::host_triple().to_string()),
        // Ensure that rustdoc works even if rustc is feature-staged
        unstable_features: UnstableFeatures::Allow,
        actually_rustdoc: true,
        debugging_opts: config::DebuggingOptions {
            force_unstable_if_unmarked,
            ..config::basic_debugging_options()
        },
        ..config::basic_options().clone()
    };

    let codemap = Rc::new(codemap::CodeMap::new(sessopts.file_path_mapping()));
    let diagnostic_handler = errors::Handler::with_tty_emitter(ColorConfig::Auto,
                                                               true,
                                                               false,
                                                               Some(codemap.clone()));

    let cstore = Rc::new(CStore::new(box rustc_trans::LlvmMetadataLoader));
    let mut sess = session::build_session_(
        sessopts, cpath, diagnostic_handler, codemap,
    );
    rustc_trans::init(&sess);
    rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));

    let mut cfg = config::build_configuration(&sess, config::parse_cfgspecs(cfgs));
    target_features::add_configuration(&mut cfg, &sess);
    sess.parse_sess.config = cfg;

    let krate = panictry!(driver::phase_1_parse_input(&driver::CompileController::basic(),
                                                      &sess,
                                                      &input));
    let krate = ReplaceBodyWithLoop::new().fold_crate(krate);

    let name = link::find_crate_name(Some(&sess), &krate.attrs, &input);

    let driver::ExpansionResult { defs, analysis, resolutions, mut hir_forest, .. } = {
        let result = driver::phase_2_configure_and_expand(&sess,
                                                          &cstore,
                                                          krate,
                                                          None,
                                                          &name,
                                                          None,
                                                          resolve::MakeGlobMap::No,
                                                          |_| Ok(()));
        abort_on_err(result, &sess)
    };

    let arena = DroplessArena::new();
    let arenas = GlobalArenas::new();
    let hir_map = hir_map::map_crate(&mut hir_forest, &defs);
    let output_filenames = driver::build_output_filenames(&input,
                                                          &None,
                                                          &None,
                                                          &[],
                                                          &sess);

    abort_on_err(driver::phase_3_run_analysis_passes(&sess,
                                                     &*cstore,
                                                     hir_map,
                                                     analysis,
                                                     resolutions,
                                                     &arena,
                                                     &arenas,
                                                     &name,
                                                     &output_filenames,
                                                     |tcx, analysis, _, _, result| {
        if let Err(_) = result {
            sess.fatal("Compilation failed, aborting rustdoc");
        }

        let ty::CrateAnalysis { access_levels, .. } = analysis;

        // Convert from a NodeId set to a DefId set since we don't always have easy access
        // to the map from defid -> nodeid
        let access_levels = AccessLevels {
            map: access_levels.map.iter()
                                  .map(|(&k, &v)| (tcx.hir.local_def_id(k), v))
                                  .collect()
        };

        let ctxt = DocContext {
            tcx,
            populated_all_crate_impls: Cell::new(false),
            access_levels: RefCell::new(access_levels),
            external_traits: Default::default(),
            renderinfo: Default::default(),
            ty_substs: Default::default(),
            lt_substs: Default::default(),
        };
        debug!("crate: {:?}", tcx.hir.krate());

        let krate = {
            let mut v = RustdocVisitor::new(&*cstore, &ctxt);
            v.visit(tcx.hir.krate());
            v.clean(&ctxt)
        };

        (krate, ctxt.renderinfo.into_inner())
    }), &sess)
}
