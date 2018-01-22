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
use rustc_driver::{self, driver, target_features, abort_on_err};
use rustc::session::{self, config};
use rustc::hir::def_id::DefId;
use rustc::hir::def::Def;
use rustc::middle::privacy::AccessLevels;
use rustc::ty::{self, TyCtxt, AllArenas};
use rustc::hir::map as hir_map;
use rustc::lint;
use rustc::util::nodemap::FxHashMap;
use rustc_resolve as resolve;
use rustc_metadata::creader::CrateLoader;
use rustc_metadata::cstore::CStore;

use syntax::ast::NodeId;
use syntax::codemap;
use syntax::feature_gate::UnstableFeatures;
use errors;
use errors::emitter::ColorConfig;

use std::cell::{RefCell, Cell};
use std::mem;
use std::rc::Rc;
use std::path::PathBuf;

use visit_ast::RustdocVisitor;
use clean;
use clean::Clean;
use html::markdown::RenderType;
use html::render::RenderInfo;

pub use rustc::session::config::Input;
pub use rustc::session::search_paths::SearchPaths;

pub type ExternalPaths = FxHashMap<DefId, (Vec<String>, clean::TypeKind)>;

pub struct DocContext<'a, 'tcx: 'a, 'rcx: 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub resolver: &'a RefCell<resolve::Resolver<'rcx>>,
    /// The stack of module NodeIds up till this point
    pub mod_ids: RefCell<Vec<NodeId>>,
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
    /// Which markdown renderer to use when extracting links.
    pub render_type: RenderType,

    // The current set of type and lifetime substitutions,
    // for expanding type aliases at the HIR level:

    /// Table type parameter definition -> substituted type
    pub ty_substs: RefCell<FxHashMap<Def, clean::Type>>,
    /// Table node id of lifetime parameter definition -> substituted lifetime
    pub lt_substs: RefCell<FxHashMap<DefId, clean::Lifetime>>,
}

impl<'a, 'tcx, 'rcx> DocContext<'a, 'tcx, 'rcx> {
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
                force_unstable_if_unmarked: bool,
                render_type: RenderType) -> (clean::Crate, RenderInfo)
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

    let mut sess = session::build_session_(
        sessopts, cpath, diagnostic_handler, codemap,
    );
    let trans = rustc_driver::get_trans(&sess);
    let cstore = Rc::new(CStore::new(trans.metadata_loader()));
    rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));

    let mut cfg = config::build_configuration(&sess, config::parse_cfgspecs(cfgs));
    target_features::add_configuration(&mut cfg, &sess, &*trans);
    sess.parse_sess.config = cfg;

    let control = &driver::CompileController::basic();

    let krate = panictry!(driver::phase_1_parse_input(control, &sess, &input));

    let name = ::rustc_trans_utils::link::find_crate_name(Some(&sess), &krate.attrs, &input);

    let mut crate_loader = CrateLoader::new(&sess, &cstore, &name);

    let resolver_arenas = resolve::Resolver::arenas();
    let result = driver::phase_2_configure_and_expand_inner(&sess,
                                                      &cstore,
                                                      krate,
                                                      None,
                                                      &name,
                                                      None,
                                                      resolve::MakeGlobMap::No,
                                                      &resolver_arenas,
                                                      &mut crate_loader,
                                                      |_| Ok(()));
    let driver::InnerExpansionResult {
        mut hir_forest,
        resolver,
        ..
    } = abort_on_err(result, &sess);

    // We need to hold on to the complete resolver, so we clone everything
    // for the analysis passes to use. Suboptimal, but necessary in the
    // current architecture.
    let defs = resolver.definitions.clone();
    let resolutions = ty::Resolutions {
        freevars: resolver.freevars.clone(),
        export_map: resolver.export_map.clone(),
        trait_map: resolver.trait_map.clone(),
        maybe_unused_trait_imports: resolver.maybe_unused_trait_imports.clone(),
        maybe_unused_extern_crates: resolver.maybe_unused_extern_crates.clone(),
    };
    let analysis = ty::CrateAnalysis {
        access_levels: Rc::new(AccessLevels::default()),
        name: name.to_string(),
        glob_map: if resolver.make_glob_map { Some(resolver.glob_map.clone()) } else { None },
    };

    let arenas = AllArenas::new();
    let hir_map = hir_map::map_crate(&sess, &*cstore, &mut hir_forest, &defs);
    let output_filenames = driver::build_output_filenames(&input,
                                                          &None,
                                                          &None,
                                                          &[],
                                                          &sess);

    let resolver = RefCell::new(resolver);

    abort_on_err(driver::phase_3_run_analysis_passes(&*trans,
                                                     control,
                                                     &sess,
                                                     &*cstore,
                                                     hir_map,
                                                     analysis,
                                                     resolutions,
                                                     &arenas,
                                                     &name,
                                                     &output_filenames,
                                                     |tcx, analysis, _, result| {
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
            resolver: &resolver,
            populated_all_crate_impls: Cell::new(false),
            access_levels: RefCell::new(access_levels),
            external_traits: Default::default(),
            renderinfo: Default::default(),
            render_type,
            ty_substs: Default::default(),
            lt_substs: Default::default(),
            mod_ids: Default::default(),
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
