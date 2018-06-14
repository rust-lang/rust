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
use rustc::hir::def_id::{DefId, CrateNum};
use rustc::hir::def::Def;
use rustc::middle::cstore::CrateStore;
use rustc::middle::privacy::AccessLevels;
use rustc::ty::{self, TyCtxt, AllArenas};
use rustc::hir::map as hir_map;
use rustc::lint::{self, LintPass};
use rustc::session::config::ErrorOutputType;
use rustc::util::nodemap::{FxHashMap, FxHashSet};
use rustc_resolve as resolve;
use rustc_metadata::creader::CrateLoader;
use rustc_metadata::cstore::CStore;
use rustc_target::spec::TargetTriple;

use syntax::ast::NodeId;
use syntax::codemap;
use syntax::edition::Edition;
use syntax::feature_gate::UnstableFeatures;
use syntax::json::JsonEmitter;
use errors;
use errors::emitter::{Emitter, EmitterWriter};

use std::cell::{RefCell, Cell};
use std::mem;
use rustc_data_structures::sync::{self, Lrc};
use std::rc::Rc;
use std::path::PathBuf;

use visit_ast::RustdocVisitor;
use clean;
use clean::Clean;
use html::render::RenderInfo;

pub use rustc::session::config::{Input, CodegenOptions};
pub use rustc::session::search_paths::SearchPaths;

pub type ExternalPaths = FxHashMap<DefId, (Vec<String>, clean::TypeKind)>;

pub struct DocContext<'a, 'tcx: 'a, 'rcx: 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub resolver: &'a RefCell<resolve::Resolver<'rcx>>,
    /// The stack of module NodeIds up till this point
    pub mod_ids: RefCell<Vec<NodeId>>,
    pub crate_name: Option<String>,
    pub cstore: Rc<CrateStore>,
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
    /// Used while populating `external_traits` to ensure we don't process the same trait twice at
    /// the same time.
    pub active_extern_traits: RefCell<Vec<DefId>>,
    // The current set of type and lifetime substitutions,
    // for expanding type aliases at the HIR level:

    /// Table type parameter definition -> substituted type
    pub ty_substs: RefCell<FxHashMap<Def, clean::Type>>,
    /// Table node id of lifetime parameter definition -> substituted lifetime
    pub lt_substs: RefCell<FxHashMap<DefId, clean::Lifetime>>,
    /// Table DefId of `impl Trait` in argument position -> bounds
    pub impl_trait_bounds: RefCell<FxHashMap<DefId, Vec<clean::GenericBound>>>,
    pub send_trait: Option<DefId>,
    pub fake_def_ids: RefCell<FxHashMap<CrateNum, DefId>>,
    pub all_fake_def_ids: RefCell<FxHashSet<DefId>>,
    /// Maps (type_id, trait_id) -> auto trait impl
    pub generated_synthetics: RefCell<FxHashSet<(DefId, DefId)>>
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

/// Creates a new diagnostic `Handler` that can be used to emit warnings and errors.
///
/// If the given `error_format` is `ErrorOutputType::Json` and no `CodeMap` is given, a new one
/// will be created for the handler.
pub fn new_handler(error_format: ErrorOutputType, codemap: Option<Lrc<codemap::CodeMap>>)
    -> errors::Handler
{
    // rustdoc doesn't override (or allow to override) anything from this that is relevant here, so
    // stick to the defaults
    let sessopts = config::basic_options();
    let emitter: Box<dyn Emitter + sync::Send> = match error_format {
        ErrorOutputType::HumanReadable(color_config) => Box::new(
            EmitterWriter::stderr(
                color_config,
                codemap.map(|cm| cm as _),
                false,
                sessopts.debugging_opts.teach,
            ).ui_testing(sessopts.debugging_opts.ui_testing)
        ),
        ErrorOutputType::Json(pretty) => {
            let codemap = codemap.unwrap_or_else(
                || Lrc::new(codemap::CodeMap::new(sessopts.file_path_mapping())));
            Box::new(
                JsonEmitter::stderr(
                    None,
                    codemap,
                    pretty,
                ).ui_testing(sessopts.debugging_opts.ui_testing)
            )
        },
        ErrorOutputType::Short(color_config) => Box::new(
            EmitterWriter::stderr(
                color_config,
                codemap.map(|cm| cm as _),
                true,
                false)
        ),
    };

    errors::Handler::with_emitter_and_flags(
        emitter,
        errors::HandlerFlags {
            can_emit_warnings: true,
            treat_err_as_bug: false,
            external_macro_backtrace: false,
            ..Default::default()
        },
    )
}

pub fn run_core(search_paths: SearchPaths,
                cfgs: Vec<String>,
                externs: config::Externs,
                input: Input,
                triple: Option<TargetTriple>,
                maybe_sysroot: Option<PathBuf>,
                allow_warnings: bool,
                crate_name: Option<String>,
                force_unstable_if_unmarked: bool,
                edition: Edition,
                cg: CodegenOptions,
                error_format: ErrorOutputType) -> (clean::Crate, RenderInfo)
{
    // Parse, resolve, and typecheck the given crate.

    let cpath = match input {
        Input::File(ref p) => Some(p.clone()),
        _ => None
    };

    let intra_link_resolution_failure_name = lint::builtin::INTRA_DOC_LINK_RESOLUTION_FAILURE.name;
    let warnings_lint_name = lint::builtin::WARNINGS.name;
    let lints = lint::builtin::HardwiredLints.get_lints()
                    .iter()
                    .chain(rustc_lint::SoftLints.get_lints())
                    .filter_map(|lint| {
                        if lint.name == warnings_lint_name ||
                           lint.name == intra_link_resolution_failure_name {
                            None
                        } else {
                            Some((lint.name_lower(), lint::Allow))
                        }
                    })
                    .collect::<Vec<_>>();

    let host_triple = TargetTriple::from_triple(config::host_triple());
    // plays with error output here!
    let sessopts = config::Options {
        maybe_sysroot,
        search_paths,
        crate_types: vec![config::CrateTypeRlib],
        lint_opts: if !allow_warnings {
            lints
        } else {
            vec![]
        },
        lint_cap: Some(lint::Forbid),
        cg,
        externs,
        target_triple: triple.unwrap_or(host_triple),
        // Ensure that rustdoc works even if rustc is feature-staged
        unstable_features: UnstableFeatures::Allow,
        actually_rustdoc: true,
        debugging_opts: config::DebuggingOptions {
            force_unstable_if_unmarked,
            ..config::basic_debugging_options()
        },
        error_format,
        edition,
        ..config::basic_options()
    };
    driver::spawn_thread_pool(sessopts, move |sessopts| {
        let codemap = Lrc::new(codemap::CodeMap::new(sessopts.file_path_mapping()));
        let diagnostic_handler = new_handler(error_format, Some(codemap.clone()));

        let mut sess = session::build_session_(
            sessopts, cpath, diagnostic_handler, codemap,
        );
        let codegen_backend = rustc_driver::get_codegen_backend(&sess);
        let cstore = Rc::new(CStore::new(codegen_backend.metadata_loader()));
        rustc_lint::register_builtins(&mut sess.lint_store.borrow_mut(), Some(&sess));

        let mut cfg = config::build_configuration(&sess, config::parse_cfgspecs(cfgs));
        target_features::add_configuration(&mut cfg, &sess, &*codegen_backend);
        sess.parse_sess.config = cfg;

        let control = &driver::CompileController::basic();

        let krate = panictry!(driver::phase_1_parse_input(control, &sess, &input));

        let name = match crate_name {
            Some(ref crate_name) => crate_name.clone(),
            None => ::rustc_codegen_utils::link::find_crate_name(Some(&sess), &krate.attrs, &input),
        };

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
            mut resolver,
            ..
        } = abort_on_err(result, &sess);

        resolver.ignore_extern_prelude_feature = true;

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
            access_levels: Lrc::new(AccessLevels::default()),
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

        abort_on_err(driver::phase_3_run_analysis_passes(&*codegen_backend,
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

            let send_trait = if crate_name == Some("core".to_string()) {
                clean::get_trait_def_id(&tcx, &["marker", "Send"], true)
            } else {
                clean::get_trait_def_id(&tcx, &["core", "marker", "Send"], false)
            };

            let ctxt = DocContext {
                tcx,
                resolver: &resolver,
                crate_name,
                cstore: cstore.clone(),
                populated_all_crate_impls: Cell::new(false),
                access_levels: RefCell::new(access_levels),
                external_traits: Default::default(),
                active_extern_traits: Default::default(),
                renderinfo: Default::default(),
                ty_substs: Default::default(),
                lt_substs: Default::default(),
                impl_trait_bounds: Default::default(),
                mod_ids: Default::default(),
                send_trait: send_trait,
                fake_def_ids: RefCell::new(FxHashMap()),
                all_fake_def_ids: RefCell::new(FxHashSet()),
                generated_synthetics: RefCell::new(FxHashSet()),
            };
            debug!("crate: {:?}", tcx.hir.krate());

            let krate = {
                let mut v = RustdocVisitor::new(&*cstore, &ctxt);
                v.visit(tcx.hir.krate());
                v.clean(&ctxt)
            };

            (krate, ctxt.renderinfo.into_inner())
        }), &sess)
    })
}
