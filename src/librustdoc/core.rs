use rustc_ast::ast::CRATE_NODE_ID;
use rustc_attr as attr;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_driver::abort_on_err;
use rustc_errors::emitter::{Emitter, EmitterWriter};
use rustc_errors::json::JsonEmitter;
use rustc_feature::UnstableFeatures;
use rustc_hir::def::Namespace::TypeNS;
use rustc_hir::def_id::{CrateNum, DefId, DefIndex, LOCAL_CRATE};
use rustc_hir::HirId;
use rustc_interface::interface;
use rustc_middle::middle::cstore::CrateStore;
use rustc_middle::middle::privacy::AccessLevels;
use rustc_middle::ty::{Ty, TyCtxt};
use rustc_resolve as resolve;
use rustc_session::config::ErrorOutputType;
use rustc_session::lint;
use rustc_session::DiagnosticOutput;
use rustc_session::{config, Session};
use rustc_span::source_map;
use rustc_span::symbol::sym;
use rustc_span::DUMMY_SP;

use rustc_data_structures::sync::{self, Lrc};
use std::cell::RefCell;
use std::mem;
use std::rc::Rc;

use crate::clean;
use crate::clean::{AttributesExt, MAX_DEF_ID};
use crate::config::{Options as RustdocOptions, RenderOptions};
use crate::html::render::RenderInfo;

use crate::passes::{self, Condition::*, ConditionalPass};

pub use rustc_session::config::{CodegenOptions, DebuggingOptions, Input, Options};
pub use rustc_session::search_paths::SearchPath;

pub type ExternalPaths = FxHashMap<DefId, (Vec<String>, clean::TypeKind)>;

pub struct DocContext<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub resolver: Rc<RefCell<interface::BoxedResolver>>,
    /// Later on moved into `html::render::CACHE_KEY`
    pub renderinfo: RefCell<RenderInfo>,
    /// Later on moved through `clean::Crate` into `html::render::CACHE_KEY`
    pub external_traits: Rc<RefCell<FxHashMap<DefId, clean::Trait>>>,
    /// Used while populating `external_traits` to ensure we don't process the same trait twice at
    /// the same time.
    pub active_extern_traits: RefCell<FxHashSet<DefId>>,
    // The current set of type and lifetime substitutions,
    // for expanding type aliases at the HIR level:
    /// Table `DefId` of type parameter -> substituted type
    pub ty_substs: RefCell<FxHashMap<DefId, clean::Type>>,
    /// Table `DefId` of lifetime parameter -> substituted lifetime
    pub lt_substs: RefCell<FxHashMap<DefId, clean::Lifetime>>,
    /// Table `DefId` of const parameter -> substituted const
    pub ct_substs: RefCell<FxHashMap<DefId, clean::Constant>>,
    /// Table synthetic type parameter for `impl Trait` in argument position -> bounds
    pub impl_trait_bounds: RefCell<FxHashMap<ImplTraitParam, Vec<clean::GenericBound>>>,
    pub fake_def_ids: RefCell<FxHashMap<CrateNum, DefId>>,
    pub all_fake_def_ids: RefCell<FxHashSet<DefId>>,
    /// Auto-trait or blanket impls processed so far, as `(self_ty, trait_def_id)`.
    // FIXME(eddyb) make this a `ty::TraitRef<'tcx>` set.
    pub generated_synthetics: RefCell<FxHashSet<(Ty<'tcx>, DefId)>>,
    pub auto_traits: Vec<DefId>,
}

impl<'tcx> DocContext<'tcx> {
    pub fn sess(&self) -> &Session {
        &self.tcx.sess
    }

    pub fn enter_resolver<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut resolve::Resolver<'_>) -> R,
    {
        self.resolver.borrow_mut().access(f)
    }

    /// Call the closure with the given parameters set as
    /// the substitutions for a type alias' RHS.
    pub fn enter_alias<F, R>(
        &self,
        ty_substs: FxHashMap<DefId, clean::Type>,
        lt_substs: FxHashMap<DefId, clean::Lifetime>,
        ct_substs: FxHashMap<DefId, clean::Constant>,
        f: F,
    ) -> R
    where
        F: FnOnce() -> R,
    {
        let (old_tys, old_lts, old_cts) = (
            mem::replace(&mut *self.ty_substs.borrow_mut(), ty_substs),
            mem::replace(&mut *self.lt_substs.borrow_mut(), lt_substs),
            mem::replace(&mut *self.ct_substs.borrow_mut(), ct_substs),
        );
        let r = f();
        *self.ty_substs.borrow_mut() = old_tys;
        *self.lt_substs.borrow_mut() = old_lts;
        *self.ct_substs.borrow_mut() = old_cts;
        r
    }

    // This is an ugly hack, but it's the simplest way to handle synthetic impls without greatly
    // refactoring either librustdoc or librustc_middle. In particular, allowing new DefIds to be
    // registered after the AST is constructed would require storing the defid mapping in a
    // RefCell, decreasing the performance for normal compilation for very little gain.
    //
    // Instead, we construct 'fake' def ids, which start immediately after the last DefId.
    // In the Debug impl for clean::Item, we explicitly check for fake
    // def ids, as we'll end up with a panic if we use the DefId Debug impl for fake DefIds
    pub fn next_def_id(&self, crate_num: CrateNum) -> DefId {
        let start_def_id = {
            let next_id = if crate_num == LOCAL_CRATE {
                self.tcx.hir().definitions().def_path_table().next_id()
            } else {
                self.enter_resolver(|r| r.cstore().def_path_table(crate_num).next_id())
            };

            DefId { krate: crate_num, index: next_id }
        };

        let mut fake_ids = self.fake_def_ids.borrow_mut();

        let def_id = *fake_ids.entry(crate_num).or_insert(start_def_id);
        fake_ids.insert(
            crate_num,
            DefId { krate: crate_num, index: DefIndex::from(def_id.index.index() + 1) },
        );

        MAX_DEF_ID.with(|m| {
            m.borrow_mut().entry(def_id.krate.clone()).or_insert(start_def_id);
        });

        self.all_fake_def_ids.borrow_mut().insert(def_id);

        def_id
    }

    /// Like the function of the same name on the HIR map, but skips calling it on fake DefIds.
    /// (This avoids a slice-index-out-of-bounds panic.)
    pub fn as_local_hir_id(&self, def_id: DefId) -> Option<HirId> {
        if self.all_fake_def_ids.borrow().contains(&def_id) {
            None
        } else {
            self.tcx.hir().as_local_hir_id(def_id)
        }
    }

    pub fn stability(&self, id: HirId) -> Option<attr::Stability> {
        self.tcx
            .hir()
            .opt_local_def_id(id)
            .and_then(|def_id| self.tcx.lookup_stability(def_id.to_def_id()))
            .cloned()
    }

    pub fn deprecation(&self, id: HirId) -> Option<attr::Deprecation> {
        self.tcx
            .hir()
            .opt_local_def_id(id)
            .and_then(|def_id| self.tcx.lookup_deprecation(def_id.to_def_id()))
    }
}

/// Creates a new diagnostic `Handler` that can be used to emit warnings and errors.
///
/// If the given `error_format` is `ErrorOutputType::Json` and no `SourceMap` is given, a new one
/// will be created for the handler.
pub fn new_handler(
    error_format: ErrorOutputType,
    source_map: Option<Lrc<source_map::SourceMap>>,
    debugging_opts: &DebuggingOptions,
) -> rustc_errors::Handler {
    let emitter: Box<dyn Emitter + sync::Send> = match error_format {
        ErrorOutputType::HumanReadable(kind) => {
            let (short, color_config) = kind.unzip();
            Box::new(
                EmitterWriter::stderr(
                    color_config,
                    source_map.map(|sm| sm as _),
                    short,
                    debugging_opts.teach,
                    debugging_opts.terminal_width,
                    false,
                )
                .ui_testing(debugging_opts.ui_testing),
            )
        }
        ErrorOutputType::Json { pretty, json_rendered } => {
            let source_map = source_map.unwrap_or_else(|| {
                Lrc::new(source_map::SourceMap::new(source_map::FilePathMapping::empty()))
            });
            Box::new(
                JsonEmitter::stderr(None, source_map, pretty, json_rendered, false)
                    .ui_testing(debugging_opts.ui_testing),
            )
        }
    };

    rustc_errors::Handler::with_emitter_and_flags(
        emitter,
        debugging_opts.diagnostic_handler_flags(true),
    )
}

pub fn run_core(options: RustdocOptions) -> (clean::Crate, RenderInfo, RenderOptions) {
    // Parse, resolve, and typecheck the given crate.

    let RustdocOptions {
        input,
        crate_name,
        proc_macro_crate,
        error_format,
        libs,
        externs,
        mut cfgs,
        codegen_options,
        debugging_options,
        target,
        edition,
        maybe_sysroot,
        lint_opts,
        describe_lints,
        lint_cap,
        mut default_passes,
        mut document_private,
        document_hidden,
        mut manual_passes,
        display_warnings,
        render_options,
        output_format,
        ..
    } = options;

    let extern_names: Vec<String> = externs
        .iter()
        .filter(|(_, entry)| entry.add_prelude)
        .map(|(name, _)| name)
        .cloned()
        .collect();

    // Add the doc cfg into the doc build.
    cfgs.push("doc".to_string());

    let cpath = Some(input.clone());
    let input = Input::File(input);

    let intra_link_resolution_failure_name = lint::builtin::INTRA_DOC_LINK_RESOLUTION_FAILURE.name;
    let warnings_lint_name = lint::builtin::WARNINGS.name;
    let missing_docs = rustc_lint::builtin::MISSING_DOCS.name;
    let missing_doc_example = rustc_lint::builtin::MISSING_DOC_CODE_EXAMPLES.name;
    let private_doc_tests = rustc_lint::builtin::PRIVATE_DOC_TESTS.name;
    let no_crate_level_docs = rustc_lint::builtin::MISSING_CRATE_LEVEL_DOCS.name;
    let invalid_codeblock_attribute_name = rustc_lint::builtin::INVALID_CODEBLOCK_ATTRIBUTE.name;

    // In addition to those specific lints, we also need to whitelist those given through
    // command line, otherwise they'll get ignored and we don't want that.
    let mut whitelisted_lints = vec![
        warnings_lint_name.to_owned(),
        intra_link_resolution_failure_name.to_owned(),
        missing_docs.to_owned(),
        missing_doc_example.to_owned(),
        private_doc_tests.to_owned(),
        no_crate_level_docs.to_owned(),
        invalid_codeblock_attribute_name.to_owned(),
    ];

    whitelisted_lints.extend(lint_opts.iter().map(|(lint, _)| lint).cloned());

    let lints = || {
        lint::builtin::HardwiredLints::get_lints()
            .into_iter()
            .chain(rustc_lint::SoftLints::get_lints().into_iter())
    };

    let lint_opts = lints()
        .filter_map(|lint| {
            if lint.name == warnings_lint_name
                || lint.name == intra_link_resolution_failure_name
                || lint.name == invalid_codeblock_attribute_name
            {
                None
            } else {
                Some((lint.name_lower(), lint::Allow))
            }
        })
        .chain(lint_opts.into_iter())
        .collect::<Vec<_>>();

    let lint_caps = lints()
        .filter_map(|lint| {
            // We don't want to whitelist *all* lints so let's
            // ignore those ones.
            if whitelisted_lints.iter().any(|l| lint.name == l) {
                None
            } else {
                Some((lint::LintId::of(lint), lint::Allow))
            }
        })
        .collect();

    let crate_types = if proc_macro_crate {
        vec![config::CrateType::ProcMacro]
    } else {
        vec![config::CrateType::Rlib]
    };
    // plays with error output here!
    let sessopts = config::Options {
        maybe_sysroot,
        search_paths: libs,
        crate_types,
        lint_opts: if !display_warnings { lint_opts } else { vec![] },
        lint_cap: Some(lint_cap.unwrap_or_else(|| lint::Forbid)),
        cg: codegen_options,
        externs,
        target_triple: target,
        unstable_features: UnstableFeatures::from_environment(),
        actually_rustdoc: true,
        debugging_opts: debugging_options,
        error_format,
        edition,
        describe_lints,
        ..Options::default()
    };

    let config = interface::Config {
        opts: sessopts,
        crate_cfg: interface::parse_cfgspecs(cfgs),
        input,
        input_path: cpath,
        output_file: None,
        output_dir: None,
        file_loader: None,
        diagnostic_output: DiagnosticOutput::Default,
        stderr: None,
        crate_name,
        lint_caps,
        register_lints: None,
        override_queries: None,
        registry: rustc_driver::diagnostics_registry(),
    };

    interface::run_compiler_in_existing_thread_pool(config, |compiler| {
        compiler.enter(|queries| {
            let sess = compiler.session();

            // We need to hold on to the complete resolver, so we cause everything to be
            // cloned for the analysis passes to use. Suboptimal, but necessary in the
            // current architecture.
            let resolver = {
                let parts = abort_on_err(queries.expansion(), sess).peek();
                let resolver = parts.1.borrow();

                // Before we actually clone it, let's force all the extern'd crates to
                // actually be loaded, just in case they're only referred to inside
                // intra-doc-links
                resolver.borrow_mut().access(|resolver| {
                    for extern_name in &extern_names {
                        resolver
                            .resolve_str_path_error(DUMMY_SP, extern_name, TypeNS, CRATE_NODE_ID)
                            .unwrap_or_else(|()| {
                                panic!("Unable to resolve external crate {}", extern_name)
                            });
                    }
                });

                // Now we're good to clone the resolver because everything should be loaded
                resolver.clone()
            };

            if sess.has_errors() {
                sess.fatal("Compilation failed, aborting rustdoc");
            }

            let mut global_ctxt = abort_on_err(queries.global_ctxt(), sess).take();

            global_ctxt.enter(|tcx| {
                tcx.analysis(LOCAL_CRATE).ok();

                // Abort if there were any errors so far
                sess.abort_if_errors();

                let access_levels = tcx.privacy_access_levels(LOCAL_CRATE);
                // Convert from a HirId set to a DefId set since we don't always have easy access
                // to the map from defid -> hirid
                let access_levels = AccessLevels {
                    map: access_levels
                        .map
                        .iter()
                        .map(|(&k, &v)| (tcx.hir().local_def_id(k), v))
                        .collect(),
                };

                let mut renderinfo = RenderInfo::default();
                renderinfo.access_levels = access_levels;
                renderinfo.output_format = output_format;

                let mut ctxt = DocContext {
                    tcx,
                    resolver,
                    external_traits: Default::default(),
                    active_extern_traits: Default::default(),
                    renderinfo: RefCell::new(renderinfo),
                    ty_substs: Default::default(),
                    lt_substs: Default::default(),
                    ct_substs: Default::default(),
                    impl_trait_bounds: Default::default(),
                    fake_def_ids: Default::default(),
                    all_fake_def_ids: Default::default(),
                    generated_synthetics: Default::default(),
                    auto_traits: tcx
                        .all_traits(LOCAL_CRATE)
                        .iter()
                        .cloned()
                        .filter(|trait_def_id| tcx.trait_is_auto(*trait_def_id))
                        .collect(),
                };
                debug!("crate: {:?}", tcx.hir().krate());

                let mut krate = clean::krate(&mut ctxt);

                if let Some(ref m) = krate.module {
                    if let None | Some("") = m.doc_value() {
                        let help = "The following guide may be of use:\n\
                             https://doc.rust-lang.org/nightly/rustdoc/how-to-write-documentation\
                             .html";
                        tcx.struct_lint_node(
                            rustc_lint::builtin::MISSING_CRATE_LEVEL_DOCS,
                            ctxt.as_local_hir_id(m.def_id).unwrap(),
                            |lint| {
                                let mut diag = lint.build(
                                    "no documentation found for this crate's top-level module",
                                );
                                diag.help(help);
                                diag.emit();
                            },
                        );
                    }
                }

                fn report_deprecated_attr(name: &str, diag: &rustc_errors::Handler) {
                    let mut msg = diag.struct_warn(&format!(
                        "the `#![doc({})]` attribute is considered deprecated",
                        name
                    ));
                    msg.warn(
                        "see issue #44136 <https://github.com/rust-lang/rust/issues/44136> \
                         for more information",
                    );

                    if name == "no_default_passes" {
                        msg.help("you may want to use `#![doc(document_private_items)]`");
                    }

                    msg.emit();
                }

                // Process all of the crate attributes, extracting plugin metadata along
                // with the passes which we are supposed to run.
                for attr in krate.module.as_ref().unwrap().attrs.lists(sym::doc) {
                    let diag = ctxt.sess().diagnostic();

                    let name = attr.name_or_empty();
                    if attr.is_word() {
                        if name == sym::no_default_passes {
                            report_deprecated_attr("no_default_passes", diag);
                            if default_passes == passes::DefaultPassOption::Default {
                                default_passes = passes::DefaultPassOption::None;
                            }
                        }
                    } else if let Some(value) = attr.value_str() {
                        let sink = match name {
                            sym::passes => {
                                report_deprecated_attr("passes = \"...\"", diag);
                                &mut manual_passes
                            }
                            sym::plugins => {
                                report_deprecated_attr("plugins = \"...\"", diag);
                                eprintln!(
                                    "WARNING: `#![doc(plugins = \"...\")]` \
                                      no longer functions; see CVE-2018-1000622"
                                );
                                continue;
                            }
                            _ => continue,
                        };
                        for name in value.as_str().split_whitespace() {
                            sink.push(name.to_string());
                        }
                    }

                    if attr.is_word() && name == sym::document_private_items {
                        document_private = true;
                    }
                }

                let passes = passes::defaults(default_passes).iter().copied().chain(
                    manual_passes.into_iter().flat_map(|name| {
                        if let Some(pass) = passes::find_pass(&name) {
                            Some(ConditionalPass::always(pass))
                        } else {
                            error!("unknown pass {}, skipping", name);
                            None
                        }
                    }),
                );

                info!("Executing passes");

                for p in passes {
                    let run = match p.condition {
                        Always => true,
                        WhenDocumentPrivate => document_private,
                        WhenNotDocumentPrivate => !document_private,
                        WhenNotDocumentHidden => !document_hidden,
                    };
                    if run {
                        debug!("running pass {}", p.pass.name);
                        krate = (p.pass.run)(krate, &ctxt);
                    }
                }

                ctxt.sess().abort_if_errors();

                (krate, ctxt.renderinfo.into_inner(), render_options)
            })
        })
    })
}

/// `DefId` or parameter index (`ty::ParamTy.index`) of a synthetic type parameter
/// for `impl Trait` in argument position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ImplTraitParam {
    DefId(DefId),
    ParamIndex(u32),
}

impl From<DefId> for ImplTraitParam {
    fn from(did: DefId) -> Self {
        ImplTraitParam::DefId(did)
    }
}

impl From<u32> for ImplTraitParam {
    fn from(idx: u32) -> Self {
        ImplTraitParam::ParamIndex(idx)
    }
}
