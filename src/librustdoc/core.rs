use std::sync::{Arc, LazyLock};
use std::{io, mem};

use rustc_data_structures::fx::{FxHashMap, FxHashSet, FxIndexMap};
use rustc_data_structures::unord::UnordSet;
use rustc_driver::USING_INTERNAL_FEATURES;
use rustc_errors::TerminalUrl;
use rustc_errors::codes::*;
use rustc_errors::emitter::{
    DynEmitter, HumanEmitter, HumanReadableErrorType, OutputTheme, stderr_destination,
};
use rustc_errors::json::JsonEmitter;
use rustc_feature::UnstableFeatures;
use rustc_hir::def::Res;
use rustc_hir::def_id::{DefId, DefIdMap, DefIdSet, LocalDefId};
use rustc_hir::intravisit::{self, Visitor};
use rustc_hir::{HirId, Path};
use rustc_lint::{MissingDoc, late_lint_mod};
use rustc_middle::hir::nested_filter;
use rustc_middle::ty::{self, ParamEnv, Ty, TyCtxt};
use rustc_session::config::{
    self, CrateType, ErrorOutputType, Input, OutFileName, OutputType, OutputTypes, ResolveDocLinks,
};
pub(crate) use rustc_session::config::{Options, UnstableOptions};
use rustc_session::{Session, lint};
use rustc_span::source_map;
use rustc_span::symbol::sym;
use tracing::{debug, info};

use crate::clean::inline::build_external_trait;
use crate::clean::{self, ItemId};
use crate::config::{Options as RustdocOptions, OutputFormat, RenderOptions};
use crate::formats::cache::Cache;
use crate::passes;
use crate::passes::Condition::*;
use crate::passes::collect_intra_doc_links::LinkCollector;

pub(crate) struct DocContext<'tcx> {
    pub(crate) tcx: TyCtxt<'tcx>,
    /// Used for normalization.
    ///
    /// Most of this logic is copied from rustc_lint::late.
    pub(crate) param_env: ParamEnv<'tcx>,
    /// Later on moved through `clean::Crate` into `cache`
    pub(crate) external_traits: FxIndexMap<DefId, clean::Trait>,
    /// Used while populating `external_traits` to ensure we don't process the same trait twice at
    /// the same time.
    pub(crate) active_extern_traits: DefIdSet,
    /// The current set of parameter instantiations for expanding type aliases at the HIR level.
    ///
    /// Maps from the `DefId` of a lifetime or type parameter to the
    /// generic argument it's currently instantiated to in this context.
    // FIXME(#82852): We don't record const params since we don't visit const exprs at all and
    // therefore wouldn't use the corresp. generic arg anyway. Add support for them.
    pub(crate) args: DefIdMap<clean::GenericArg>,
    pub(crate) current_type_aliases: DefIdMap<usize>,
    /// Table synthetic type parameter for `impl Trait` in argument position -> bounds
    pub(crate) impl_trait_bounds: FxHashMap<ImplTraitParam, Vec<clean::GenericBound>>,
    /// Auto-trait or blanket impls processed so far, as `(self_ty, trait_def_id)`.
    // FIXME(eddyb) make this a `ty::TraitRef<'tcx>` set.
    pub(crate) generated_synthetics: FxHashSet<(Ty<'tcx>, DefId)>,
    pub(crate) auto_traits: Vec<DefId>,
    /// The options given to rustdoc that could be relevant to a pass.
    pub(crate) render_options: RenderOptions,
    /// This same cache is used throughout rustdoc, including in [`crate::html::render`].
    pub(crate) cache: Cache,
    /// Used by [`clean::inline`] to tell if an item has already been inlined.
    pub(crate) inlined: FxHashSet<ItemId>,
    /// Used by `calculate_doc_coverage`.
    pub(crate) output_format: OutputFormat,
    /// Used by `strip_private`.
    pub(crate) show_coverage: bool,
}

impl<'tcx> DocContext<'tcx> {
    pub(crate) fn sess(&self) -> &'tcx Session {
        self.tcx.sess
    }

    pub(crate) fn with_param_env<T, F: FnOnce(&mut Self) -> T>(
        &mut self,
        def_id: DefId,
        f: F,
    ) -> T {
        let old_param_env = mem::replace(&mut self.param_env, self.tcx.param_env(def_id));
        let ret = f(self);
        self.param_env = old_param_env;
        ret
    }

    pub(crate) fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        ty::TypingEnv {
            typing_mode: ty::TypingMode::non_body_analysis(),
            param_env: self.param_env,
        }
    }

    /// Call the closure with the given parameters set as
    /// the generic parameters for a type alias' RHS.
    pub(crate) fn enter_alias<F, R>(
        &mut self,
        args: DefIdMap<clean::GenericArg>,
        def_id: DefId,
        f: F,
    ) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let old_args = mem::replace(&mut self.args, args);
        *self.current_type_aliases.entry(def_id).or_insert(0) += 1;
        let r = f(self);
        self.args = old_args;
        if let Some(count) = self.current_type_aliases.get_mut(&def_id) {
            *count -= 1;
            if *count == 0 {
                self.current_type_aliases.remove(&def_id);
            }
        }
        r
    }

    /// Like `tcx.local_def_id_to_hir_id()`, but skips calling it on fake DefIds.
    /// (This avoids a slice-index-out-of-bounds panic.)
    pub(crate) fn as_local_hir_id(tcx: TyCtxt<'_>, item_id: ItemId) -> Option<HirId> {
        match item_id {
            ItemId::DefId(real_id) => {
                real_id.as_local().map(|def_id| tcx.local_def_id_to_hir_id(def_id))
            }
            // FIXME: Can this be `Some` for `Auto` or `Blanket`?
            _ => None,
        }
    }

    /// Returns `true` if the JSON output format is enabled for generating the crate content.
    ///
    /// If another option like `--show-coverage` is enabled, it will return `false`.
    pub(crate) fn is_json_output(&self) -> bool {
        self.output_format.is_json() && !self.show_coverage
    }
}

/// Creates a new `DiagCtxt` that can be used to emit warnings and errors.
///
/// If the given `error_format` is `ErrorOutputType::Json` and no `SourceMap` is given, a new one
/// will be created for the `DiagCtxt`.
pub(crate) fn new_dcx(
    error_format: ErrorOutputType,
    source_map: Option<Arc<source_map::SourceMap>>,
    diagnostic_width: Option<usize>,
    unstable_opts: &UnstableOptions,
) -> rustc_errors::DiagCtxt {
    let fallback_bundle = rustc_errors::fallback_fluent_bundle(
        rustc_driver::DEFAULT_LOCALE_RESOURCES.to_vec(),
        false,
    );
    let emitter: Box<DynEmitter> = match error_format {
        ErrorOutputType::HumanReadable { kind, color_config } => {
            let short = kind.short();
            Box::new(
                HumanEmitter::new(stderr_destination(color_config), fallback_bundle)
                    .sm(source_map.map(|sm| sm as _))
                    .short_message(short)
                    .diagnostic_width(diagnostic_width)
                    .track_diagnostics(unstable_opts.track_diagnostics)
                    .theme(if let HumanReadableErrorType::Unicode = kind {
                        OutputTheme::Unicode
                    } else {
                        OutputTheme::Ascii
                    })
                    .ui_testing(unstable_opts.ui_testing),
            )
        }
        ErrorOutputType::Json { pretty, json_rendered, color_config } => {
            let source_map = source_map.unwrap_or_else(|| {
                Arc::new(source_map::SourceMap::new(source_map::FilePathMapping::empty()))
            });
            Box::new(
                JsonEmitter::new(
                    Box::new(io::BufWriter::new(io::stderr())),
                    Some(source_map),
                    fallback_bundle,
                    pretty,
                    json_rendered,
                    color_config,
                )
                .ui_testing(unstable_opts.ui_testing)
                .diagnostic_width(diagnostic_width)
                .track_diagnostics(unstable_opts.track_diagnostics)
                .terminal_url(TerminalUrl::No),
            )
        }
    };

    rustc_errors::DiagCtxt::new(emitter).with_flags(unstable_opts.dcx_flags(true))
}

/// Parse, resolve, and typecheck the given crate.
pub(crate) fn create_config(
    input: Input,
    RustdocOptions {
        crate_name,
        proc_macro_crate,
        error_format,
        diagnostic_width,
        libs,
        externs,
        mut cfgs,
        check_cfgs,
        codegen_options,
        unstable_opts,
        target,
        edition,
        sysroot,
        lint_opts,
        describe_lints,
        lint_cap,
        scrape_examples_options,
        expanded_args,
        remap_path_prefix,
        ..
    }: RustdocOptions,
    render_options: &RenderOptions,
) -> rustc_interface::Config {
    // Add the doc cfg into the doc build.
    cfgs.push("doc".to_string());

    // By default, rustdoc ignores all lints.
    // Specifically unblock lints relevant to documentation or the lint machinery itself.
    let mut lints_to_show = vec![
        // it's unclear whether these should be part of rustdoc directly (#77364)
        rustc_lint::builtin::MISSING_DOCS.name.to_string(),
        rustc_lint::builtin::INVALID_DOC_ATTRIBUTES.name.to_string(),
        // these are definitely not part of rustdoc, but we want to warn on them anyway.
        rustc_lint::builtin::RENAMED_AND_REMOVED_LINTS.name.to_string(),
        rustc_lint::builtin::UNKNOWN_LINTS.name.to_string(),
        rustc_lint::builtin::UNEXPECTED_CFGS.name.to_string(),
        // this lint is needed to support `#[expect]` attributes
        rustc_lint::builtin::UNFULFILLED_LINT_EXPECTATIONS.name.to_string(),
    ];
    lints_to_show.extend(crate::lint::RUSTDOC_LINTS.iter().map(|lint| lint.name.to_string()));

    let (lint_opts, lint_caps) = crate::lint::init_lints(lints_to_show, lint_opts, |lint| {
        Some((lint.name_lower(), lint::Allow))
    });

    let crate_types =
        if proc_macro_crate { vec![CrateType::ProcMacro] } else { vec![CrateType::Rlib] };
    let resolve_doc_links = if render_options.document_private {
        ResolveDocLinks::All
    } else {
        ResolveDocLinks::Exported
    };
    let test = scrape_examples_options.map(|opts| opts.scrape_tests).unwrap_or(false);
    // plays with error output here!
    let sessopts = config::Options {
        sysroot,
        search_paths: libs,
        crate_types,
        lint_opts,
        lint_cap,
        cg: codegen_options,
        externs,
        target_triple: target,
        unstable_features: UnstableFeatures::from_environment(crate_name.as_deref()),
        actually_rustdoc: true,
        resolve_doc_links,
        unstable_opts,
        error_format,
        diagnostic_width,
        edition,
        describe_lints,
        crate_name,
        test,
        remap_path_prefix,
        output_types: if let Some(file) = render_options.dep_info() {
            OutputTypes::new(&[(
                OutputType::DepInfo,
                file.map(|f| OutFileName::Real(f.to_path_buf())),
            )])
        } else {
            OutputTypes::new(&[])
        },
        ..Options::default()
    };

    rustc_interface::Config {
        opts: sessopts,
        crate_cfg: cfgs,
        crate_check_cfg: check_cfgs,
        input,
        output_file: None,
        output_dir: None,
        file_loader: None,
        locale_resources: rustc_driver::DEFAULT_LOCALE_RESOURCES.to_vec(),
        lint_caps,
        psess_created: None,
        hash_untracked_state: None,
        register_lints: Some(Box::new(crate::lint::register_lints)),
        override_queries: Some(|_sess, providers| {
            // We do not register late module lints, so this only runs `MissingDoc`.
            // Most lints will require typechecking, so just don't run them.
            providers.lint_mod = |tcx, module_def_id| late_lint_mod(tcx, module_def_id, MissingDoc);
            // hack so that `used_trait_imports` won't try to call typeck
            providers.used_trait_imports = |_, _| {
                static EMPTY_SET: LazyLock<UnordSet<LocalDefId>> = LazyLock::new(UnordSet::default);
                &EMPTY_SET
            };
            // In case typeck does end up being called, don't ICE in case there were name resolution errors
            providers.typeck = move |tcx, def_id| {
                // Closures' tables come from their outermost function,
                // as they are part of the same "inference environment".
                // This avoids emitting errors for the parent twice (see similar code in `typeck_with_fallback`)
                let typeck_root_def_id = tcx.typeck_root_def_id(def_id.to_def_id()).expect_local();
                if typeck_root_def_id != def_id {
                    return tcx.typeck(typeck_root_def_id);
                }

                let body = tcx.hir_body_owned_by(def_id);
                debug!("visiting body for {def_id:?}");
                EmitIgnoredResolutionErrors::new(tcx).visit_body(body);
                (rustc_interface::DEFAULT_QUERY_PROVIDERS.typeck)(tcx, def_id)
            };
        }),
        extra_symbols: Vec::new(),
        make_codegen_backend: None,
        registry: rustc_driver::diagnostics_registry(),
        ice_file: None,
        using_internal_features: &USING_INTERNAL_FEATURES,
        expanded_args,
    }
}

pub(crate) fn run_global_ctxt(
    tcx: TyCtxt<'_>,
    show_coverage: bool,
    render_options: RenderOptions,
    output_format: OutputFormat,
) -> (clean::Crate, RenderOptions, Cache) {
    // Certain queries assume that some checks were run elsewhere
    // (see https://github.com/rust-lang/rust/pull/73566#issuecomment-656954425),
    // so type-check everything other than function bodies in this crate before running lints.

    // NOTE: this does not call `tcx.analysis()` so that we won't
    // typeck function bodies or run the default rustc lints.
    // (see `override_queries` in the `config`)

    // NOTE: These are copy/pasted from typeck/lib.rs and should be kept in sync with those changes.
    let _ = tcx.sess.time("wf_checking", || {
        tcx.try_par_hir_for_each_module(|module| tcx.ensure_ok().check_mod_type_wf(module))
    });

    tcx.dcx().abort_if_errors();

    tcx.sess.time("missing_docs", || rustc_lint::check_crate(tcx));
    tcx.sess.time("check_mod_attrs", || {
        tcx.hir_for_each_module(|module| tcx.ensure_ok().check_mod_attrs(module))
    });
    rustc_passes::stability::check_unused_or_stable_features(tcx);

    let auto_traits =
        tcx.all_traits().filter(|&trait_def_id| tcx.trait_is_auto(trait_def_id)).collect();

    let mut ctxt = DocContext {
        tcx,
        param_env: ParamEnv::empty(),
        external_traits: Default::default(),
        active_extern_traits: Default::default(),
        args: Default::default(),
        current_type_aliases: Default::default(),
        impl_trait_bounds: Default::default(),
        generated_synthetics: Default::default(),
        auto_traits,
        cache: Cache::new(render_options.document_private, render_options.document_hidden),
        inlined: FxHashSet::default(),
        output_format,
        render_options,
        show_coverage,
    };

    for cnum in tcx.crates(()) {
        crate::visit_lib::lib_embargo_visit_item(&mut ctxt, cnum.as_def_id());
    }

    // Small hack to force the Sized trait to be present.
    //
    // Note that in case of `#![no_core]`, the trait is not available.
    if let Some(sized_trait_did) = ctxt.tcx.lang_items().sized_trait() {
        let sized_trait = build_external_trait(&mut ctxt, sized_trait_did);
        ctxt.external_traits.insert(sized_trait_did, sized_trait);
    }

    debug!("crate: {:?}", tcx.hir_crate(()));

    let mut krate = tcx.sess.time("clean_crate", || clean::krate(&mut ctxt));

    if krate.module.doc_value().is_empty() {
        let help = format!(
            "The following guide may be of use:\n\
            {}/rustdoc/how-to-write-documentation.html",
            crate::DOC_RUST_LANG_ORG_VERSION
        );
        tcx.node_lint(
            crate::lint::MISSING_CRATE_LEVEL_DOCS,
            DocContext::as_local_hir_id(tcx, krate.module.item_id).unwrap(),
            |lint| {
                lint.primary_message("no documentation found for this crate's top-level module");
                lint.help(help);
            },
        );
    }

    // Process all of the crate attributes, extracting plugin metadata along
    // with the passes which we are supposed to run.
    for attr in krate.module.attrs.lists(sym::doc) {
        let name = attr.name_or_empty();

        if attr.is_word() && name == sym::document_private_items {
            ctxt.render_options.document_private = true;
        }
    }

    info!("Executing passes");

    let mut visited = FxHashMap::default();
    let mut ambiguous = FxIndexMap::default();

    for p in passes::defaults(show_coverage) {
        let run = match p.condition {
            Always => true,
            WhenDocumentPrivate => ctxt.render_options.document_private,
            WhenNotDocumentPrivate => !ctxt.render_options.document_private,
            WhenNotDocumentHidden => !ctxt.render_options.document_hidden,
        };
        if run {
            debug!("running pass {}", p.pass.name);
            if let Some(run_fn) = p.pass.run {
                krate = tcx.sess.time(p.pass.name, || run_fn(krate, &mut ctxt));
            } else {
                let (k, LinkCollector { visited_links, ambiguous_links, .. }) =
                    passes::collect_intra_doc_links::collect_intra_doc_links(krate, &mut ctxt);
                krate = k;
                visited = visited_links;
                ambiguous = ambiguous_links;
            }
        }
    }

    tcx.sess.time("check_lint_expectations", || tcx.check_expectations(Some(sym::rustdoc)));

    krate = tcx.sess.time("create_format_cache", || Cache::populate(&mut ctxt, krate));

    let mut collector =
        LinkCollector { cx: &mut ctxt, visited_links: visited, ambiguous_links: ambiguous };
    collector.resolve_ambiguities();

    tcx.dcx().abort_if_errors();

    (krate, ctxt.render_options, ctxt.cache)
}

/// Due to <https://github.com/rust-lang/rust/pull/73566>,
/// the name resolution pass may find errors that are never emitted.
/// If typeck is called after this happens, then we'll get an ICE:
/// 'Res::Error found but not reported'. To avoid this, emit the errors now.
struct EmitIgnoredResolutionErrors<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> EmitIgnoredResolutionErrors<'tcx> {
    fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx }
    }
}

impl<'tcx> Visitor<'tcx> for EmitIgnoredResolutionErrors<'tcx> {
    type NestedFilter = nested_filter::OnlyBodies;

    fn maybe_tcx(&mut self) -> Self::MaybeTyCtxt {
        // We need to recurse into nested closures,
        // since those will fallback to the parent for type checking.
        self.tcx
    }

    fn visit_path(&mut self, path: &Path<'tcx>, _id: HirId) {
        debug!("visiting path {path:?}");
        if path.res == Res::Err {
            // We have less context here than in rustc_resolve,
            // so we can only emit the name and span.
            // However we can give a hint that rustc_resolve will have more info.
            let label = format!(
                "could not resolve path `{}`",
                path.segments
                    .iter()
                    .map(|segment| segment.ident.as_str())
                    .intersperse("::")
                    .collect::<String>()
            );
            rustc_errors::struct_span_code_err!(
                self.tcx.dcx(),
                path.span,
                E0433,
                "failed to resolve: {label}",
            )
            .with_span_label(path.span, label)
            .with_note("this error was originally ignored because you are running `rustdoc`")
            .with_note("try running again with `rustc` or `cargo check` and you may get a more detailed error")
            .emit();
        }
        // We could have an outer resolution that succeeded,
        // but with generic parameters that failed.
        // Recurse into the segments so we catch those too.
        intravisit::walk_path(self, path);
    }
}

/// `DefId` or parameter index (`ty::ParamTy.index`) of a synthetic type parameter
/// for `impl Trait` in argument position.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum ImplTraitParam {
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
