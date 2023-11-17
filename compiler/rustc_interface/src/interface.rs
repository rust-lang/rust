use crate::util;

use rustc_ast::token;
use rustc_ast::{self as ast, LitKind, MetaItemKind};
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_data_structures::defer;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::stable_hasher::StableHasher;
use rustc_data_structures::sync::Lrc;
use rustc_errors::registry::Registry;
use rustc_errors::{ErrorGuaranteed, Handler};
use rustc_lint::LintStore;
use rustc_middle::util::Providers;
use rustc_middle::{bug, ty};
use rustc_parse::maybe_new_parser_from_source_str;
use rustc_query_impl::QueryCtxt;
use rustc_query_system::query::print_query_stack;
use rustc_session::config::{
    self, Cfg, CheckCfg, ExpectedValues, Input, OutFileName, OutputFilenames,
};
use rustc_session::filesearch::sysroot_candidates;
use rustc_session::parse::ParseSess;
use rustc_session::{lint, CompilerIO, EarlyErrorHandler, Session};
use rustc_span::source_map::FileLoader;
use rustc_span::symbol::sym;
use rustc_span::FileName;
use std::path::PathBuf;
use std::result;
use std::sync::Arc;

pub type Result<T> = result::Result<T, ErrorGuaranteed>;

/// Represents a compiler session. Note that every `Compiler` contains a
/// `Session`, but `Compiler` also contains some things that cannot be in
/// `Session`, due to `Session` being in a crate that has many fewer
/// dependencies than this crate.
///
/// Can be used to run `rustc_interface` queries.
/// Created by passing [`Config`] to [`run_compiler`].
pub struct Compiler {
    sess: Session,
    codegen_backend: Box<dyn CodegenBackend>,
    pub(crate) override_queries: Option<fn(&Session, &mut Providers)>,
}

impl Compiler {
    pub fn session(&self) -> &Session {
        &self.sess
    }
    pub fn codegen_backend(&self) -> &dyn CodegenBackend {
        &*self.codegen_backend
    }
    pub fn build_output_filenames(
        &self,
        sess: &Session,
        attrs: &[ast::Attribute],
    ) -> OutputFilenames {
        util::build_output_filenames(attrs, sess)
    }
}

/// Converts strings provided as `--cfg [cfgspec]` into a `Cfg`.
pub(crate) fn parse_cfg(handler: &EarlyErrorHandler, cfgs: Vec<String>) -> Cfg {
    cfgs.into_iter()
        .map(|s| {
            let sess = ParseSess::with_silent_emitter(Some(format!(
                "this error occurred on the command line: `--cfg={s}`"
            )));
            let filename = FileName::cfg_spec_source_code(&s);

            macro_rules! error {
                ($reason: expr) => {
                    handler.early_error(format!(
                        concat!("invalid `--cfg` argument: `{}` (", $reason, ")"),
                        s
                    ));
                };
            }

            match maybe_new_parser_from_source_str(&sess, filename, s.to_string()) {
                Ok(mut parser) => match parser.parse_meta_item() {
                    Ok(meta_item) if parser.token == token::Eof => {
                        if meta_item.path.segments.len() != 1 {
                            error!("argument key must be an identifier");
                        }
                        match &meta_item.kind {
                            MetaItemKind::List(..) => {}
                            MetaItemKind::NameValue(lit) if !lit.kind.is_str() => {
                                error!("argument value must be a string");
                            }
                            MetaItemKind::NameValue(..) | MetaItemKind::Word => {
                                let ident = meta_item.ident().expect("multi-segment cfg key");
                                return (ident.name, meta_item.value_str());
                            }
                        }
                    }
                    Ok(..) => {}
                    Err(err) => err.cancel(),
                },
                Err(errs) => drop(errs),
            }

            // If the user tried to use a key="value" flag, but is missing the quotes, provide
            // a hint about how to resolve this.
            if s.contains('=') && !s.contains("=\"") && !s.ends_with('"') {
                error!(concat!(
                    r#"expected `key` or `key="value"`, ensure escaping is appropriate"#,
                    r#" for your shell, try 'key="value"' or key=\"value\""#
                ));
            } else {
                error!(r#"expected `key` or `key="value"`"#);
            }
        })
        .collect::<Cfg>()
}

/// Converts strings provided as `--check-cfg [specs]` into a `CheckCfg`.
pub(crate) fn parse_check_cfg(handler: &EarlyErrorHandler, specs: Vec<String>) -> CheckCfg {
    // If any --check-cfg is passed then exhaustive_values and exhaustive_names
    // are enabled by default.
    let exhaustive_names = !specs.is_empty();
    let exhaustive_values = !specs.is_empty();
    let mut check_cfg = CheckCfg { exhaustive_names, exhaustive_values, ..CheckCfg::default() };

    let mut old_syntax = None;
    for s in specs {
        let sess = ParseSess::with_silent_emitter(Some(format!(
            "this error occurred on the command line: `--check-cfg={s}`"
        )));
        let filename = FileName::cfg_spec_source_code(&s);

        macro_rules! error {
            ($reason:expr) => {
                handler.early_error(format!(
                    concat!("invalid `--check-cfg` argument: `{}` (", $reason, ")"),
                    s
                ))
            };
        }

        let expected_error = || -> ! {
            error!("expected `cfg(name, values(\"value1\", \"value2\", ... \"valueN\"))`")
        };

        let Ok(mut parser) = maybe_new_parser_from_source_str(&sess, filename, s.to_string())
        else {
            expected_error();
        };

        let meta_item = match parser.parse_meta_item() {
            Ok(meta_item) if parser.token == token::Eof => meta_item,
            Ok(..) => expected_error(),
            Err(err) => {
                err.cancel();
                expected_error();
            }
        };

        let Some(args) = meta_item.meta_item_list() else {
            expected_error();
        };

        let mut set_old_syntax = || {
            // defaults are flipped for the old syntax
            if old_syntax == None {
                check_cfg.exhaustive_names = false;
                check_cfg.exhaustive_values = false;
            }
            old_syntax = Some(true);
        };

        if meta_item.has_name(sym::names) {
            set_old_syntax();

            check_cfg.exhaustive_names = true;
            for arg in args {
                if arg.is_word()
                    && let Some(ident) = arg.ident()
                {
                    check_cfg.expecteds.entry(ident.name).or_insert(ExpectedValues::Any);
                } else {
                    error!("`names()` arguments must be simple identifiers");
                }
            }
        } else if meta_item.has_name(sym::values) {
            set_old_syntax();

            if let Some((name, values)) = args.split_first() {
                if name.is_word()
                    && let Some(ident) = name.ident()
                {
                    let expected_values = check_cfg
                        .expecteds
                        .entry(ident.name)
                        .and_modify(|expected_values| match expected_values {
                            ExpectedValues::Some(_) => {}
                            ExpectedValues::Any => {
                                // handle the case where names(...) was done
                                // before values by changing to a list
                                *expected_values = ExpectedValues::Some(FxHashSet::default());
                            }
                        })
                        .or_insert_with(|| ExpectedValues::Some(FxHashSet::default()));

                    let ExpectedValues::Some(expected_values) = expected_values else {
                        bug!("`expected_values` should be a list a values")
                    };

                    for val in values {
                        if let Some(LitKind::Str(s, _)) = val.lit().map(|lit| &lit.kind) {
                            expected_values.insert(Some(*s));
                        } else {
                            error!("`values()` arguments must be string literals");
                        }
                    }

                    if values.is_empty() {
                        expected_values.insert(None);
                    }
                } else {
                    error!("`values()` first argument must be a simple identifier");
                }
            } else if args.is_empty() {
                check_cfg.exhaustive_values = true;
            } else {
                expected_error();
            }
        } else if meta_item.has_name(sym::cfg) {
            old_syntax = Some(false);

            let mut names = Vec::new();
            let mut values: FxHashSet<_> = Default::default();

            let mut any_specified = false;
            let mut values_specified = false;
            let mut values_any_specified = false;

            for arg in args {
                if arg.is_word()
                    && let Some(ident) = arg.ident()
                {
                    if values_specified {
                        error!("`cfg()` names cannot be after values");
                    }
                    names.push(ident);
                } else if arg.has_name(sym::any)
                    && let Some(args) = arg.meta_item_list()
                {
                    if any_specified {
                        error!("`any()` cannot be specified multiple times");
                    }
                    any_specified = true;
                    if !args.is_empty() {
                        error!("`any()` must be empty");
                    }
                } else if arg.has_name(sym::values)
                    && let Some(args) = arg.meta_item_list()
                {
                    if names.is_empty() {
                        error!("`values()` cannot be specified before the names");
                    } else if values_specified {
                        error!("`values()` cannot be specified multiple times");
                    }
                    values_specified = true;

                    for arg in args {
                        if let Some(LitKind::Str(s, _)) = arg.lit().map(|lit| &lit.kind) {
                            values.insert(Some(*s));
                        } else if arg.has_name(sym::any)
                            && let Some(args) = arg.meta_item_list()
                        {
                            if values_any_specified {
                                error!("`any()` in `values()` cannot be specified multiple times");
                            }
                            values_any_specified = true;
                            if !args.is_empty() {
                                error!("`any()` must be empty");
                            }
                        } else {
                            error!("`values()` arguments must be string literals or `any()`");
                        }
                    }
                } else {
                    error!(
                        "`cfg()` arguments must be simple identifiers, `any()` or `values(...)`"
                    );
                }
            }

            if values.is_empty() && !values_any_specified && !any_specified {
                values.insert(None);
            } else if !values.is_empty() && values_any_specified {
                error!(
                    "`values()` arguments cannot specify string literals and `any()` at the same time"
                );
            }

            if any_specified {
                if names.is_empty()
                    && values.is_empty()
                    && !values_specified
                    && !values_any_specified
                {
                    check_cfg.exhaustive_names = false;
                } else {
                    error!("`cfg(any())` can only be provided in isolation");
                }
            } else {
                for name in names {
                    check_cfg
                        .expecteds
                        .entry(name.name)
                        .and_modify(|v| match v {
                            ExpectedValues::Some(v) if !values_any_specified => {
                                v.extend(values.clone())
                            }
                            ExpectedValues::Some(_) => *v = ExpectedValues::Any,
                            ExpectedValues::Any => {}
                        })
                        .or_insert_with(|| {
                            if values_any_specified {
                                ExpectedValues::Any
                            } else {
                                ExpectedValues::Some(values.clone())
                            }
                        });
                }
            }
        } else {
            expected_error();
        }
    }

    check_cfg
}

/// The compiler configuration
pub struct Config {
    /// Command line options
    pub opts: config::Options,

    /// Unparsed cfg! configuration in addition to the default ones.
    pub crate_cfg: Vec<String>,
    pub crate_check_cfg: Vec<String>,

    pub input: Input,
    pub output_dir: Option<PathBuf>,
    pub output_file: Option<OutFileName>,
    pub ice_file: Option<PathBuf>,
    pub file_loader: Option<Box<dyn FileLoader + Send + Sync>>,
    pub locale_resources: &'static [&'static str],

    pub lint_caps: FxHashMap<lint::LintId, lint::Level>,

    /// This is a callback from the driver that is called when [`ParseSess`] is created.
    pub parse_sess_created: Option<Box<dyn FnOnce(&mut ParseSess) + Send>>,

    /// This is a callback to hash otherwise untracked state used by the caller, if the
    /// hash changes between runs the incremental cache will be cleared.
    ///
    /// e.g. used by Clippy to hash its config file
    pub hash_untracked_state: Option<Box<dyn FnOnce(&Session, &mut StableHasher) + Send>>,

    /// This is a callback from the driver that is called when we're registering lints;
    /// it is called during lint loading when we have the LintStore in a non-shared state.
    ///
    /// Note that if you find a Some here you probably want to call that function in the new
    /// function being registered.
    pub register_lints: Option<Box<dyn Fn(&Session, &mut LintStore) + Send + Sync>>,

    /// This is a callback from the driver that is called just after we have populated
    /// the list of queries.
    pub override_queries: Option<fn(&Session, &mut Providers)>,

    /// This is a callback from the driver that is called to create a codegen backend.
    pub make_codegen_backend:
        Option<Box<dyn FnOnce(&config::Options) -> Box<dyn CodegenBackend> + Send>>,

    /// Registry of diagnostics codes.
    pub registry: Registry,

    /// The inner atomic value is set to true when a feature marked as `internal` is
    /// enabled. Makes it so that "please report a bug" is hidden, as ICEs with
    /// internal features are wontfix, and they are usually the cause of the ICEs.
    /// None signifies that this is not tracked.
    pub using_internal_features: Arc<std::sync::atomic::AtomicBool>,

    /// All commandline args used to invoke the compiler, with @file args fully expanded.
    /// This will only be used within debug info, e.g. in the pdb file on windows
    /// This is mainly useful for other tools that reads that debuginfo to figure out
    /// how to call the compiler with the same arguments.
    pub expanded_args: Vec<String>,
}

// JUSTIFICATION: before session exists, only config
#[allow(rustc::bad_opt_access)]
pub fn run_compiler<R: Send>(config: Config, f: impl FnOnce(&Compiler) -> R + Send) -> R {
    trace!("run_compiler");

    // Set parallel mode before thread pool creation, which will create `Lock`s.
    rustc_data_structures::sync::set_dyn_thread_safe_mode(config.opts.unstable_opts.threads > 1);

    util::run_in_thread_pool_with_globals(
        config.opts.edition,
        config.opts.unstable_opts.threads,
        || {
            crate::callbacks::setup_callbacks();

            let handler = EarlyErrorHandler::new(config.opts.error_format);

            let codegen_backend = if let Some(make_codegen_backend) = config.make_codegen_backend {
                make_codegen_backend(&config.opts)
            } else {
                util::get_codegen_backend(
                    &handler,
                    &config.opts.maybe_sysroot,
                    config.opts.unstable_opts.codegen_backend.as_deref(),
                )
            };

            let temps_dir = config.opts.unstable_opts.temps_dir.as_deref().map(PathBuf::from);

            let bundle = match rustc_errors::fluent_bundle(
                config.opts.maybe_sysroot.clone(),
                sysroot_candidates().to_vec(),
                config.opts.unstable_opts.translate_lang.clone(),
                config.opts.unstable_opts.translate_additional_ftl.as_deref(),
                config.opts.unstable_opts.translate_directionality_markers,
            ) {
                Ok(bundle) => bundle,
                Err(e) => {
                    handler.early_error(format!("failed to load fluent bundle: {e}"));
                }
            };

            let mut locale_resources = Vec::from(config.locale_resources);
            locale_resources.push(codegen_backend.locale_resource());

            // target_override is documented to be called before init(), so this is okay
            let target_override = codegen_backend.target_override(&config.opts);

            let mut sess = rustc_session::build_session(
                &handler,
                config.opts,
                CompilerIO {
                    input: config.input,
                    output_dir: config.output_dir,
                    output_file: config.output_file,
                    temps_dir,
                },
                bundle,
                config.registry.clone(),
                locale_resources,
                config.lint_caps,
                config.file_loader,
                target_override,
                util::rustc_version_str().unwrap_or("unknown"),
                config.ice_file,
                config.using_internal_features,
                config.expanded_args,
            );

            codegen_backend.init(&sess);

            let cfg = parse_cfg(&handler, config.crate_cfg);
            let mut cfg = config::build_configuration(&sess, cfg);
            util::add_configuration(&mut cfg, &mut sess, &*codegen_backend);
            sess.parse_sess.config = cfg;

            let mut check_cfg = parse_check_cfg(&handler, config.crate_check_cfg);
            check_cfg.fill_well_known(&sess.target);
            sess.parse_sess.check_config = check_cfg;

            if let Some(parse_sess_created) = config.parse_sess_created {
                parse_sess_created(&mut sess.parse_sess);
            }

            if let Some(hash_untracked_state) = config.hash_untracked_state {
                let mut hasher = StableHasher::new();
                hash_untracked_state(&sess, &mut hasher);
                sess.opts.untracked_state_hash = hasher.finish()
            }

            // Even though the session holds the lint store, we can't build the
            // lint store until after the session exists. And we wait until now
            // so that `register_lints` sees the fully initialized session.
            let mut lint_store = rustc_lint::new_lint_store(sess.enable_internal_lints());
            if let Some(register_lints) = config.register_lints.as_deref() {
                register_lints(&sess, &mut lint_store);
                sess.registered_lints = true;
            }
            sess.lint_store = Some(Lrc::new(lint_store));

            let compiler =
                Compiler { sess, codegen_backend, override_queries: config.override_queries };

            rustc_span::set_source_map(compiler.sess.parse_sess.clone_source_map(), move || {
                let r = {
                    let _sess_abort_error = defer(|| {
                        compiler.sess.finish_diagnostics(&config.registry);
                    });

                    f(&compiler)
                };

                let prof = compiler.sess.prof.clone();

                prof.generic_activity("drop_compiler").run(move || drop(compiler));
                r
            })
        },
    )
}

pub fn try_print_query_stack(
    handler: &Handler,
    num_frames: Option<usize>,
    file: Option<std::fs::File>,
) {
    eprintln!("query stack during panic:");

    // Be careful relying on global state here: this code is called from
    // a panic hook, which means that the global `Handler` may be in a weird
    // state if it was responsible for triggering the panic.
    let i = ty::tls::with_context_opt(|icx| {
        if let Some(icx) = icx {
            ty::print::with_no_queries!(print_query_stack(
                QueryCtxt::new(icx.tcx),
                icx.query,
                handler,
                num_frames,
                file,
            ))
        } else {
            0
        }
    });

    if num_frames == None || num_frames >= Some(i) {
        eprintln!("end of query stack");
    } else {
        eprintln!("we're just showing a limited slice of the query stack");
    }
}
