use crate::util;

use rustc_ast::token;
use rustc_ast::{self as ast, LitKind, MetaItemKind};
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_data_structures::defer;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use rustc_errors::registry::Registry;
use rustc_errors::{ErrorGuaranteed, Handler};
use rustc_lint::LintStore;
use rustc_middle::query::{ExternProviders, Providers};
use rustc_middle::{bug, ty};
use rustc_parse::maybe_new_parser_from_source_str;
use rustc_query_impl::QueryCtxt;
use rustc_query_system::query::print_query_stack;
use rustc_session::config::{self, CheckCfg, ExpectedValues, Input, OutFileName, OutputFilenames};
use rustc_session::parse::{CrateConfig, ParseSess};
use rustc_session::CompilerIO;
use rustc_session::Session;
use rustc_session::{lint, EarlyErrorHandler};
use rustc_span::source_map::{FileLoader, FileName};
use rustc_span::symbol::sym;
use std::path::PathBuf;
use std::result;

pub type Result<T> = result::Result<T, ErrorGuaranteed>;

/// Represents a compiler session. Note that every `Compiler` contains a
/// `Session`, but `Compiler` also contains some things that cannot be in
/// `Session`, due to `Session` being in a crate that has many fewer
/// dependencies than this crate.
///
/// Can be used to run `rustc_interface` queries.
/// Created by passing [`Config`] to [`run_compiler`].
pub struct Compiler {
    pub(crate) sess: Lrc<Session>,
    codegen_backend: Lrc<dyn CodegenBackend>,
    pub(crate) register_lints: Option<Box<dyn Fn(&Session, &mut LintStore) + Send + Sync>>,
    pub(crate) override_queries: Option<fn(&Session, &mut Providers, &mut ExternProviders)>,
}

impl Compiler {
    pub fn session(&self) -> &Lrc<Session> {
        &self.sess
    }
    pub fn codegen_backend(&self) -> &Lrc<dyn CodegenBackend> {
        &self.codegen_backend
    }
    pub fn register_lints(&self) -> &Option<Box<dyn Fn(&Session, &mut LintStore) + Send + Sync>> {
        &self.register_lints
    }
    pub fn build_output_filenames(
        &self,
        sess: &Session,
        attrs: &[ast::Attribute],
    ) -> OutputFilenames {
        util::build_output_filenames(attrs, sess)
    }
}

#[allow(rustc::bad_opt_access)]
pub fn set_thread_safe_mode(sopts: &config::UnstableOptions) {
    rustc_data_structures::sync::set_dyn_thread_safe_mode(sopts.threads > 1);
}

/// Converts strings provided as `--cfg [cfgspec]` into a `crate_cfg`.
pub fn parse_cfgspecs(
    handler: &EarlyErrorHandler,
    cfgspecs: Vec<String>,
) -> FxHashSet<(String, Option<String>)> {
    rustc_span::create_default_session_if_not_set_then(move |_| {
        let cfg = cfgspecs
            .into_iter()
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
            .collect::<CrateConfig>();
        cfg.into_iter().map(|(a, b)| (a.to_string(), b.map(|b| b.to_string()))).collect()
    })
}

/// Converts strings provided as `--check-cfg [specs]` into a `CheckCfg`.
pub fn parse_check_cfg(handler: &EarlyErrorHandler, specs: Vec<String>) -> CheckCfg {
    rustc_span::create_default_session_if_not_set_then(move |_| {
        let mut check_cfg = CheckCfg::default();

        for s in specs {
            let sess = ParseSess::with_silent_emitter(Some(format!(
                "this error occurred on the command line: `--check-cfg={s}`"
            )));
            let filename = FileName::cfg_spec_source_code(&s);

            macro_rules! error {
                ($reason: expr) => {
                    handler.early_error(format!(
                        concat!("invalid `--check-cfg` argument: `{}` (", $reason, ")"),
                        s
                    ))
                };
            }

            let expected_error = || {
                error!(
                    "expected `names(name1, name2, ... nameN)` or \
                        `values(name, \"value1\", \"value2\", ... \"valueN\")`"
                )
            };

            match maybe_new_parser_from_source_str(&sess, filename, s.to_string()) {
                Ok(mut parser) => match parser.parse_meta_item() {
                    Ok(meta_item) if parser.token == token::Eof => {
                        if let Some(args) = meta_item.meta_item_list() {
                            if meta_item.has_name(sym::names) {
                                check_cfg.exhaustive_names = true;
                                for arg in args {
                                    if arg.is_word() && arg.ident().is_some() {
                                        let ident = arg.ident().expect("multi-segment cfg key");
                                        check_cfg
                                            .expecteds
                                            .entry(ident.name.to_string())
                                            .or_insert(ExpectedValues::Any);
                                    } else {
                                        error!("`names()` arguments must be simple identifiers");
                                    }
                                }
                            } else if meta_item.has_name(sym::values) {
                                if let Some((name, values)) = args.split_first() {
                                    if name.is_word() && name.ident().is_some() {
                                        let ident = name.ident().expect("multi-segment cfg key");
                                        let expected_values = check_cfg
                                            .expecteds
                                            .entry(ident.name.to_string())
                                            .and_modify(|expected_values| match expected_values {
                                                ExpectedValues::Some(_) => {}
                                                ExpectedValues::Any => {
                                                    // handle the case where names(...) was done
                                                    // before values by changing to a list
                                                    *expected_values =
                                                        ExpectedValues::Some(FxHashSet::default());
                                                }
                                            })
                                            .or_insert_with(|| {
                                                ExpectedValues::Some(FxHashSet::default())
                                            });

                                        let ExpectedValues::Some(expected_values) = expected_values else {
                                            bug!("`expected_values` should be a list a values")
                                        };

                                        for val in values {
                                            if let Some(LitKind::Str(s, _)) =
                                                val.lit().map(|lit| &lit.kind)
                                            {
                                                expected_values.insert(Some(s.to_string()));
                                            } else {
                                                error!(
                                                    "`values()` arguments must be string literals"
                                                );
                                            }
                                        }

                                        if values.is_empty() {
                                            expected_values.insert(None);
                                        }
                                    } else {
                                        error!(
                                            "`values()` first argument must be a simple identifier"
                                        );
                                    }
                                } else if args.is_empty() {
                                    check_cfg.exhaustive_values = true;
                                } else {
                                    expected_error();
                                }
                            } else {
                                expected_error();
                            }
                        } else {
                            expected_error();
                        }
                    }
                    Ok(..) => expected_error(),
                    Err(err) => {
                        err.cancel();
                        expected_error();
                    }
                },
                Err(errs) => {
                    drop(errs);
                    expected_error();
                }
            }
        }

        check_cfg
    })
}

/// The compiler configuration
pub struct Config {
    /// Command line options
    pub opts: config::Options,

    /// cfg! configuration in addition to the default ones
    pub crate_cfg: FxHashSet<(String, Option<String>)>,
    pub crate_check_cfg: CheckCfg,

    pub input: Input,
    pub output_dir: Option<PathBuf>,
    pub output_file: Option<OutFileName>,
    pub file_loader: Option<Box<dyn FileLoader + Send + Sync>>,
    pub locale_resources: &'static [&'static str],

    pub lint_caps: FxHashMap<lint::LintId, lint::Level>,

    /// This is a callback from the driver that is called when [`ParseSess`] is created.
    pub parse_sess_created: Option<Box<dyn FnOnce(&mut ParseSess) + Send>>,

    /// This is a callback from the driver that is called when we're registering lints;
    /// it is called during plugin registration when we have the LintStore in a non-shared state.
    ///
    /// Note that if you find a Some here you probably want to call that function in the new
    /// function being registered.
    pub register_lints: Option<Box<dyn Fn(&Session, &mut LintStore) + Send + Sync>>,

    /// This is a callback from the driver that is called just after we have populated
    /// the list of queries.
    ///
    /// The second parameter is local providers and the third parameter is external providers.
    pub override_queries: Option<fn(&Session, &mut Providers, &mut ExternProviders)>,

    /// This is a callback from the driver that is called to create a codegen backend.
    pub make_codegen_backend:
        Option<Box<dyn FnOnce(&config::Options) -> Box<dyn CodegenBackend> + Send>>,

    /// Registry of diagnostics codes.
    pub registry: Registry,
}

// JUSTIFICATION: before session exists, only config
#[allow(rustc::bad_opt_access)]
pub fn run_compiler<R: Send>(config: Config, f: impl FnOnce(&Compiler) -> R + Send) -> R {
    trace!("run_compiler");
    util::run_in_thread_pool_with_globals(
        config.opts.edition,
        config.opts.unstable_opts.threads,
        || {
            crate::callbacks::setup_callbacks();

            let registry = &config.registry;

            let handler = EarlyErrorHandler::new(config.opts.error_format);

            let temps_dir = config.opts.unstable_opts.temps_dir.as_deref().map(PathBuf::from);
            let (mut sess, codegen_backend) = util::create_session(
                &handler,
                config.opts,
                config.crate_cfg,
                config.crate_check_cfg,
                config.locale_resources,
                config.file_loader,
                CompilerIO {
                    input: config.input,
                    output_dir: config.output_dir,
                    output_file: config.output_file,
                    temps_dir,
                },
                config.lint_caps,
                config.make_codegen_backend,
                registry.clone(),
            );

            if let Some(parse_sess_created) = config.parse_sess_created {
                parse_sess_created(&mut sess.parse_sess);
            }

            let compiler = Compiler {
                sess: Lrc::new(sess),
                codegen_backend: Lrc::from(codegen_backend),
                register_lints: config.register_lints,
                override_queries: config.override_queries,
            };

            rustc_span::set_source_map(compiler.sess.parse_sess.clone_source_map(), move || {
                let r = {
                    let _sess_abort_error = defer(|| {
                        compiler.sess.finish_diagnostics(registry);
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

pub fn try_print_query_stack(handler: &Handler, num_frames: Option<usize>) {
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
                num_frames
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
