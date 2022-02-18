pub use crate::passes::BoxedResolver;
use crate::util;

use rustc_ast::token;
use rustc_ast::{self as ast, LitKind, MetaItemKind};
use rustc_codegen_ssa::traits::CodegenBackend;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::OnDrop;
use rustc_errors::registry::Registry;
use rustc_errors::{ErrorReported, Handler};
use rustc_lint::LintStore;
use rustc_middle::ty;
use rustc_parse::maybe_new_parser_from_source_str;
use rustc_query_impl::QueryCtxt;
use rustc_session::config::{self, CheckCfg, ErrorOutputType, Input, OutputFilenames};
use rustc_session::early_error;
use rustc_session::lint;
use rustc_session::parse::{CrateConfig, ParseSess};
use rustc_session::{DiagnosticOutput, Session};
use rustc_span::source_map::{FileLoader, FileName};
use rustc_span::symbol::sym;
use std::path::PathBuf;
use std::result;

pub type Result<T> = result::Result<T, ErrorReported>;

/// Represents a compiler session.
///
/// Can be used to run `rustc_interface` queries.
/// Created by passing [`Config`] to [`run_compiler`].
pub struct Compiler {
    pub(crate) sess: Lrc<Session>,
    codegen_backend: Lrc<Box<dyn CodegenBackend>>,
    pub(crate) input: Input,
    pub(crate) input_path: Option<PathBuf>,
    pub(crate) output_dir: Option<PathBuf>,
    pub(crate) output_file: Option<PathBuf>,
    pub(crate) temps_dir: Option<PathBuf>,
    pub(crate) register_lints: Option<Box<dyn Fn(&Session, &mut LintStore) + Send + Sync>>,
    pub(crate) override_queries:
        Option<fn(&Session, &mut ty::query::Providers, &mut ty::query::ExternProviders)>,
}

impl Compiler {
    pub fn session(&self) -> &Lrc<Session> {
        &self.sess
    }
    pub fn codegen_backend(&self) -> &Lrc<Box<dyn CodegenBackend>> {
        &self.codegen_backend
    }
    pub fn input(&self) -> &Input {
        &self.input
    }
    pub fn output_dir(&self) -> &Option<PathBuf> {
        &self.output_dir
    }
    pub fn output_file(&self) -> &Option<PathBuf> {
        &self.output_file
    }
    pub fn temps_dir(&self) -> &Option<PathBuf> {
        &self.temps_dir
    }
    pub fn register_lints(&self) -> &Option<Box<dyn Fn(&Session, &mut LintStore) + Send + Sync>> {
        &self.register_lints
    }
    pub fn build_output_filenames(
        &self,
        sess: &Session,
        attrs: &[ast::Attribute],
    ) -> OutputFilenames {
        util::build_output_filenames(
            &self.input,
            &self.output_dir,
            &self.output_file,
            &self.temps_dir,
            attrs,
            sess,
        )
    }
}

/// Converts strings provided as `--cfg [cfgspec]` into a `crate_cfg`.
pub fn parse_cfgspecs(cfgspecs: Vec<String>) -> FxHashSet<(String, Option<String>)> {
    rustc_span::create_default_session_if_not_set_then(move |_| {
        let cfg = cfgspecs
            .into_iter()
            .map(|s| {
                let sess = ParseSess::with_silent_emitter(Some(format!(
                    "this error occurred on the command line: `--cfg={}`",
                    s
                )));
                let filename = FileName::cfg_spec_source_code(&s);

                macro_rules! error {
                    ($reason: expr) => {
                        early_error(
                            ErrorOutputType::default(),
                            &format!(concat!("invalid `--cfg` argument: `{}` (", $reason, ")"), s),
                        );
                    };
                }

                match maybe_new_parser_from_source_str(&sess, filename, s.to_string()) {
                    Ok(mut parser) => match &mut parser.parse_meta_item() {
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
                    Err(errs) => errs.into_iter().for_each(|mut err| err.cancel()),
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
pub fn parse_check_cfg(specs: Vec<String>) -> CheckCfg {
    rustc_span::create_default_session_if_not_set_then(move |_| {
        let mut cfg = CheckCfg::default();

        'specs: for s in specs {
            let sess = ParseSess::with_silent_emitter(Some(format!(
                "this error occurred on the command line: `--check-cfg={}`",
                s
            )));
            let filename = FileName::cfg_spec_source_code(&s);

            macro_rules! error {
                ($reason: expr) => {
                    early_error(
                        ErrorOutputType::default(),
                        &format!(
                            concat!("invalid `--check-cfg` argument: `{}` (", $reason, ")"),
                            s
                        ),
                    );
                };
            }

            match maybe_new_parser_from_source_str(&sess, filename, s.to_string()) {
                Ok(mut parser) => match &mut parser.parse_meta_item() {
                    Ok(meta_item) if parser.token == token::Eof => {
                        if let Some(args) = meta_item.meta_item_list() {
                            if meta_item.has_name(sym::names) {
                                cfg.names_checked = true;
                                for arg in args {
                                    if arg.is_word() && arg.ident().is_some() {
                                        let ident = arg.ident().expect("multi-segment cfg key");
                                        cfg.names_valid.insert(ident.name.to_string());
                                    } else {
                                        error!("`names()` arguments must be simple identifers");
                                    }
                                }
                                continue 'specs;
                            } else if meta_item.has_name(sym::values) {
                                if let Some((name, values)) = args.split_first() {
                                    if name.is_word() && name.ident().is_some() {
                                        let ident = name.ident().expect("multi-segment cfg key");
                                        cfg.values_checked.insert(ident.to_string());
                                        for val in values {
                                            if let Some(LitKind::Str(s, _)) =
                                                val.literal().map(|lit| &lit.kind)
                                            {
                                                cfg.values_valid
                                                    .insert((ident.to_string(), s.to_string()));
                                            } else {
                                                error!(
                                                    "`values()` arguments must be string literals"
                                                );
                                            }
                                        }

                                        continue 'specs;
                                    } else {
                                        error!(
                                            "`values()` first argument must be a simple identifer"
                                        );
                                    }
                                }
                            }
                        }
                    }
                    Ok(..) => {}
                    Err(err) => err.cancel(),
                },
                Err(errs) => errs.into_iter().for_each(|mut err| err.cancel()),
            }

            error!(
                "expected `names(name1, name2, ... nameN)` or \
                `values(name, \"value1\", \"value2\", ... \"valueN\")`"
            );
        }

        cfg.names_valid.extend(cfg.values_checked.iter().cloned());
        cfg
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
    pub input_path: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
    pub file_loader: Option<Box<dyn FileLoader + Send + Sync>>,
    pub diagnostic_output: DiagnosticOutput,

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
    pub override_queries:
        Option<fn(&Session, &mut ty::query::Providers, &mut ty::query::ExternProviders)>,

    /// This is a callback from the driver that is called to create a codegen backend.
    pub make_codegen_backend:
        Option<Box<dyn FnOnce(&config::Options) -> Box<dyn CodegenBackend> + Send>>,

    /// Registry of diagnostics codes.
    pub registry: Registry,
}

pub fn create_compiler_and_run<R>(config: Config, f: impl FnOnce(&Compiler) -> R) -> R {
    crate::callbacks::setup_callbacks();

    let registry = &config.registry;
    let (mut sess, codegen_backend) = util::create_session(
        config.opts,
        config.crate_cfg,
        config.crate_check_cfg,
        config.diagnostic_output,
        config.file_loader,
        config.input_path.clone(),
        config.lint_caps,
        config.make_codegen_backend,
        registry.clone(),
    );

    if let Some(parse_sess_created) = config.parse_sess_created {
        parse_sess_created(
            &mut Lrc::get_mut(&mut sess)
                .expect("create_session() should never share the returned session")
                .parse_sess,
        );
    }

    let temps_dir = sess.opts.debugging_opts.temps_dir.as_ref().map(|o| PathBuf::from(&o));

    let compiler = Compiler {
        sess,
        codegen_backend,
        input: config.input,
        input_path: config.input_path,
        output_dir: config.output_dir,
        output_file: config.output_file,
        temps_dir,
        register_lints: config.register_lints,
        override_queries: config.override_queries,
    };

    rustc_span::with_source_map(compiler.sess.parse_sess.clone_source_map(), move || {
        let r = {
            let _sess_abort_error = OnDrop(|| {
                compiler.sess.finish_diagnostics(registry);
            });

            f(&compiler)
        };

        let prof = compiler.sess.prof.clone();
        prof.generic_activity("drop_compiler").run(move || drop(compiler));
        r
    })
}

pub fn run_compiler<R: Send>(config: Config, f: impl FnOnce(&Compiler) -> R + Send) -> R {
    tracing::trace!("run_compiler");
    util::run_in_thread_pool_with_globals(
        config.opts.edition,
        config.opts.debugging_opts.threads,
        || create_compiler_and_run(config, f),
    )
}

pub fn try_print_query_stack(handler: &Handler, num_frames: Option<usize>) {
    eprintln!("query stack during panic:");

    // Be careful relying on global state here: this code is called from
    // a panic hook, which means that the global `Handler` may be in a weird
    // state if it was responsible for triggering the panic.
    let i = ty::tls::with_context_opt(|icx| {
        if let Some(icx) = icx {
            QueryCtxt::from_tcx(icx.tcx).try_print_query_stack(icx.query, handler, num_frames)
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
