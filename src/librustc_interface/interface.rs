pub use crate::passes::BoxedResolver;
use crate::util;

use rustc::lint;
use rustc::session::config::{self, ErrorOutputType, Input};
use rustc::session::early_error;
use rustc::session::{DiagnosticOutput, Session};
use rustc::ty;
use rustc::util::common::ErrorReported;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::OnDrop;
use rustc_errors::registry::Registry;
use rustc_lint::LintStore;
use rustc_parse::new_parser_from_source_str;
use rustc_span::edition;
use rustc_span::source_map::{FileLoader, FileName, SourceMap};
use std::path::PathBuf;
use std::result;
use std::sync::{Arc, Mutex};
use syntax::ast::{self, MetaItemKind};
use syntax::sess::ParseSess;
use syntax::token;

pub type Result<T> = result::Result<T, ErrorReported>;

/// Represents a compiler session.
/// Can be used to run `rustc_interface` queries.
/// Created by passing `Config` to `run_compiler`.
pub struct Compiler {
    pub(crate) sess: Lrc<Session>,
    codegen_backend: Lrc<Box<dyn CodegenBackend>>,
    source_map: Lrc<SourceMap>,
    pub(crate) input: Input,
    pub(crate) input_path: Option<PathBuf>,
    pub(crate) output_dir: Option<PathBuf>,
    pub(crate) output_file: Option<PathBuf>,
    pub(crate) crate_name: Option<String>,
    pub(crate) register_lints: Option<Box<dyn Fn(&Session, &mut LintStore) + Send + Sync>>,
    pub(crate) override_queries:
        Option<fn(&Session, &mut ty::query::Providers<'_>, &mut ty::query::Providers<'_>)>,
}

impl Compiler {
    pub fn session(&self) -> &Lrc<Session> {
        &self.sess
    }
    pub fn codegen_backend(&self) -> &Lrc<Box<dyn CodegenBackend>> {
        &self.codegen_backend
    }
    pub fn source_map(&self) -> &Lrc<SourceMap> {
        &self.source_map
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
}

/// Converts strings provided as `--cfg [cfgspec]` into a `crate_cfg`.
pub fn parse_cfgspecs(cfgspecs: Vec<String>) -> FxHashSet<(String, Option<String>)> {
    syntax::with_default_globals(move || {
        let cfg = cfgspecs
            .into_iter()
            .map(|s| {
                let sess = ParseSess::with_silent_emitter();
                let filename = FileName::cfg_spec_source_code(&s);
                let mut parser = new_parser_from_source_str(&sess, filename, s.to_string());

                macro_rules! error {
                    ($reason: expr) => {
                        early_error(
                            ErrorOutputType::default(),
                            &format!(concat!("invalid `--cfg` argument: `{}` (", $reason, ")"), s),
                        );
                    };
                }

                match &mut parser.parse_meta_item() {
                    Ok(meta_item) if parser.token == token::Eof => {
                        if meta_item.path.segments.len() != 1 {
                            error!("argument key must be an identifier");
                        }
                        match &meta_item.kind {
                            MetaItemKind::List(..) => {
                                error!(r#"expected `key` or `key="value"`"#);
                            }
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
                }

                error!(r#"expected `key` or `key="value"`"#);
            })
            .collect::<ast::CrateConfig>();
        cfg.into_iter().map(|(a, b)| (a.to_string(), b.map(|b| b.to_string()))).collect()
    })
}

/// The compiler configuration
pub struct Config {
    /// Command line options
    pub opts: config::Options,

    /// cfg! configuration in addition to the default ones
    pub crate_cfg: FxHashSet<(String, Option<String>)>,

    pub input: Input,
    pub input_path: Option<PathBuf>,
    pub output_dir: Option<PathBuf>,
    pub output_file: Option<PathBuf>,
    pub file_loader: Option<Box<dyn FileLoader + Send + Sync>>,
    pub diagnostic_output: DiagnosticOutput,

    /// Set to capture stderr output during compiler execution
    pub stderr: Option<Arc<Mutex<Vec<u8>>>>,

    pub crate_name: Option<String>,
    pub lint_caps: FxHashMap<lint::LintId, lint::Level>,

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
        Option<fn(&Session, &mut ty::query::Providers<'_>, &mut ty::query::Providers<'_>)>,

    /// Registry of diagnostics codes.
    pub registry: Registry,
}

pub fn run_compiler_in_existing_thread_pool<R>(
    config: Config,
    f: impl FnOnce(&Compiler) -> R,
) -> R {
    let registry = &config.registry;
    let (sess, codegen_backend, source_map) = util::create_session(
        config.opts,
        config.crate_cfg,
        config.diagnostic_output,
        config.file_loader,
        config.input_path.clone(),
        config.lint_caps,
        registry.clone(),
    );

    let compiler = Compiler {
        sess,
        codegen_backend,
        source_map,
        input: config.input,
        input_path: config.input_path,
        output_dir: config.output_dir,
        output_file: config.output_file,
        crate_name: config.crate_name,
        register_lints: config.register_lints,
        override_queries: config.override_queries,
    };

    let r = {
        let _sess_abort_error = OnDrop(|| {
            compiler.sess.diagnostic().print_error_count(registry);
        });

        f(&compiler)
    };

    let prof = compiler.sess.prof.clone();
    prof.generic_activity("drop_compiler").run(move || drop(compiler));
    r
}

pub fn run_compiler<R: Send>(mut config: Config, f: impl FnOnce(&Compiler) -> R + Send) -> R {
    let stderr = config.stderr.take();
    util::spawn_thread_pool(
        config.opts.edition,
        config.opts.debugging_opts.threads,
        &stderr,
        || run_compiler_in_existing_thread_pool(config, f),
    )
}

pub fn default_thread_pool<R: Send>(edition: edition::Edition, f: impl FnOnce() -> R + Send) -> R {
    // the 1 here is duplicating code in config.opts.debugging_opts.threads
    // which also defaults to 1; it ultimately doesn't matter as the default
    // isn't threaded, and just ignores this parameter
    util::spawn_thread_pool(edition, 1, &None, f)
}
