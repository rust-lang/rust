use crate::queries::Queries;
use crate::util;
use crate::profile;
pub use crate::passes::BoxedResolver;

use rustc::lint;
use rustc::session::config::{self, Input};
use rustc::session::{DiagnosticOutput, Session};
use rustc::util::common::ErrorReported;
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc_data_structures::OnDrop;
use rustc_data_structures::sync::Lrc;
use rustc_data_structures::fx::{FxHashSet, FxHashMap};
use rustc_metadata::cstore::CStore;
use std::io::Write;
use std::path::PathBuf;
use std::result;
use std::sync::{Arc, Mutex};
use syntax;
use syntax::source_map::{FileLoader, SourceMap};
use syntax_pos::edition;

pub type Result<T> = result::Result<T, ErrorReported>;

/// Represents a compiler session.
/// Can be used run `rustc_interface` queries.
/// Created by passing `Config` to `run_compiler`.
pub struct Compiler {
    pub(crate) sess: Lrc<Session>,
    codegen_backend: Lrc<Box<dyn CodegenBackend>>,
    source_map: Lrc<SourceMap>,
    pub(crate) input: Input,
    pub(crate) input_path: Option<PathBuf>,
    pub(crate) output_dir: Option<PathBuf>,
    pub(crate) output_file: Option<PathBuf>,
    pub(crate) queries: Queries,
    pub(crate) cstore: Lrc<CStore>,
    pub(crate) crate_name: Option<String>,
}

impl Compiler {
    pub fn session(&self) -> &Lrc<Session> {
        &self.sess
    }
    pub fn codegen_backend(&self) -> &Lrc<Box<dyn CodegenBackend>> {
        &self.codegen_backend
    }
    pub fn cstore(&self) -> &Lrc<CStore> {
        &self.cstore
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
}

pub fn run_compiler_in_existing_thread_pool<F, R>(config: Config, f: F) -> R
where
    F: FnOnce(&Compiler) -> R,
{
    let (sess, codegen_backend, source_map) = util::create_session(
        config.opts,
        config.crate_cfg,
        config.diagnostic_output,
        config.file_loader,
        config.input_path.clone(),
        config.lint_caps,
    );

    let cstore = Lrc::new(CStore::new(codegen_backend.metadata_loader()));

    let compiler = Compiler {
        sess,
        codegen_backend,
        source_map,
        cstore,
        input: config.input,
        input_path: config.input_path,
        output_dir: config.output_dir,
        output_file: config.output_file,
        queries: Default::default(),
        crate_name: config.crate_name,
    };

    let _sess_abort_error = OnDrop(|| {
        compiler.sess.diagnostic().print_error_count(&util::diagnostics_registry());
    });

    if compiler.sess.profile_queries() {
        profile::begin(&compiler.sess);
    }

    let r = f(&compiler);

    if compiler.sess.profile_queries() {
        profile::dump(&compiler.sess, "profile_queries".to_string())
    }

    r
}

pub fn run_compiler<F, R>(mut config: Config, f: F) -> R
where
    F: FnOnce(&Compiler) -> R + Send,
    R: Send,
{
    let stderr = config.stderr.take();
    util::spawn_thread_pool(
        config.opts.edition,
        config.opts.debugging_opts.threads,
        &stderr,
        || run_compiler_in_existing_thread_pool(config, f),
    )
}

pub fn default_thread_pool<F, R>(edition: edition::Edition, f: F) -> R
where
    F: FnOnce() -> R + Send,
    R: Send,
{
    util::spawn_thread_pool(edition, None, &None, f)
}
