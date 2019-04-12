use crate::util;
use crate::profile;
use crate::passes;
pub use crate::passes::BoxedResolver;

use rustc::lint;
use rustc::session::config::{self, Input, InputsAndOutputs, OutputType};
use rustc::hir::def_id::LOCAL_CRATE;
use rustc::session::{DiagnosticOutput, Session};
use rustc::util::common::ErrorReported;
use rustc::ty::{self, TyCtxt};
use rustc_codegen_utils::codegen_backend::CodegenBackend;
use rustc_data_structures::OnDrop;
use rustc_data_structures::sync::{Lrc, OneThread};
use rustc_data_structures::fx::{FxHashSet, FxHashMap};
use rustc_metadata::cstore::CStore;
use std::io::Write;
use std::path::PathBuf;
use std::result;
use std::sync::{Arc, Mutex};
use std::mem;
use syntax;
use syntax::source_map::{FileLoader, SourceMap};
use syntax_pos::edition;

pub type Result<T> = result::Result<T, ErrorReported>;

/// Represents a compiler session.
/// Can be used run `rustc_interface` queries.
/// Created by passing `Config` to `run_compiler`.
pub struct Compiler {
    pub(crate) sess: Lrc<Session>,
    codegen_backend: Arc<dyn CodegenBackend + Send + Sync>,
    source_map: Lrc<SourceMap>,
    pub(crate) io: InputsAndOutputs,
    pub(crate) cstore: Lrc<CStore>,
    pub(crate) crate_name: Option<String>,
}

impl Compiler {
    pub fn session(&self) -> &Lrc<Session> {
        &self.sess
    }
    pub fn codegen_backend(&self) -> &Arc<dyn CodegenBackend + Send + Sync> {
        &self.codegen_backend
    }
    pub fn cstore(&self) -> &Lrc<CStore> {
        &self.cstore
    }
    pub fn source_map(&self) -> &Lrc<SourceMap> {
        &self.source_map
    }
    pub fn input(&self) -> &Input {
        &self.io.input
    }
    pub fn output_dir(&self) -> &Option<PathBuf> {
        &self.io.output_dir
    }
    pub fn output_file(&self) -> &Option<PathBuf> {
        &self.io.output_file
    }
    pub fn enter<F, R>(self, f: F) -> R
    where
        F: FnOnce(&Compiler, TyCtxt<'_>) -> R
    {
        passes::enter_global_ctxt(&self, f)
    }
    pub fn linker(&self, tcx: TyCtxt<'_>) -> Result<Linker> {
        tcx.ongoing_codegen(LOCAL_CRATE).map(|ongoing_codegen| {
            Linker {
                sess: self.sess.clone(),
                ongoing_codegen,
                codegen_backend: self.codegen_backend.clone(),
            }
        })
    }
    pub fn compile(self) -> Result<()> {
        let link = self.enter(|compiler, tcx| {
            tcx.prepare_outputs(())?;

            if tcx.sess.opts.output_types.contains_key(&OutputType::DepInfo)
                && tcx.sess.opts.output_types.len() == 1
            {
                return Ok(None)
            }

            tcx.lower_ast_to_hir(())?;
            // Drop AST after lowering HIR to free memory
            mem::drop(tcx.expand_macros(()).unwrap().ast_crate.steal());

            compiler.linker(tcx).map(|linker| Some(linker))
        })?;

        // Run linker outside `enter` so GlobalCtxt is freed
        if let Some(linker) = link {
            linker.link()
        } else {
            Ok(())
        }
    }
}

pub struct Linker {
    sess: Lrc<Session>,
    ongoing_codegen: Lrc<ty::OngoingCodegen>,
    codegen_backend: Arc<dyn CodegenBackend + Send + Sync>,
}

impl Linker {
    pub fn link(self) -> Result<()> {
        self.codegen_backend.join_codegen_and_link(
            OneThread::into_inner(self.ongoing_codegen.codegen_object.steal()),
            &self.sess,
            &self.ongoing_codegen.dep_graph,
            &self.ongoing_codegen.outputs,
        ).map_err(|_| ErrorReported)
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
    F: FnOnce(Compiler) -> R,
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
        sess: sess.clone(),
        codegen_backend,
        source_map,
        cstore,
        io: InputsAndOutputs {
            input: config.input,
            input_path: config.input_path,
            output_dir: config.output_dir,
            output_file: config.output_file,
        },
        crate_name: config.crate_name,
    };

    let _sess_abort_error = OnDrop(|| {
        sess.diagnostic().print_error_count(&util::diagnostics_registry());
    });

    if sess.profile_queries() {
        profile::begin(&sess);
    }

    let r = f(compiler);

    if sess.profile_queries() {
        profile::dump(&sess, "profile_queries".to_string())
    }

    r
}

pub fn run_compiler<F, R>(mut config: Config, f: F) -> R
where
    F: FnOnce(Compiler) -> R + Send,
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
