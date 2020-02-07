use crate::code_stats::CodeStats;
pub use crate::code_stats::{DataTypeKind, FieldInfo, SizeKind, VariantInfo};

use crate::cgu_reuse_tracker::CguReuseTracker;
use rustc_data_structures::fingerprint::Fingerprint;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};

use crate::config::{self, OutputType, PrintRequest, Sanitizer, SwitchWithOptPath};
use crate::filesearch;
use crate::lint;
use crate::search_paths::{PathKind, SearchPath};
use rustc_data_structures::profiling::duration_to_secs_str;
use rustc_errors::ErrorReported;

use rustc_data_structures::base_n;
use rustc_data_structures::impl_stable_hash_via_hash;
use rustc_data_structures::sync::{
    self, AtomicU64, AtomicUsize, Lock, Lrc, Once, OneThread, Ordering, Ordering::SeqCst,
};

use crate::parse::ParseSess;
use rustc_errors::annotate_snippet_emitter_writer::AnnotateSnippetEmitterWriter;
use rustc_errors::emitter::HumanReadableErrorType;
use rustc_errors::emitter::{Emitter, EmitterWriter};
use rustc_errors::json::JsonEmitter;
use rustc_errors::{Applicability, DiagnosticBuilder, DiagnosticId};
use rustc_span::edition::Edition;
use rustc_span::source_map;
use rustc_span::{MultiSpan, Span};

use rustc_data_structures::flock;
use rustc_data_structures::jobserver::{self, Client};
use rustc_data_structures::profiling::{SelfProfiler, SelfProfilerRef};
use rustc_target::spec::{PanicStrategy, RelroLevel, Target, TargetTriple};

use std::cell::{self, RefCell};
use std::env;
use std::fmt;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

pub struct OptimizationFuel {
    /// If `-zfuel=crate=n` is specified, initially set to `n`, otherwise `0`.
    remaining: u64,
    /// We're rejecting all further optimizations.
    out_of_fuel: bool,
}

/// Represents the data associated with a compilation
/// session for a single crate.
pub struct Session {
    pub target: config::Config,
    pub host: Target,
    pub opts: config::Options,
    pub host_tlib_path: SearchPath,
    /// `None` if the host and target are the same.
    pub target_tlib_path: Option<SearchPath>,
    pub parse_sess: ParseSess,
    pub sysroot: PathBuf,
    /// The name of the root source file of the crate, in the local file system.
    /// `None` means that there is no source file.
    pub local_crate_source_file: Option<PathBuf>,
    /// The directory the compiler has been executed in plus a flag indicating
    /// if the value stored here has been affected by path remapping.
    pub working_dir: (PathBuf, bool),

    /// Set of `(DiagnosticId, Option<Span>, message)` tuples tracking
    /// (sub)diagnostics that have been set once, but should not be set again,
    /// in order to avoid redundantly verbose output (Issue #24690, #44953).
    pub one_time_diagnostics: Lock<FxHashSet<(DiagnosticMessageId, Option<Span>, String)>>,
    pub crate_types: Once<Vec<config::CrateType>>,
    /// The `crate_disambiguator` is constructed out of all the `-C metadata`
    /// arguments passed to the compiler. Its value together with the crate-name
    /// forms a unique global identifier for the crate. It is used to allow
    /// multiple crates with the same name to coexist. See the
    /// `rustc_codegen_llvm::back::symbol_names` module for more information.
    pub crate_disambiguator: Once<CrateDisambiguator>,

    features: Once<rustc_feature::Features>,

    /// The maximum recursion limit for potentially infinitely recursive
    /// operations such as auto-dereference and monomorphization.
    pub recursion_limit: Once<usize>,

    /// The maximum length of types during monomorphization.
    pub type_length_limit: Once<usize>,

    /// Map from imported macro spans (which consist of
    /// the localized span for the macro body) to the
    /// macro name and definition span in the source crate.
    pub imported_macro_spans: OneThread<RefCell<FxHashMap<Span, (String, Span)>>>,

    incr_comp_session: OneThread<RefCell<IncrCompSession>>,
    /// Used for incremental compilation tests. Will only be populated if
    /// `-Zquery-dep-graph` is specified.
    pub cgu_reuse_tracker: CguReuseTracker,

    /// Used by `-Z self-profile`.
    pub prof: SelfProfilerRef,

    /// Some measurements that are being gathered during compilation.
    pub perf_stats: PerfStats,

    /// Data about code being compiled, gathered during compilation.
    pub code_stats: CodeStats,

    /// If `-zfuel=crate=n` is specified, `Some(crate)`.
    optimization_fuel_crate: Option<String>,

    /// Tracks fuel info if `-zfuel=crate=n` is specified.
    optimization_fuel: Lock<OptimizationFuel>,

    // The next two are public because the driver needs to read them.
    /// If `-zprint-fuel=crate`, `Some(crate)`.
    pub print_fuel_crate: Option<String>,
    /// Always set to zero and incremented so that we can print fuel expended by a crate.
    pub print_fuel: AtomicU64,

    /// Loaded up early on in the initialization of this `Session` to avoid
    /// false positives about a job server in our environment.
    pub jobserver: Client,

    /// Cap lint level specified by a driver specifically.
    pub driver_lint_caps: FxHashMap<lint::LintId, lint::Level>,

    /// `Span`s of trait methods that weren't found to avoid emitting object safety errors
    pub trait_methods_not_found: Lock<FxHashSet<Span>>,

    /// Mapping from ident span to path span for paths that don't exist as written, but that
    /// exist under `std`. For example, wrote `str::from_utf8` instead of `std::str::from_utf8`.
    pub confused_type_with_std_module: Lock<FxHashMap<Span, Span>>,

    /// Path for libraries that will take preference over libraries shipped by Rust.
    /// Used by windows-gnu targets to priortize system mingw-w64 libraries.
    pub system_library_path: OneThread<RefCell<Option<Option<PathBuf>>>>,
}

pub struct PerfStats {
    /// The accumulated time spent on computing symbol hashes.
    pub symbol_hash_time: Lock<Duration>,
    /// The accumulated time spent decoding def path tables from metadata.
    pub decode_def_path_tables_time: Lock<Duration>,
    /// Total number of values canonicalized queries constructed.
    pub queries_canonicalized: AtomicUsize,
    /// Number of times this query is invoked.
    pub normalize_ty_after_erasing_regions: AtomicUsize,
    /// Number of times this query is invoked.
    pub normalize_projection_ty: AtomicUsize,
}

/// Enum to support dispatch of one-time diagnostics (in `Session.diag_once`).
enum DiagnosticBuilderMethod {
    Note,
    SpanNote,
    SpanSuggestion(String), // suggestion
                            // Add more variants as needed to support one-time diagnostics.
}

/// Diagnostic message ID, used by `Session.one_time_diagnostics` to avoid
/// emitting the same message more than once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DiagnosticMessageId {
    ErrorId(u16), // EXXXX error code as integer
    LintId(lint::LintId),
    StabilityId(Option<NonZeroU32>), // issue number
}

impl From<&'static lint::Lint> for DiagnosticMessageId {
    fn from(lint: &'static lint::Lint) -> Self {
        DiagnosticMessageId::LintId(lint::LintId::of(lint))
    }
}

impl Session {
    pub fn local_crate_disambiguator(&self) -> CrateDisambiguator {
        *self.crate_disambiguator.get()
    }

    pub fn struct_span_warn<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> DiagnosticBuilder<'_> {
        self.diagnostic().struct_span_warn(sp, msg)
    }
    pub fn struct_span_warn_with_code<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: &str,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_> {
        self.diagnostic().struct_span_warn_with_code(sp, msg, code)
    }
    pub fn struct_warn(&self, msg: &str) -> DiagnosticBuilder<'_> {
        self.diagnostic().struct_warn(msg)
    }
    pub fn struct_span_err<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> DiagnosticBuilder<'_> {
        self.diagnostic().struct_span_err(sp, msg)
    }
    pub fn struct_span_err_with_code<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: &str,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_> {
        self.diagnostic().struct_span_err_with_code(sp, msg, code)
    }
    // FIXME: This method should be removed (every error should have an associated error code).
    pub fn struct_err(&self, msg: &str) -> DiagnosticBuilder<'_> {
        self.diagnostic().struct_err(msg)
    }
    pub fn struct_err_with_code(&self, msg: &str, code: DiagnosticId) -> DiagnosticBuilder<'_> {
        self.diagnostic().struct_err_with_code(msg, code)
    }
    pub fn struct_span_fatal<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> DiagnosticBuilder<'_> {
        self.diagnostic().struct_span_fatal(sp, msg)
    }
    pub fn struct_span_fatal_with_code<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: &str,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_> {
        self.diagnostic().struct_span_fatal_with_code(sp, msg, code)
    }
    pub fn struct_fatal(&self, msg: &str) -> DiagnosticBuilder<'_> {
        self.diagnostic().struct_fatal(msg)
    }

    pub fn span_fatal<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> ! {
        self.diagnostic().span_fatal(sp, msg).raise()
    }
    pub fn span_fatal_with_code<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: &str,
        code: DiagnosticId,
    ) -> ! {
        self.diagnostic().span_fatal_with_code(sp, msg, code).raise()
    }
    pub fn fatal(&self, msg: &str) -> ! {
        self.diagnostic().fatal(msg).raise()
    }
    pub fn span_err_or_warn<S: Into<MultiSpan>>(&self, is_warning: bool, sp: S, msg: &str) {
        if is_warning {
            self.span_warn(sp, msg);
        } else {
            self.span_err(sp, msg);
        }
    }
    pub fn span_err<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.diagnostic().span_err(sp, msg)
    }
    pub fn span_err_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: DiagnosticId) {
        self.diagnostic().span_err_with_code(sp, &msg, code)
    }
    pub fn err(&self, msg: &str) {
        self.diagnostic().err(msg)
    }
    pub fn err_count(&self) -> usize {
        self.diagnostic().err_count()
    }
    pub fn has_errors(&self) -> bool {
        self.diagnostic().has_errors()
    }
    pub fn has_errors_or_delayed_span_bugs(&self) -> bool {
        self.diagnostic().has_errors_or_delayed_span_bugs()
    }
    pub fn abort_if_errors(&self) {
        self.diagnostic().abort_if_errors();
    }
    pub fn compile_status(&self) -> Result<(), ErrorReported> {
        if self.has_errors() {
            self.diagnostic().emit_stashed_diagnostics();
            Err(ErrorReported)
        } else {
            Ok(())
        }
    }
    // FIXME(matthewjasper) Remove this method, it should never be needed.
    pub fn track_errors<F, T>(&self, f: F) -> Result<T, ErrorReported>
    where
        F: FnOnce() -> T,
    {
        let old_count = self.err_count();
        let result = f();
        let errors = self.err_count() - old_count;
        if errors == 0 { Ok(result) } else { Err(ErrorReported) }
    }
    pub fn span_warn<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.diagnostic().span_warn(sp, msg)
    }
    pub fn span_warn_with_code<S: Into<MultiSpan>>(&self, sp: S, msg: &str, code: DiagnosticId) {
        self.diagnostic().span_warn_with_code(sp, msg, code)
    }
    pub fn warn(&self, msg: &str) {
        self.diagnostic().warn(msg)
    }
    pub fn opt_span_warn<S: Into<MultiSpan>>(&self, opt_sp: Option<S>, msg: &str) {
        match opt_sp {
            Some(sp) => self.span_warn(sp, msg),
            None => self.warn(msg),
        }
    }
    /// Delay a span_bug() call until abort_if_errors()
    pub fn delay_span_bug<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.diagnostic().delay_span_bug(sp, msg)
    }
    pub fn note_without_error(&self, msg: &str) {
        self.diagnostic().note_without_error(msg)
    }
    pub fn span_note_without_error<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.diagnostic().span_note_without_error(sp, msg)
    }

    pub fn diagnostic(&self) -> &rustc_errors::Handler {
        &self.parse_sess.span_diagnostic
    }

    /// Analogous to calling methods on the given `DiagnosticBuilder`, but
    /// deduplicates on lint ID, span (if any), and message for this `Session`
    fn diag_once<'a, 'b>(
        &'a self,
        diag_builder: &'b mut DiagnosticBuilder<'a>,
        method: DiagnosticBuilderMethod,
        msg_id: DiagnosticMessageId,
        message: &str,
        span_maybe: Option<Span>,
    ) {
        let id_span_message = (msg_id, span_maybe, message.to_owned());
        let fresh = self.one_time_diagnostics.borrow_mut().insert(id_span_message);
        if fresh {
            match method {
                DiagnosticBuilderMethod::Note => {
                    diag_builder.note(message);
                }
                DiagnosticBuilderMethod::SpanNote => {
                    let span = span_maybe.expect("`span_note` needs a span");
                    diag_builder.span_note(span, message);
                }
                DiagnosticBuilderMethod::SpanSuggestion(suggestion) => {
                    let span = span_maybe.expect("`span_suggestion_*` needs a span");
                    diag_builder.span_suggestion(
                        span,
                        message,
                        suggestion,
                        Applicability::Unspecified,
                    );
                }
            }
        }
    }

    pub fn diag_span_note_once<'a, 'b>(
        &'a self,
        diag_builder: &'b mut DiagnosticBuilder<'a>,
        msg_id: DiagnosticMessageId,
        span: Span,
        message: &str,
    ) {
        self.diag_once(
            diag_builder,
            DiagnosticBuilderMethod::SpanNote,
            msg_id,
            message,
            Some(span),
        );
    }

    pub fn diag_note_once<'a, 'b>(
        &'a self,
        diag_builder: &'b mut DiagnosticBuilder<'a>,
        msg_id: DiagnosticMessageId,
        message: &str,
    ) {
        self.diag_once(diag_builder, DiagnosticBuilderMethod::Note, msg_id, message, None);
    }

    pub fn diag_span_suggestion_once<'a, 'b>(
        &'a self,
        diag_builder: &'b mut DiagnosticBuilder<'a>,
        msg_id: DiagnosticMessageId,
        span: Span,
        message: &str,
        suggestion: String,
    ) {
        self.diag_once(
            diag_builder,
            DiagnosticBuilderMethod::SpanSuggestion(suggestion),
            msg_id,
            message,
            Some(span),
        );
    }

    pub fn source_map(&self) -> &source_map::SourceMap {
        self.parse_sess.source_map()
    }
    pub fn verbose(&self) -> bool {
        self.opts.debugging_opts.verbose
    }
    pub fn time_passes(&self) -> bool {
        self.opts.debugging_opts.time_passes || self.opts.debugging_opts.time
    }
    pub fn instrument_mcount(&self) -> bool {
        self.opts.debugging_opts.instrument_mcount
    }
    pub fn time_llvm_passes(&self) -> bool {
        self.opts.debugging_opts.time_llvm_passes
    }
    pub fn meta_stats(&self) -> bool {
        self.opts.debugging_opts.meta_stats
    }
    pub fn asm_comments(&self) -> bool {
        self.opts.debugging_opts.asm_comments
    }
    pub fn verify_llvm_ir(&self) -> bool {
        self.opts.debugging_opts.verify_llvm_ir || cfg!(always_verify_llvm_ir)
    }
    pub fn borrowck_stats(&self) -> bool {
        self.opts.debugging_opts.borrowck_stats
    }
    pub fn print_llvm_passes(&self) -> bool {
        self.opts.debugging_opts.print_llvm_passes
    }
    pub fn binary_dep_depinfo(&self) -> bool {
        self.opts.debugging_opts.binary_dep_depinfo
    }

    /// Gets the features enabled for the current compilation session.
    /// DO NOT USE THIS METHOD if there is a TyCtxt available, as it circumvents
    /// dependency tracking. Use tcx.features() instead.
    #[inline]
    pub fn features_untracked(&self) -> &rustc_feature::Features {
        self.features.get()
    }

    pub fn init_features(&self, features: rustc_feature::Features) {
        self.features.set(features);
    }

    /// Calculates the flavor of LTO to use for this compilation.
    pub fn lto(&self) -> config::Lto {
        // If our target has codegen requirements ignore the command line
        if self.target.target.options.requires_lto {
            return config::Lto::Fat;
        }

        // If the user specified something, return that. If they only said `-C
        // lto` and we've for whatever reason forced off ThinLTO via the CLI,
        // then ensure we can't use a ThinLTO.
        match self.opts.cg.lto {
            config::LtoCli::Unspecified => {
                // The compiler was invoked without the `-Clto` flag. Fall
                // through to the default handling
            }
            config::LtoCli::No => {
                // The user explicitly opted out of any kind of LTO
                return config::Lto::No;
            }
            config::LtoCli::Yes | config::LtoCli::Fat | config::LtoCli::NoParam => {
                // All of these mean fat LTO
                return config::Lto::Fat;
            }
            config::LtoCli::Thin => {
                return if self.opts.cli_forced_thinlto_off {
                    config::Lto::Fat
                } else {
                    config::Lto::Thin
                };
            }
        }

        // Ok at this point the target doesn't require anything and the user
        // hasn't asked for anything. Our next decision is whether or not
        // we enable "auto" ThinLTO where we use multiple codegen units and
        // then do ThinLTO over those codegen units. The logic below will
        // either return `No` or `ThinLocal`.

        // If processing command line options determined that we're incompatible
        // with ThinLTO (e.g., `-C lto --emit llvm-ir`) then return that option.
        if self.opts.cli_forced_thinlto_off {
            return config::Lto::No;
        }

        // If `-Z thinlto` specified process that, but note that this is mostly
        // a deprecated option now that `-C lto=thin` exists.
        if let Some(enabled) = self.opts.debugging_opts.thinlto {
            if enabled {
                return config::Lto::ThinLocal;
            } else {
                return config::Lto::No;
            }
        }

        // If there's only one codegen unit and LTO isn't enabled then there's
        // no need for ThinLTO so just return false.
        if self.codegen_units() == 1 {
            return config::Lto::No;
        }

        // Now we're in "defaults" territory. By default we enable ThinLTO for
        // optimized compiles (anything greater than O0).
        match self.opts.optimize {
            config::OptLevel::No => config::Lto::No,
            _ => config::Lto::ThinLocal,
        }
    }

    /// Returns the panic strategy for this compile session. If the user explicitly selected one
    /// using '-C panic', use that, otherwise use the panic strategy defined by the target.
    pub fn panic_strategy(&self) -> PanicStrategy {
        self.opts.cg.panic.unwrap_or(self.target.target.options.panic_strategy)
    }
    pub fn fewer_names(&self) -> bool {
        let more_names = self.opts.output_types.contains_key(&OutputType::LlvmAssembly)
            || self.opts.output_types.contains_key(&OutputType::Bitcode);

        // Address sanitizer and memory sanitizer use alloca name when reporting an issue.
        let more_names = match self.opts.debugging_opts.sanitizer {
            Some(Sanitizer::Address) => true,
            Some(Sanitizer::Memory) => true,
            _ => more_names,
        };

        self.opts.debugging_opts.fewer_names || !more_names
    }

    pub fn no_landing_pads(&self) -> bool {
        self.opts.debugging_opts.no_landing_pads || self.panic_strategy() == PanicStrategy::Abort
    }
    pub fn unstable_options(&self) -> bool {
        self.opts.debugging_opts.unstable_options
    }
    pub fn overflow_checks(&self) -> bool {
        self.opts
            .cg
            .overflow_checks
            .or(self.opts.debugging_opts.force_overflow_checks)
            .unwrap_or(self.opts.debug_assertions)
    }

    pub fn crt_static(&self) -> bool {
        // If the target does not opt in to crt-static support, use its default.
        if self.target.target.options.crt_static_respected {
            self.crt_static_feature()
        } else {
            self.target.target.options.crt_static_default
        }
    }

    pub fn crt_static_feature(&self) -> bool {
        let requested_features = self.opts.cg.target_feature.split(',');
        let found_negative = requested_features.clone().any(|r| r == "-crt-static");
        let found_positive = requested_features.clone().any(|r| r == "+crt-static");

        // If the target we're compiling for requests a static crt by default,
        // then see if the `-crt-static` feature was passed to disable that.
        // Otherwise if we don't have a static crt by default then see if the
        // `+crt-static` feature was passed.
        if self.target.target.options.crt_static_default { !found_negative } else { found_positive }
    }

    pub fn must_not_eliminate_frame_pointers(&self) -> bool {
        // "mcount" function relies on stack pointer.
        // See <https://sourceware.org/binutils/docs/gprof/Implementation.html>.
        if self.instrument_mcount() {
            true
        } else if let Some(x) = self.opts.cg.force_frame_pointers {
            x
        } else {
            !self.target.target.options.eliminate_frame_pointer
        }
    }

    /// Returns the symbol name for the registrar function,
    /// given the crate `Svh` and the function `DefIndex`.
    pub fn generate_plugin_registrar_symbol(&self, disambiguator: CrateDisambiguator) -> String {
        format!("__rustc_plugin_registrar_{}__", disambiguator.to_fingerprint().to_hex())
    }

    pub fn generate_proc_macro_decls_symbol(&self, disambiguator: CrateDisambiguator) -> String {
        format!("__rustc_proc_macro_decls_{}__", disambiguator.to_fingerprint().to_hex())
    }

    pub fn target_filesearch(&self, kind: PathKind) -> filesearch::FileSearch<'_> {
        filesearch::FileSearch::new(
            &self.sysroot,
            self.opts.target_triple.triple(),
            &self.opts.search_paths,
            // `target_tlib_path == None` means it's the same as `host_tlib_path`.
            self.target_tlib_path.as_ref().unwrap_or(&self.host_tlib_path),
            kind,
        )
    }
    pub fn host_filesearch(&self, kind: PathKind) -> filesearch::FileSearch<'_> {
        filesearch::FileSearch::new(
            &self.sysroot,
            config::host_triple(),
            &self.opts.search_paths,
            &self.host_tlib_path,
            kind,
        )
    }

    pub fn set_incr_session_load_dep_graph(&self, load: bool) {
        let mut incr_comp_session = self.incr_comp_session.borrow_mut();

        if let IncrCompSession::Active { ref mut load_dep_graph, .. } = *incr_comp_session {
            *load_dep_graph = load;
        }
    }

    pub fn incr_session_load_dep_graph(&self) -> bool {
        let incr_comp_session = self.incr_comp_session.borrow();
        match *incr_comp_session {
            IncrCompSession::Active { load_dep_graph, .. } => load_dep_graph,
            _ => false,
        }
    }

    pub fn init_incr_comp_session(
        &self,
        session_dir: PathBuf,
        lock_file: flock::Lock,
        load_dep_graph: bool,
    ) {
        let mut incr_comp_session = self.incr_comp_session.borrow_mut();

        if let IncrCompSession::NotInitialized = *incr_comp_session {
        } else {
            panic!("Trying to initialize IncrCompSession `{:?}`", *incr_comp_session)
        }

        *incr_comp_session =
            IncrCompSession::Active { session_directory: session_dir, lock_file, load_dep_graph };
    }

    pub fn finalize_incr_comp_session(&self, new_directory_path: PathBuf) {
        let mut incr_comp_session = self.incr_comp_session.borrow_mut();

        if let IncrCompSession::Active { .. } = *incr_comp_session {
        } else {
            panic!("trying to finalize `IncrCompSession` `{:?}`", *incr_comp_session);
        }

        // Note: this will also drop the lock file, thus unlocking the directory.
        *incr_comp_session = IncrCompSession::Finalized { session_directory: new_directory_path };
    }

    pub fn mark_incr_comp_session_as_invalid(&self) {
        let mut incr_comp_session = self.incr_comp_session.borrow_mut();

        let session_directory = match *incr_comp_session {
            IncrCompSession::Active { ref session_directory, .. } => session_directory.clone(),
            IncrCompSession::InvalidBecauseOfErrors { .. } => return,
            _ => panic!("trying to invalidate `IncrCompSession` `{:?}`", *incr_comp_session),
        };

        // Note: this will also drop the lock file, thus unlocking the directory.
        *incr_comp_session = IncrCompSession::InvalidBecauseOfErrors { session_directory };
    }

    pub fn incr_comp_session_dir(&self) -> cell::Ref<'_, PathBuf> {
        let incr_comp_session = self.incr_comp_session.borrow();
        cell::Ref::map(incr_comp_session, |incr_comp_session| match *incr_comp_session {
            IncrCompSession::NotInitialized => panic!(
                "trying to get session directory from `IncrCompSession`: {:?}",
                *incr_comp_session,
            ),
            IncrCompSession::Active { ref session_directory, .. }
            | IncrCompSession::Finalized { ref session_directory }
            | IncrCompSession::InvalidBecauseOfErrors { ref session_directory } => {
                session_directory
            }
        })
    }

    pub fn incr_comp_session_dir_opt(&self) -> Option<cell::Ref<'_, PathBuf>> {
        self.opts.incremental.as_ref().map(|_| self.incr_comp_session_dir())
    }

    pub fn print_perf_stats(&self) {
        println!(
            "Total time spent computing symbol hashes:      {}",
            duration_to_secs_str(*self.perf_stats.symbol_hash_time.lock())
        );
        println!(
            "Total time spent decoding DefPath tables:      {}",
            duration_to_secs_str(*self.perf_stats.decode_def_path_tables_time.lock())
        );
        println!(
            "Total queries canonicalized:                   {}",
            self.perf_stats.queries_canonicalized.load(Ordering::Relaxed)
        );
        println!(
            "normalize_ty_after_erasing_regions:            {}",
            self.perf_stats.normalize_ty_after_erasing_regions.load(Ordering::Relaxed)
        );
        println!(
            "normalize_projection_ty:                       {}",
            self.perf_stats.normalize_projection_ty.load(Ordering::Relaxed)
        );
    }

    /// We want to know if we're allowed to do an optimization for crate foo from -z fuel=foo=n.
    /// This expends fuel if applicable, and records fuel if applicable.
    pub fn consider_optimizing<T: Fn() -> String>(&self, crate_name: &str, msg: T) -> bool {
        let mut ret = true;
        if let Some(ref c) = self.optimization_fuel_crate {
            if c == crate_name {
                assert_eq!(self.threads(), 1);
                let mut fuel = self.optimization_fuel.lock();
                ret = fuel.remaining != 0;
                if fuel.remaining == 0 && !fuel.out_of_fuel {
                    eprintln!("optimization-fuel-exhausted: {}", msg());
                    fuel.out_of_fuel = true;
                } else if fuel.remaining > 0 {
                    fuel.remaining -= 1;
                }
            }
        }
        if let Some(ref c) = self.print_fuel_crate {
            if c == crate_name {
                assert_eq!(self.threads(), 1);
                self.print_fuel.fetch_add(1, SeqCst);
            }
        }
        ret
    }

    /// Returns the number of query threads that should be used for this
    /// compilation
    pub fn threads(&self) -> usize {
        self.opts.debugging_opts.threads
    }

    /// Returns the number of codegen units that should be used for this
    /// compilation
    pub fn codegen_units(&self) -> usize {
        if let Some(n) = self.opts.cli_forced_codegen_units {
            return n;
        }
        if let Some(n) = self.target.target.options.default_codegen_units {
            return n as usize;
        }

        // Why is 16 codegen units the default all the time?
        //
        // The main reason for enabling multiple codegen units by default is to
        // leverage the ability for the codegen backend to do codegen and
        // optimization in parallel. This allows us, especially for large crates, to
        // make good use of all available resources on the machine once we've
        // hit that stage of compilation. Large crates especially then often
        // take a long time in codegen/optimization and this helps us amortize that
        // cost.
        //
        // Note that a high number here doesn't mean that we'll be spawning a
        // large number of threads in parallel. The backend of rustc contains
        // global rate limiting through the `jobserver` crate so we'll never
        // overload the system with too much work, but rather we'll only be
        // optimizing when we're otherwise cooperating with other instances of
        // rustc.
        //
        // Rather a high number here means that we should be able to keep a lot
        // of idle cpus busy. By ensuring that no codegen unit takes *too* long
        // to build we'll be guaranteed that all cpus will finish pretty closely
        // to one another and we should make relatively optimal use of system
        // resources
        //
        // Note that the main cost of codegen units is that it prevents LLVM
        // from inlining across codegen units. Users in general don't have a lot
        // of control over how codegen units are split up so it's our job in the
        // compiler to ensure that undue performance isn't lost when using
        // codegen units (aka we can't require everyone to slap `#[inline]` on
        // everything).
        //
        // If we're compiling at `-O0` then the number doesn't really matter too
        // much because performance doesn't matter and inlining is ok to lose.
        // In debug mode we just want to try to guarantee that no cpu is stuck
        // doing work that could otherwise be farmed to others.
        //
        // In release mode, however (O1 and above) performance does indeed
        // matter! To recover the loss in performance due to inlining we'll be
        // enabling ThinLTO by default (the function for which is just below).
        // This will ensure that we recover any inlining wins we otherwise lost
        // through codegen unit partitioning.
        //
        // ---
        //
        // Ok that's a lot of words but the basic tl;dr; is that we want a high
        // number here -- but not too high. Additionally we're "safe" to have it
        // always at the same number at all optimization levels.
        //
        // As a result 16 was chosen here! Mostly because it was a power of 2
        // and most benchmarks agreed it was roughly a local optimum. Not very
        // scientific.
        16
    }

    pub fn teach(&self, code: &DiagnosticId) -> bool {
        self.opts.debugging_opts.teach && self.diagnostic().must_teach(code)
    }

    pub fn rust_2015(&self) -> bool {
        self.opts.edition == Edition::Edition2015
    }

    /// Are we allowed to use features from the Rust 2018 edition?
    pub fn rust_2018(&self) -> bool {
        self.opts.edition >= Edition::Edition2018
    }

    pub fn edition(&self) -> Edition {
        self.opts.edition
    }

    /// Returns `true` if we cannot skip the PLT for shared library calls.
    pub fn needs_plt(&self) -> bool {
        // Check if the current target usually needs PLT to be enabled.
        // The user can use the command line flag to override it.
        let needs_plt = self.target.target.options.needs_plt;

        let dbg_opts = &self.opts.debugging_opts;

        let relro_level = dbg_opts.relro_level.unwrap_or(self.target.target.options.relro_level);

        // Only enable this optimization by default if full relro is also enabled.
        // In this case, lazy binding was already unavailable, so nothing is lost.
        // This also ensures `-Wl,-z,now` is supported by the linker.
        let full_relro = RelroLevel::Full == relro_level;

        // If user didn't explicitly forced us to use / skip the PLT,
        // then try to skip it where possible.
        dbg_opts.plt.unwrap_or(needs_plt || !full_relro)
    }
}

pub fn build_session(
    sopts: config::Options,
    local_crate_source_file: Option<PathBuf>,
    registry: rustc_errors::registry::Registry,
) -> Session {
    let file_path_mapping = sopts.file_path_mapping();

    build_session_with_source_map(
        sopts,
        local_crate_source_file,
        registry,
        Lrc::new(source_map::SourceMap::new(file_path_mapping)),
        DiagnosticOutput::Default,
        Default::default(),
    )
}

fn default_emitter(
    sopts: &config::Options,
    registry: rustc_errors::registry::Registry,
    source_map: &Lrc<source_map::SourceMap>,
    emitter_dest: Option<Box<dyn Write + Send>>,
) -> Box<dyn Emitter + sync::Send> {
    let macro_backtrace = sopts.debugging_opts.macro_backtrace;
    match (sopts.error_format, emitter_dest) {
        (config::ErrorOutputType::HumanReadable(kind), dst) => {
            let (short, color_config) = kind.unzip();

            if let HumanReadableErrorType::AnnotateSnippet(_) = kind {
                let emitter = AnnotateSnippetEmitterWriter::new(
                    Some(source_map.clone()),
                    short,
                    macro_backtrace,
                );
                Box::new(emitter.ui_testing(sopts.debugging_opts.ui_testing()))
            } else {
                let emitter = match dst {
                    None => EmitterWriter::stderr(
                        color_config,
                        Some(source_map.clone()),
                        short,
                        sopts.debugging_opts.teach,
                        sopts.debugging_opts.terminal_width,
                        macro_backtrace,
                    ),
                    Some(dst) => EmitterWriter::new(
                        dst,
                        Some(source_map.clone()),
                        short,
                        false, // no teach messages when writing to a buffer
                        false, // no colors when writing to a buffer
                        None,  // no terminal width
                        macro_backtrace,
                    ),
                };
                Box::new(emitter.ui_testing(sopts.debugging_opts.ui_testing()))
            }
        }
        (config::ErrorOutputType::Json { pretty, json_rendered }, None) => Box::new(
            JsonEmitter::stderr(
                Some(registry),
                source_map.clone(),
                pretty,
                json_rendered,
                macro_backtrace,
            )
            .ui_testing(sopts.debugging_opts.ui_testing()),
        ),
        (config::ErrorOutputType::Json { pretty, json_rendered }, Some(dst)) => Box::new(
            JsonEmitter::new(
                dst,
                Some(registry),
                source_map.clone(),
                pretty,
                json_rendered,
                macro_backtrace,
            )
            .ui_testing(sopts.debugging_opts.ui_testing()),
        ),
    }
}

pub enum DiagnosticOutput {
    Default,
    Raw(Box<dyn Write + Send>),
}

pub fn build_session_with_source_map(
    sopts: config::Options,
    local_crate_source_file: Option<PathBuf>,
    registry: rustc_errors::registry::Registry,
    source_map: Lrc<source_map::SourceMap>,
    diagnostics_output: DiagnosticOutput,
    lint_caps: FxHashMap<lint::LintId, lint::Level>,
) -> Session {
    // FIXME: This is not general enough to make the warning lint completely override
    // normal diagnostic warnings, since the warning lint can also be denied and changed
    // later via the source code.
    let warnings_allow = sopts
        .lint_opts
        .iter()
        .filter(|&&(ref key, _)| *key == "warnings")
        .map(|&(_, ref level)| *level == lint::Allow)
        .last()
        .unwrap_or(false);
    let cap_lints_allow = sopts.lint_cap.map_or(false, |cap| cap == lint::Allow);
    let can_emit_warnings = !(warnings_allow || cap_lints_allow);

    let write_dest = match diagnostics_output {
        DiagnosticOutput::Default => None,
        DiagnosticOutput::Raw(write) => Some(write),
    };
    let emitter = default_emitter(&sopts, registry, &source_map, write_dest);

    let diagnostic_handler = rustc_errors::Handler::with_emitter_and_flags(
        emitter,
        sopts.debugging_opts.diagnostic_handler_flags(can_emit_warnings),
    );

    build_session_(sopts, local_crate_source_file, diagnostic_handler, source_map, lint_caps)
}

fn build_session_(
    sopts: config::Options,
    local_crate_source_file: Option<PathBuf>,
    span_diagnostic: rustc_errors::Handler,
    source_map: Lrc<source_map::SourceMap>,
    driver_lint_caps: FxHashMap<lint::LintId, lint::Level>,
) -> Session {
    let self_profiler = if let SwitchWithOptPath::Enabled(ref d) = sopts.debugging_opts.self_profile
    {
        let directory =
            if let Some(ref directory) = d { directory } else { std::path::Path::new(".") };

        let profiler = SelfProfiler::new(
            directory,
            sopts.crate_name.as_ref().map(|s| &s[..]),
            &sopts.debugging_opts.self_profile_events,
        );
        match profiler {
            Ok(profiler) => Some(Arc::new(profiler)),
            Err(e) => {
                early_warn(sopts.error_format, &format!("failed to create profiler: {}", e));
                None
            }
        }
    } else {
        None
    };

    let host_triple = TargetTriple::from_triple(config::host_triple());
    let host = Target::search(&host_triple).unwrap_or_else(|e| {
        span_diagnostic.fatal(&format!("Error loading host specification: {}", e)).raise()
    });
    let target_cfg = config::build_target_config(&sopts, &span_diagnostic);

    let parse_sess = ParseSess::with_span_handler(span_diagnostic, source_map);
    let sysroot = match &sopts.maybe_sysroot {
        Some(sysroot) => sysroot.clone(),
        None => filesearch::get_or_default_sysroot(),
    };

    let host_triple = config::host_triple();
    let target_triple = sopts.target_triple.triple();
    let host_tlib_path = SearchPath::from_sysroot_and_triple(&sysroot, host_triple);
    let target_tlib_path = if host_triple == target_triple {
        None
    } else {
        Some(SearchPath::from_sysroot_and_triple(&sysroot, target_triple))
    };

    let file_path_mapping = sopts.file_path_mapping();

    let local_crate_source_file =
        local_crate_source_file.map(|path| file_path_mapping.map_prefix(path).0);

    let optimization_fuel_crate = sopts.debugging_opts.fuel.as_ref().map(|i| i.0.clone());
    let optimization_fuel = Lock::new(OptimizationFuel {
        remaining: sopts.debugging_opts.fuel.as_ref().map(|i| i.1).unwrap_or(0),
        out_of_fuel: false,
    });
    let print_fuel_crate = sopts.debugging_opts.print_fuel.clone();
    let print_fuel = AtomicU64::new(0);

    let working_dir = env::current_dir().unwrap_or_else(|e| {
        parse_sess.span_diagnostic.fatal(&format!("Current directory is invalid: {}", e)).raise()
    });
    let working_dir = file_path_mapping.map_prefix(working_dir);

    let cgu_reuse_tracker = if sopts.debugging_opts.query_dep_graph {
        CguReuseTracker::new()
    } else {
        CguReuseTracker::new_disabled()
    };

    let prof = SelfProfilerRef::new(
        self_profiler,
        sopts.debugging_opts.time_passes || sopts.debugging_opts.time,
        sopts.debugging_opts.time_passes,
    );

    let sess = Session {
        target: target_cfg,
        host,
        opts: sopts,
        host_tlib_path,
        target_tlib_path,
        parse_sess,
        sysroot,
        local_crate_source_file,
        working_dir,
        one_time_diagnostics: Default::default(),
        crate_types: Once::new(),
        crate_disambiguator: Once::new(),
        features: Once::new(),
        recursion_limit: Once::new(),
        type_length_limit: Once::new(),
        imported_macro_spans: OneThread::new(RefCell::new(FxHashMap::default())),
        incr_comp_session: OneThread::new(RefCell::new(IncrCompSession::NotInitialized)),
        cgu_reuse_tracker,
        prof,
        perf_stats: PerfStats {
            symbol_hash_time: Lock::new(Duration::from_secs(0)),
            decode_def_path_tables_time: Lock::new(Duration::from_secs(0)),
            queries_canonicalized: AtomicUsize::new(0),
            normalize_ty_after_erasing_regions: AtomicUsize::new(0),
            normalize_projection_ty: AtomicUsize::new(0),
        },
        code_stats: Default::default(),
        optimization_fuel_crate,
        optimization_fuel,
        print_fuel_crate,
        print_fuel,
        jobserver: jobserver::client(),
        driver_lint_caps,
        trait_methods_not_found: Lock::new(Default::default()),
        confused_type_with_std_module: Lock::new(Default::default()),
        system_library_path: OneThread::new(RefCell::new(Default::default())),
    };

    validate_commandline_args_with_session_available(&sess);

    sess
}

// If it is useful to have a Session available already for validating a
// commandline argument, you can do so here.
fn validate_commandline_args_with_session_available(sess: &Session) {
    // Since we don't know if code in an rlib will be linked to statically or
    // dynamically downstream, rustc generates `__imp_` symbols that help the
    // MSVC linker deal with this lack of knowledge (#27438). Unfortunately,
    // these manually generated symbols confuse LLD when it tries to merge
    // bitcode during ThinLTO. Therefore we disallow dynamic linking on MSVC
    // when compiling for LLD ThinLTO. This way we can validly just not generate
    // the `dllimport` attributes and `__imp_` symbols in that case.
    if sess.opts.cg.linker_plugin_lto.enabled()
        && sess.opts.cg.prefer_dynamic
        && sess.target.target.options.is_like_msvc
    {
        sess.err(
            "Linker plugin based LTO is not supported together with \
                  `-C prefer-dynamic` when targeting MSVC",
        );
    }

    // Make sure that any given profiling data actually exists so LLVM can't
    // decide to silently skip PGO.
    if let Some(ref path) = sess.opts.cg.profile_use {
        if !path.exists() {
            sess.err(&format!(
                "File `{}` passed to `-C profile-use` does not exist.",
                path.display()
            ));
        }
    }

    // PGO does not work reliably with panic=unwind on Windows. Let's make it
    // an error to combine the two for now. It always runs into an assertions
    // if LLVM is built with assertions, but without assertions it sometimes
    // does not crash and will probably generate a corrupted binary.
    // We should only display this error if we're actually going to run PGO.
    // If we're just supposed to print out some data, don't show the error (#61002).
    if sess.opts.cg.profile_generate.enabled()
        && sess.target.target.options.is_like_msvc
        && sess.panic_strategy() == PanicStrategy::Unwind
        && sess.opts.prints.iter().all(|&p| p == PrintRequest::NativeStaticLibs)
    {
        sess.err(
            "Profile-guided optimization does not yet work in conjunction \
                  with `-Cpanic=unwind` on Windows when targeting MSVC. \
                  See issue #61002 <https://github.com/rust-lang/rust/issues/61002> \
                  for more information.",
        );
    }

    // Sanitizers can only be used on some tested platforms.
    if let Some(ref sanitizer) = sess.opts.debugging_opts.sanitizer {
        const ASAN_SUPPORTED_TARGETS: &[&str] = &[
            "x86_64-unknown-linux-gnu",
            "x86_64-apple-darwin",
            "x86_64-fuchsia",
            "aarch64-fuchsia",
        ];
        const TSAN_SUPPORTED_TARGETS: &[&str] =
            &["x86_64-unknown-linux-gnu", "x86_64-apple-darwin"];
        const LSAN_SUPPORTED_TARGETS: &[&str] =
            &["x86_64-unknown-linux-gnu", "x86_64-apple-darwin"];
        const MSAN_SUPPORTED_TARGETS: &[&str] = &["x86_64-unknown-linux-gnu"];

        let supported_targets = match *sanitizer {
            Sanitizer::Address => ASAN_SUPPORTED_TARGETS,
            Sanitizer::Thread => TSAN_SUPPORTED_TARGETS,
            Sanitizer::Leak => LSAN_SUPPORTED_TARGETS,
            Sanitizer::Memory => MSAN_SUPPORTED_TARGETS,
        };

        if !supported_targets.contains(&&*sess.opts.target_triple.triple()) {
            sess.err(&format!(
                "{:?}Sanitizer only works with the `{}` target",
                sanitizer,
                supported_targets.join("` or `")
            ));
        }
    }
}

/// Hash value constructed out of all the `-C metadata` arguments passed to the
/// compiler. Together with the crate-name forms a unique global identifier for
/// the crate.
#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Clone, Copy, RustcEncodable, RustcDecodable)]
pub struct CrateDisambiguator(Fingerprint);

impl CrateDisambiguator {
    pub fn to_fingerprint(self) -> Fingerprint {
        self.0
    }
}

impl fmt::Display for CrateDisambiguator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let (a, b) = self.0.as_value();
        let as_u128 = a as u128 | ((b as u128) << 64);
        f.write_str(&base_n::encode(as_u128, base_n::CASE_INSENSITIVE))
    }
}

impl From<Fingerprint> for CrateDisambiguator {
    fn from(fingerprint: Fingerprint) -> CrateDisambiguator {
        CrateDisambiguator(fingerprint)
    }
}

impl_stable_hash_via_hash!(CrateDisambiguator);

/// Holds data on the current incremental compilation session, if there is one.
#[derive(Debug)]
pub enum IncrCompSession {
    /// This is the state the session will be in until the incr. comp. dir is
    /// needed.
    NotInitialized,
    /// This is the state during which the session directory is private and can
    /// be modified.
    Active { session_directory: PathBuf, lock_file: flock::Lock, load_dep_graph: bool },
    /// This is the state after the session directory has been finalized. In this
    /// state, the contents of the directory must not be modified any more.
    Finalized { session_directory: PathBuf },
    /// This is an error state that is reached when some compilation error has
    /// occurred. It indicates that the contents of the session directory must
    /// not be used, since they might be invalid.
    InvalidBecauseOfErrors { session_directory: PathBuf },
}

pub fn early_error(output: config::ErrorOutputType, msg: &str) -> ! {
    let emitter: Box<dyn Emitter + sync::Send> = match output {
        config::ErrorOutputType::HumanReadable(kind) => {
            let (short, color_config) = kind.unzip();
            Box::new(EmitterWriter::stderr(color_config, None, short, false, None, false))
        }
        config::ErrorOutputType::Json { pretty, json_rendered } => {
            Box::new(JsonEmitter::basic(pretty, json_rendered, false))
        }
    };
    let handler = rustc_errors::Handler::with_emitter(true, None, emitter);
    handler.struct_fatal(msg).emit();
    rustc_errors::FatalError.raise();
}

pub fn early_warn(output: config::ErrorOutputType, msg: &str) {
    let emitter: Box<dyn Emitter + sync::Send> = match output {
        config::ErrorOutputType::HumanReadable(kind) => {
            let (short, color_config) = kind.unzip();
            Box::new(EmitterWriter::stderr(color_config, None, short, false, None, false))
        }
        config::ErrorOutputType::Json { pretty, json_rendered } => {
            Box::new(JsonEmitter::basic(pretty, json_rendered, false))
        }
    };
    let handler = rustc_errors::Handler::with_emitter(true, None, emitter);
    handler.struct_warn(msg).emit();
}

pub type CompileResult = Result<(), ErrorReported>;
