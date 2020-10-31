use crate::cgu_reuse_tracker::CguReuseTracker;
use crate::code_stats::CodeStats;
pub use crate::code_stats::{DataTypeKind, FieldInfo, SizeKind, VariantInfo};
use crate::config::{self, CrateType, OutputType, PrintRequest, SanitizerSet, SwitchWithOptPath};
use crate::filesearch;
use crate::lint::{self, LintId};
use crate::parse::ParseSess;
use crate::search_paths::{PathKind, SearchPath};

pub use rustc_ast::attr::MarkedAttrs;
pub use rustc_ast::crate_disambiguator::CrateDisambiguator;
pub use rustc_ast::Attribute;
use rustc_data_structures::flock;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::jobserver::{self, Client};
use rustc_data_structures::profiling::{duration_to_secs_str, SelfProfiler, SelfProfilerRef};
use rustc_data_structures::sync::{
    self, AtomicU64, AtomicUsize, Lock, Lrc, OnceCell, OneThread, Ordering, Ordering::SeqCst,
};
use rustc_errors::annotate_snippet_emitter_writer::AnnotateSnippetEmitterWriter;
use rustc_errors::emitter::{Emitter, EmitterWriter, HumanReadableErrorType};
use rustc_errors::json::JsonEmitter;
use rustc_errors::registry::Registry;
use rustc_errors::{Applicability, Diagnostic, DiagnosticBuilder, DiagnosticId, ErrorReported};
use rustc_lint_defs::FutureBreakage;
use rustc_span::edition::Edition;
use rustc_span::source_map::{FileLoader, MultiSpan, RealFileLoader, SourceMap, Span};
use rustc_span::{sym, SourceFileHashAlgorithm, Symbol};
use rustc_target::asm::InlineAsmArch;
use rustc_target::spec::{CodeModel, PanicStrategy, RelocModel, RelroLevel};
use rustc_target::spec::{Target, TargetTriple, TlsModel};

use std::cell::{self, RefCell};
use std::env;
use std::fmt;
use std::io::Write;
use std::num::NonZeroU32;
use std::ops::{Div, Mul};
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

pub trait SessionLintStore: sync::Send + sync::Sync {
    fn name_to_lint(&self, lint_name: &str) -> LintId;
}

pub struct OptimizationFuel {
    /// If `-zfuel=crate=n` is specified, initially set to `n`, otherwise `0`.
    remaining: u64,
    /// We're rejecting all further optimizations.
    out_of_fuel: bool,
}

/// The behavior of the CTFE engine when an error occurs with regards to backtraces.
#[derive(Clone, Copy)]
pub enum CtfeBacktrace {
    /// Do nothing special, return the error as usual without a backtrace.
    Disabled,
    /// Capture a backtrace at the point the error is created and return it in the error
    /// (to be printed later if/when the error ever actually gets shown to the user).
    Capture,
    /// Capture a backtrace at the point the error is created and immediately print it out.
    Immediate,
}

/// New-type wrapper around `usize` for representing limits. Ensures that comparisons against
/// limits are consistent throughout the compiler.
#[derive(Clone, Copy, Debug)]
pub struct Limit(pub usize);

impl Limit {
    /// Create a new limit from a `usize`.
    pub fn new(value: usize) -> Self {
        Limit(value)
    }

    /// Check that `value` is within the limit. Ensures that the same comparisons are used
    /// throughout the compiler, as mismatches can cause ICEs, see #72540.
    pub fn value_within_limit(&self, value: usize) -> bool {
        value <= self.0
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Div<usize> for Limit {
    type Output = Limit;

    fn div(self, rhs: usize) -> Self::Output {
        Limit::new(self.0 / rhs)
    }
}

impl Mul<usize> for Limit {
    type Output = Limit;

    fn mul(self, rhs: usize) -> Self::Output {
        Limit::new(self.0 * rhs)
    }
}

/// Represents the data associated with a compilation
/// session for a single crate.
pub struct Session {
    pub target: Target,
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
    crate_types: OnceCell<Vec<CrateType>>,
    /// The `crate_disambiguator` is constructed out of all the `-C metadata`
    /// arguments passed to the compiler. Its value together with the crate-name
    /// forms a unique global identifier for the crate. It is used to allow
    /// multiple crates with the same name to coexist. See the
    /// `rustc_codegen_llvm::back::symbol_names` module for more information.
    pub crate_disambiguator: OnceCell<CrateDisambiguator>,

    features: OnceCell<rustc_feature::Features>,

    lint_store: OnceCell<Lrc<dyn SessionLintStore>>,

    /// The maximum recursion limit for potentially infinitely recursive
    /// operations such as auto-dereference and monomorphization.
    pub recursion_limit: OnceCell<Limit>,

    /// The maximum length of types during monomorphization.
    pub type_length_limit: OnceCell<Limit>,

    /// The maximum blocks a const expression can evaluate.
    pub const_eval_limit: OnceCell<Limit>,

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

    /// Tracks the current behavior of the CTFE engine when an error occurs.
    /// Options range from returning the error without a backtrace to returning an error
    /// and immediately printing the backtrace to stderr.
    pub ctfe_backtrace: Lock<CtfeBacktrace>,

    /// This tracks where `-Zunleash-the-miri-inside-of-you` was used to get around a
    /// const check, optionally with the relevant feature gate.  We use this to
    /// warn about unleashing, but with a single diagnostic instead of dozens that
    /// drown everything else in noise.
    miri_unleashed_features: Lock<Vec<(Span, Option<Symbol>)>>,

    /// Base directory containing the `src/` for the Rust standard library, and
    /// potentially `rustc` as well, if we can can find it. Right now it's always
    /// `$sysroot/lib/rustlib/src/rust` (i.e. the `rustup` `rust-src` component).
    ///
    /// This directory is what the virtual `/rustc/$hash` is translated back to,
    /// if Rust was built with path remapping to `/rustc/$hash` enabled
    /// (the `rust.remap-debuginfo` option in `config.toml`).
    pub real_rust_source_base_dir: Option<PathBuf>,

    /// Architecture to use for interpreting asm!.
    pub asm_arch: Option<InlineAsmArch>,

    /// Set of enabled features for the current target.
    pub target_features: FxHashSet<Symbol>,

    known_attrs: Lock<MarkedAttrs>,
    used_attrs: Lock<MarkedAttrs>,

    /// `Span`s for `if` conditions that we have suggested turning into `if let`.
    pub if_let_suggestions: Lock<FxHashSet<Span>>,
}

pub struct PerfStats {
    /// The accumulated time spent on computing symbol hashes.
    pub symbol_hash_time: Lock<Duration>,
    /// Total number of values canonicalized queries constructed.
    pub queries_canonicalized: AtomicUsize,
    /// Number of times this query is invoked.
    pub normalize_generic_arg_after_erasing_regions: AtomicUsize,
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

/// Trait implemented by error types. This should not be implemented manually. Instead, use
/// `#[derive(SessionDiagnostic)]` -- see [rustc_macros::SessionDiagnostic].
pub trait SessionDiagnostic<'a> {
    /// Write out as a diagnostic out of `sess`.
    #[must_use]
    fn into_diagnostic(self, sess: &'a Session) -> DiagnosticBuilder<'a>;
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
    pub fn miri_unleashed_feature(&self, span: Span, feature_gate: Option<Symbol>) {
        self.miri_unleashed_features.lock().push((span, feature_gate));
    }

    fn check_miri_unleashed_features(&self) {
        let unleashed_features = self.miri_unleashed_features.lock();
        if !unleashed_features.is_empty() {
            let mut must_err = false;
            // Create a diagnostic pointing at where things got unleashed.
            let mut diag = self.struct_warn("skipping const checks");
            for &(span, feature_gate) in unleashed_features.iter() {
                // FIXME: `span_label` doesn't do anything, so we use "help" as a hack.
                if let Some(feature_gate) = feature_gate {
                    diag.span_help(span, &format!("skipping check for `{}` feature", feature_gate));
                    // The unleash flag must *not* be used to just "hack around" feature gates.
                    must_err = true;
                } else {
                    diag.span_help(span, "skipping check that does not even have a feature gate");
                }
            }
            diag.emit();
            // If we should err, make sure we did.
            if must_err && !self.has_errors() {
                // We have skipped a feature gate, and not run into other errors... reject.
                self.err(
                    "`-Zunleash-the-miri-inside-of-you` may not be used to circumvent feature \
                     gates, except when testing error paths in the CTFE engine",
                );
            }
        }
    }

    /// Invoked all the way at the end to finish off diagnostics printing.
    pub fn finish_diagnostics(&self, registry: &Registry) {
        self.check_miri_unleashed_features();
        self.diagnostic().print_error_count(registry);
        self.emit_future_breakage();
    }

    fn emit_future_breakage(&self) {
        if !self.opts.debugging_opts.emit_future_incompat_report {
            return;
        }

        let diags = self.diagnostic().take_future_breakage_diagnostics();
        if diags.is_empty() {
            return;
        }
        // If any future-breakage lints were registered, this lint store
        // should be available
        let lint_store = self.lint_store.get().expect("`lint_store` not initialized!");
        let diags_and_breakage: Vec<(FutureBreakage, Diagnostic)> = diags
            .into_iter()
            .map(|diag| {
                let lint_name = match &diag.code {
                    Some(DiagnosticId::Lint { name, has_future_breakage: true }) => name,
                    _ => panic!("Unexpected code in diagnostic {:?}", diag),
                };
                let lint = lint_store.name_to_lint(&lint_name);
                let future_breakage =
                    lint.lint.future_incompatible.unwrap().future_breakage.unwrap();
                (future_breakage, diag)
            })
            .collect();
        self.parse_sess.span_diagnostic.emit_future_breakage_report(diags_and_breakage);
    }

    pub fn local_crate_disambiguator(&self) -> CrateDisambiguator {
        self.crate_disambiguator.get().copied().unwrap()
    }

    pub fn crate_types(&self) -> &[CrateType] {
        self.crate_types.get().unwrap().as_slice()
    }

    pub fn init_crate_types(&self, crate_types: Vec<CrateType>) {
        self.crate_types.set(crate_types).expect("`crate_types` was initialized twice")
    }

    pub fn recursion_limit(&self) -> Limit {
        self.recursion_limit.get().copied().unwrap()
    }

    pub fn type_length_limit(&self) -> Limit {
        self.type_length_limit.get().copied().unwrap()
    }

    pub fn const_eval_limit(&self) -> Limit {
        self.const_eval_limit.get().copied().unwrap()
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
    pub fn struct_span_allow<S: Into<MultiSpan>>(&self, sp: S, msg: &str) -> DiagnosticBuilder<'_> {
        self.diagnostic().struct_span_allow(sp, msg)
    }
    pub fn struct_allow(&self, msg: &str) -> DiagnosticBuilder<'_> {
        self.diagnostic().struct_allow(msg)
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
    pub fn emit_err<'a>(&'a self, err: impl SessionDiagnostic<'a>) {
        err.into_diagnostic(self).emit()
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
    #[track_caller]
    pub fn delay_span_bug<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.diagnostic().delay_span_bug(sp, msg)
    }

    /// Used for code paths of expensive computations that should only take place when
    /// warnings or errors are emitted. If no messages are emitted ("good path"), then
    /// it's likely a bug.
    pub fn delay_good_path_bug(&self, msg: &str) {
        if self.opts.debugging_opts.print_type_sizes
            || self.opts.debugging_opts.query_dep_graph
            || self.opts.debugging_opts.dump_mir.is_some()
            || self.opts.debugging_opts.unpretty.is_some()
            || self.opts.output_types.contains_key(&OutputType::Mir)
            || std::env::var_os("RUSTC_LOG").is_some()
        {
            return;
        }

        self.diagnostic().delay_good_path_bug(msg)
    }

    pub fn note_without_error(&self, msg: &str) {
        self.diagnostic().note_without_error(msg)
    }
    pub fn span_note_without_error<S: Into<MultiSpan>>(&self, sp: S, msg: &str) {
        self.diagnostic().span_note_without_error(sp, msg)
    }
    pub fn struct_note_without_error(&self, msg: &str) -> DiagnosticBuilder<'_> {
        self.diagnostic().struct_note_without_error(msg)
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

    #[inline]
    pub fn source_map(&self) -> &SourceMap {
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
        self.opts.debugging_opts.verify_llvm_ir || option_env!("RUSTC_VERIFY_LLVM_IR").is_some()
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
        self.features.get().unwrap()
    }

    pub fn init_features(&self, features: rustc_feature::Features) {
        match self.features.set(features) {
            Ok(()) => {}
            Err(_) => panic!("`features` was initialized twice"),
        }
    }

    pub fn init_lint_store(&self, lint_store: Lrc<dyn SessionLintStore>) {
        self.lint_store
            .set(lint_store)
            .map_err(|_| ())
            .expect("`lint_store` was initialized twice");
    }

    /// Calculates the flavor of LTO to use for this compilation.
    pub fn lto(&self) -> config::Lto {
        // If our target has codegen requirements ignore the command line
        if self.target.options.requires_lto {
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
        self.opts.cg.panic.unwrap_or(self.target.options.panic_strategy)
    }
    pub fn fewer_names(&self) -> bool {
        let more_names = self.opts.output_types.contains_key(&OutputType::LlvmAssembly)
            || self.opts.output_types.contains_key(&OutputType::Bitcode)
            // AddressSanitizer and MemorySanitizer use alloca name when reporting an issue.
            || self.opts.debugging_opts.sanitizer.intersects(SanitizerSet::ADDRESS | SanitizerSet::MEMORY);

        self.opts.debugging_opts.fewer_names || !more_names
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

    /// Check whether this compile session and crate type use static crt.
    pub fn crt_static(&self, crate_type: Option<CrateType>) -> bool {
        if !self.target.options.crt_static_respected {
            // If the target does not opt in to crt-static support, use its default.
            return self.target.options.crt_static_default;
        }

        let requested_features = self.opts.cg.target_feature.split(',');
        let found_negative = requested_features.clone().any(|r| r == "-crt-static");
        let found_positive = requested_features.clone().any(|r| r == "+crt-static");

        if found_positive || found_negative {
            found_positive
        } else if crate_type == Some(CrateType::ProcMacro)
            || crate_type == None && self.opts.crate_types.contains(&CrateType::ProcMacro)
        {
            // FIXME: When crate_type is not available,
            // we use compiler options to determine the crate_type.
            // We can't check `#![crate_type = "proc-macro"]` here.
            false
        } else {
            self.target.options.crt_static_default
        }
    }

    pub fn relocation_model(&self) -> RelocModel {
        self.opts.cg.relocation_model.unwrap_or(self.target.options.relocation_model)
    }

    pub fn code_model(&self) -> Option<CodeModel> {
        self.opts.cg.code_model.or(self.target.options.code_model)
    }

    pub fn tls_model(&self) -> TlsModel {
        self.opts.debugging_opts.tls_model.unwrap_or(self.target.options.tls_model)
    }

    pub fn must_not_eliminate_frame_pointers(&self) -> bool {
        // "mcount" function relies on stack pointer.
        // See <https://sourceware.org/binutils/docs/gprof/Implementation.html>.
        if self.instrument_mcount() {
            true
        } else if let Some(x) = self.opts.cg.force_frame_pointers {
            x
        } else {
            !self.target.options.eliminate_frame_pointer
        }
    }

    pub fn must_emit_unwind_tables(&self) -> bool {
        // This is used to control the emission of the `uwtable` attribute on
        // LLVM functions.
        //
        // At the very least, unwind tables are needed when compiling with
        // `-C panic=unwind`.
        //
        // On some targets (including windows), however, exceptions include
        // other events such as illegal instructions, segfaults, etc. This means
        // that on Windows we end up still needing unwind tables even if the `-C
        // panic=abort` flag is passed.
        //
        // You can also find more info on why Windows needs unwind tables in:
        //      https://bugzilla.mozilla.org/show_bug.cgi?id=1302078
        //
        // If a target requires unwind tables, then they must be emitted.
        // Otherwise, we can defer to the `-C force-unwind-tables=<yes/no>`
        // value, if it is provided, or disable them, if not.
        if self.panic_strategy() == PanicStrategy::Unwind {
            true
        } else if self.target.options.requires_uwtable {
            true
        } else {
            self.opts.cg.force_unwind_tables.unwrap_or(false)
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
            "Total queries canonicalized:                   {}",
            self.perf_stats.queries_canonicalized.load(Ordering::Relaxed)
        );
        println!(
            "normalize_generic_arg_after_erasing_regions:   {}",
            self.perf_stats.normalize_generic_arg_after_erasing_regions.load(Ordering::Relaxed)
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
                    self.warn(&format!("optimization-fuel-exhausted: {}", msg()));
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
        if let Some(n) = self.target.options.default_codegen_units {
            return n as usize;
        }

        // If incremental compilation is turned on, we default to a high number
        // codegen units in order to reduce the "collateral damage" small
        // changes cause.
        if self.opts.incremental.is_some() {
            return 256;
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
        let needs_plt = self.target.options.needs_plt;

        let dbg_opts = &self.opts.debugging_opts;

        let relro_level = dbg_opts.relro_level.unwrap_or(self.target.options.relro_level);

        // Only enable this optimization by default if full relro is also enabled.
        // In this case, lazy binding was already unavailable, so nothing is lost.
        // This also ensures `-Wl,-z,now` is supported by the linker.
        let full_relro = RelroLevel::Full == relro_level;

        // If user didn't explicitly forced us to use / skip the PLT,
        // then try to skip it where possible.
        dbg_opts.plt.unwrap_or(needs_plt || !full_relro)
    }

    /// Checks if LLVM lifetime markers should be emitted.
    pub fn emit_lifetime_markers(&self) -> bool {
        self.opts.optimize != config::OptLevel::No
        // AddressSanitizer uses lifetimes to detect use after scope bugs.
        // MemorySanitizer uses lifetimes to detect use of uninitialized stack variables.
        || self.opts.debugging_opts.sanitizer.intersects(SanitizerSet::ADDRESS | SanitizerSet::MEMORY)
    }

    pub fn link_dead_code(&self) -> bool {
        match self.opts.cg.link_dead_code {
            Some(explicitly_set) => explicitly_set,
            None => {
                self.opts.debugging_opts.instrument_coverage && !self.target.options.is_like_msvc
                // Issue #76038: (rustc `-Clink-dead-code` causes MSVC linker to produce invalid
                // binaries when LLVM InstrProf counters are enabled). As described by this issue,
                // the "link dead code" option produces incorrect binaries when compiled and linked
                // under MSVC. The resulting Rust programs typically crash with a segmentation
                // fault, or produce an empty "*.profraw" file (profiling counter results normally
                // generated during program exit).
                //
                // If not targeting MSVC, `-Z instrument-coverage` implies `-C link-dead-code`, so
                // unexecuted code is still counted as zero, rather than be optimized out. Note that
                // instrumenting dead code can be explicitly disabled with:
                //
                //     `-Z instrument-coverage -C link-dead-code=no`.
                //
                // FIXME(richkadel): Investigate if `instrument-coverage` implementation can inject
                // [zero counters](https://llvm.org/docs/CoverageMappingFormat.html#counter) in the
                // coverage map when "dead code" is removed, rather than forcing `link-dead-code`.
                // This may not be possible, however, if (as it seems to appear) the "dead code"
                // that would otherwise not be linked is only identified as "dead" by the native
                // linker. If that's the case, I believe it is too late for the Rust compiler to
                // leverage any information it might be able to get from the linker regarding what
                // code is dead, to be able to add those counters.
                //
                // On the other hand, if any Rust compiler passes are optimizing out dead code blocks
                // we should inject "zero" counters for those code regions.
            }
        }
    }

    pub fn mark_attr_known(&self, attr: &Attribute) {
        self.known_attrs.lock().mark(attr)
    }

    pub fn is_attr_known(&self, attr: &Attribute) -> bool {
        self.known_attrs.lock().is_marked(attr)
    }

    pub fn mark_attr_used(&self, attr: &Attribute) {
        self.used_attrs.lock().mark(attr)
    }

    pub fn is_attr_used(&self, attr: &Attribute) -> bool {
        self.used_attrs.lock().is_marked(attr)
    }

    /// Returns `true` if the attribute's path matches the argument. If it
    /// matches, then the attribute is marked as used.
    ///
    /// This method should only be used by rustc, other tools can use
    /// `Attribute::has_name` instead, because only rustc is supposed to report
    /// the `unused_attributes` lint. (`MetaItem` and `NestedMetaItem` are
    /// produced by lowering an `Attribute` and don't have identity, so they
    /// only have the `has_name` method, and you need to mark the original
    /// `Attribute` as used when necessary.)
    pub fn check_name(&self, attr: &Attribute, name: Symbol) -> bool {
        let matches = attr.has_name(name);
        if matches {
            self.mark_attr_used(attr);
        }
        matches
    }

    pub fn is_proc_macro_attr(&self, attr: &Attribute) -> bool {
        [sym::proc_macro, sym::proc_macro_attribute, sym::proc_macro_derive]
            .iter()
            .any(|kind| self.check_name(attr, *kind))
    }

    pub fn contains_name(&self, attrs: &[Attribute], name: Symbol) -> bool {
        attrs.iter().any(|item| self.check_name(item, name))
    }

    pub fn find_by_name<'a>(
        &'a self,
        attrs: &'a [Attribute],
        name: Symbol,
    ) -> Option<&'a Attribute> {
        attrs.iter().find(|attr| self.check_name(attr, name))
    }

    pub fn filter_by_name<'a>(
        &'a self,
        attrs: &'a [Attribute],
        name: Symbol,
    ) -> impl Iterator<Item = &'a Attribute> {
        attrs.iter().filter(move |attr| self.check_name(attr, name))
    }

    pub fn first_attr_value_str_by_name(
        &self,
        attrs: &[Attribute],
        name: Symbol,
    ) -> Option<Symbol> {
        attrs.iter().find(|at| self.check_name(at, name)).and_then(|at| at.value_str())
    }
}

fn default_emitter(
    sopts: &config::Options,
    registry: rustc_errors::registry::Registry,
    source_map: Lrc<SourceMap>,
    emitter_dest: Option<Box<dyn Write + Send>>,
) -> Box<dyn Emitter + sync::Send> {
    let macro_backtrace = sopts.debugging_opts.macro_backtrace;
    match (sopts.error_format, emitter_dest) {
        (config::ErrorOutputType::HumanReadable(kind), dst) => {
            let (short, color_config) = kind.unzip();

            if let HumanReadableErrorType::AnnotateSnippet(_) = kind {
                let emitter =
                    AnnotateSnippetEmitterWriter::new(Some(source_map), short, macro_backtrace);
                Box::new(emitter.ui_testing(sopts.debugging_opts.ui_testing))
            } else {
                let emitter = match dst {
                    None => EmitterWriter::stderr(
                        color_config,
                        Some(source_map),
                        short,
                        sopts.debugging_opts.teach,
                        sopts.debugging_opts.terminal_width,
                        macro_backtrace,
                    ),
                    Some(dst) => EmitterWriter::new(
                        dst,
                        Some(source_map),
                        short,
                        false, // no teach messages when writing to a buffer
                        false, // no colors when writing to a buffer
                        None,  // no terminal width
                        macro_backtrace,
                    ),
                };
                Box::new(emitter.ui_testing(sopts.debugging_opts.ui_testing))
            }
        }
        (config::ErrorOutputType::Json { pretty, json_rendered }, None) => Box::new(
            JsonEmitter::stderr(
                Some(registry),
                source_map,
                pretty,
                json_rendered,
                sopts.debugging_opts.terminal_width,
                macro_backtrace,
            )
            .ui_testing(sopts.debugging_opts.ui_testing),
        ),
        (config::ErrorOutputType::Json { pretty, json_rendered }, Some(dst)) => Box::new(
            JsonEmitter::new(
                dst,
                Some(registry),
                source_map,
                pretty,
                json_rendered,
                sopts.debugging_opts.terminal_width,
                macro_backtrace,
            )
            .ui_testing(sopts.debugging_opts.ui_testing),
        ),
    }
}

pub enum DiagnosticOutput {
    Default,
    Raw(Box<dyn Write + Send>),
}

pub fn build_session(
    sopts: config::Options,
    local_crate_source_file: Option<PathBuf>,
    registry: rustc_errors::registry::Registry,
    diagnostics_output: DiagnosticOutput,
    driver_lint_caps: FxHashMap<lint::LintId, lint::Level>,
    file_loader: Option<Box<dyn FileLoader + Send + Sync + 'static>>,
    target_override: Option<Target>,
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

    let target_cfg = config::build_target_config(&sopts, target_override);
    let host_triple = TargetTriple::from_triple(config::host_triple());
    let host = Target::search(&host_triple).unwrap_or_else(|e| {
        early_error(sopts.error_format, &format!("Error loading host specification: {}", e))
    });

    let loader = file_loader.unwrap_or(Box::new(RealFileLoader));
    let hash_kind = sopts.debugging_opts.src_hash_algorithm.unwrap_or_else(|| {
        if target_cfg.options.is_like_msvc {
            SourceFileHashAlgorithm::Sha1
        } else {
            SourceFileHashAlgorithm::Md5
        }
    });
    let source_map = Lrc::new(SourceMap::with_file_loader_and_hash_kind(
        loader,
        sopts.file_path_mapping(),
        hash_kind,
    ));
    let emitter = default_emitter(&sopts, registry, source_map.clone(), write_dest);

    let span_diagnostic = rustc_errors::Handler::with_emitter_and_flags(
        emitter,
        sopts.debugging_opts.diagnostic_handler_flags(can_emit_warnings),
    );

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

    let ctfe_backtrace = Lock::new(match env::var("RUSTC_CTFE_BACKTRACE") {
        Ok(ref val) if val == "immediate" => CtfeBacktrace::Immediate,
        Ok(ref val) if val != "0" => CtfeBacktrace::Capture,
        _ => CtfeBacktrace::Disabled,
    });

    // Try to find a directory containing the Rust `src`, for more details see
    // the doc comment on the `real_rust_source_base_dir` field.
    let real_rust_source_base_dir = {
        // This is the location used by the `rust-src` `rustup` component.
        let mut candidate = sysroot.join("lib/rustlib/src/rust");
        if let Ok(metadata) = candidate.symlink_metadata() {
            // Replace the symlink rustbuild creates, with its destination.
            // We could try to use `fs::canonicalize` instead, but that might
            // produce unnecessarily verbose path.
            if metadata.file_type().is_symlink() {
                if let Ok(symlink_dest) = std::fs::read_link(&candidate) {
                    candidate = symlink_dest;
                }
            }
        }

        // Only use this directory if it has a file we can expect to always find.
        if candidate.join("library/std/src/lib.rs").is_file() { Some(candidate) } else { None }
    };

    let asm_arch = if target_cfg.options.allow_asm {
        InlineAsmArch::from_str(&target_cfg.arch).ok()
    } else {
        None
    };

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
        crate_types: OnceCell::new(),
        crate_disambiguator: OnceCell::new(),
        features: OnceCell::new(),
        lint_store: OnceCell::new(),
        recursion_limit: OnceCell::new(),
        type_length_limit: OnceCell::new(),
        const_eval_limit: OnceCell::new(),
        incr_comp_session: OneThread::new(RefCell::new(IncrCompSession::NotInitialized)),
        cgu_reuse_tracker,
        prof,
        perf_stats: PerfStats {
            symbol_hash_time: Lock::new(Duration::from_secs(0)),
            queries_canonicalized: AtomicUsize::new(0),
            normalize_generic_arg_after_erasing_regions: AtomicUsize::new(0),
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
        ctfe_backtrace,
        miri_unleashed_features: Lock::new(Default::default()),
        real_rust_source_base_dir,
        asm_arch,
        target_features: FxHashSet::default(),
        known_attrs: Lock::new(MarkedAttrs::new()),
        used_attrs: Lock::new(MarkedAttrs::new()),
        if_let_suggestions: Default::default(),
    };

    validate_commandline_args_with_session_available(&sess);

    sess
}

// If it is useful to have a Session available already for validating a
// commandline argument, you can do so here.
fn validate_commandline_args_with_session_available(sess: &Session) {
    // Since we don't know if code in an rlib will be linked to statically or
    // dynamically downstream, rustc generates `__imp_` symbols that help linkers
    // on Windows deal with this lack of knowledge (#27438). Unfortunately,
    // these manually generated symbols confuse LLD when it tries to merge
    // bitcode during ThinLTO. Therefore we disallow dynamic linking on Windows
    // when compiling for LLD ThinLTO. This way we can validly just not generate
    // the `dllimport` attributes and `__imp_` symbols in that case.
    if sess.opts.cg.linker_plugin_lto.enabled()
        && sess.opts.cg.prefer_dynamic
        && sess.target.options.is_like_windows
    {
        sess.err(
            "Linker plugin based LTO is not supported together with \
                  `-C prefer-dynamic` when targeting Windows-like targets",
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

    // Unwind tables cannot be disabled if the target requires them.
    if let Some(include_uwtables) = sess.opts.cg.force_unwind_tables {
        if sess.panic_strategy() == PanicStrategy::Unwind && !include_uwtables {
            sess.err(
                "panic=unwind requires unwind tables, they cannot be disabled \
                     with `-C force-unwind-tables=no`.",
            );
        }

        if sess.target.options.requires_uwtable && !include_uwtables {
            sess.err(
                "target requires unwind tables, they cannot be disabled with \
                     `-C force-unwind-tables=no`.",
            );
        }
    }

    // PGO does not work reliably with panic=unwind on Windows. Let's make it
    // an error to combine the two for now. It always runs into an assertions
    // if LLVM is built with assertions, but without assertions it sometimes
    // does not crash and will probably generate a corrupted binary.
    // We should only display this error if we're actually going to run PGO.
    // If we're just supposed to print out some data, don't show the error (#61002).
    if sess.opts.cg.profile_generate.enabled()
        && sess.target.options.is_like_msvc
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

    const ASAN_SUPPORTED_TARGETS: &[&str] = &[
        "aarch64-fuchsia",
        "aarch64-unknown-linux-gnu",
        "x86_64-apple-darwin",
        "x86_64-fuchsia",
        "x86_64-unknown-freebsd",
        "x86_64-unknown-linux-gnu",
    ];
    const LSAN_SUPPORTED_TARGETS: &[&str] =
        &["aarch64-unknown-linux-gnu", "x86_64-apple-darwin", "x86_64-unknown-linux-gnu"];
    const MSAN_SUPPORTED_TARGETS: &[&str] =
        &["aarch64-unknown-linux-gnu", "x86_64-unknown-freebsd", "x86_64-unknown-linux-gnu"];
    const TSAN_SUPPORTED_TARGETS: &[&str] = &[
        "aarch64-unknown-linux-gnu",
        "x86_64-apple-darwin",
        "x86_64-unknown-freebsd",
        "x86_64-unknown-linux-gnu",
    ];

    // Sanitizers can only be used on some tested platforms.
    for s in sess.opts.debugging_opts.sanitizer {
        let supported_targets = match s {
            SanitizerSet::ADDRESS => ASAN_SUPPORTED_TARGETS,
            SanitizerSet::LEAK => LSAN_SUPPORTED_TARGETS,
            SanitizerSet::MEMORY => MSAN_SUPPORTED_TARGETS,
            SanitizerSet::THREAD => TSAN_SUPPORTED_TARGETS,
            _ => panic!("unrecognized sanitizer {}", s),
        };
        if !supported_targets.contains(&&*sess.opts.target_triple.triple()) {
            sess.err(&format!(
                "`-Zsanitizer={}` only works with targets: {}",
                s,
                supported_targets.join(", ")
            ));
        }
        let conflicting = sess.opts.debugging_opts.sanitizer - s;
        if !conflicting.is_empty() {
            sess.err(&format!(
                "`-Zsanitizer={}` is incompatible with `-Zsanitizer={}`",
                s, conflicting,
            ));
            // Don't report additional errors.
            break;
        }
    }
}

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
            Box::new(JsonEmitter::basic(pretty, json_rendered, None, false))
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
            Box::new(JsonEmitter::basic(pretty, json_rendered, None, false))
        }
    };
    let handler = rustc_errors::Handler::with_emitter(true, None, emitter);
    handler.struct_warn(msg).emit();
}
