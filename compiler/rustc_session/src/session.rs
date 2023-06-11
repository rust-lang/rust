use crate::cgu_reuse_tracker::CguReuseTracker;
use crate::code_stats::CodeStats;
pub use crate::code_stats::{DataTypeKind, FieldInfo, FieldKind, SizeKind, VariantInfo};
use crate::config::Input;
use crate::config::{
    self, CrateType, InstrumentCoverage, OptLevel, OutFileName, OutputType, SwitchWithOptPath,
};
use crate::errors;
use crate::parse::{add_feature_diagnostics, ParseSess};
use crate::search_paths::{PathKind, SearchPath};
use crate::{filesearch, lint};

pub use rustc_ast::attr::MarkedAttrs;
pub use rustc_ast::Attribute;
use rustc_data_structures::flock;
use rustc_data_structures::fx::{FxHashMap, FxIndexSet};
use rustc_data_structures::jobserver::{self, Client};
use rustc_data_structures::profiling::{duration_to_secs_str, SelfProfiler, SelfProfilerRef};
use rustc_data_structures::sync::{
    self, AtomicU64, AtomicUsize, Lock, Lrc, OnceCell, OneThread, Ordering, Ordering::SeqCst,
};
use rustc_errors::annotate_snippet_emitter_writer::AnnotateSnippetEmitterWriter;
use rustc_errors::emitter::{Emitter, EmitterWriter, HumanReadableErrorType};
use rustc_errors::json::JsonEmitter;
use rustc_errors::registry::Registry;
use rustc_errors::{
    error_code, fallback_fluent_bundle, DiagnosticBuilder, DiagnosticId, DiagnosticMessage,
    ErrorGuaranteed, FluentBundle, IntoDiagnostic, LazyFallbackBundle, MultiSpan, Noted,
    TerminalUrl,
};
use rustc_macros::HashStable_Generic;
pub use rustc_span::def_id::StableCrateId;
use rustc_span::edition::Edition;
use rustc_span::source_map::{FileLoader, RealFileLoader, SourceMap, Span};
use rustc_span::{SourceFileHashAlgorithm, Symbol};
use rustc_target::asm::InlineAsmArch;
use rustc_target::spec::{CodeModel, PanicStrategy, RelocModel, RelroLevel};
use rustc_target::spec::{
    DebuginfoKind, SanitizerSet, SplitDebuginfo, StackProtector, Target, TargetTriple, TlsModel,
};

use std::cell::{self, RefCell};
use std::env;
use std::fmt;
use std::ops::{Div, Mul};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

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
#[derive(Clone, Copy, Debug, HashStable_Generic)]
pub struct Limit(pub usize);

impl Limit {
    /// Create a new limit from a `usize`.
    pub fn new(value: usize) -> Self {
        Limit(value)
    }

    /// Check that `value` is within the limit. Ensures that the same comparisons are used
    /// throughout the compiler, as mismatches can cause ICEs, see #72540.
    #[inline]
    pub fn value_within_limit(&self, value: usize) -> bool {
        value <= self.0
    }
}

impl From<usize> for Limit {
    fn from(value: usize) -> Self {
        Self::new(value)
    }
}

impl fmt::Display for Limit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
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

impl rustc_errors::IntoDiagnosticArg for Limit {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        self.to_string().into_diagnostic_arg()
    }
}

#[derive(Clone, Copy, Debug, HashStable_Generic)]
pub struct Limits {
    /// The maximum recursion limit for potentially infinitely recursive
    /// operations such as auto-dereference and monomorphization.
    pub recursion_limit: Limit,
    /// The size at which the `large_assignments` lint starts
    /// being emitted.
    pub move_size_limit: Limit,
    /// The maximum length of types during monomorphization.
    pub type_length_limit: Limit,
}

pub struct CompilerIO {
    pub input: Input,
    pub output_dir: Option<PathBuf>,
    pub output_file: Option<OutFileName>,
    pub temps_dir: Option<PathBuf>,
}

/// Represents the data associated with a compilation
/// session for a single crate.
pub struct Session {
    pub target: Target,
    pub host: Target,
    pub opts: config::Options,
    pub host_tlib_path: Lrc<SearchPath>,
    pub target_tlib_path: Lrc<SearchPath>,
    pub parse_sess: ParseSess,
    pub sysroot: PathBuf,
    /// Input, input file path and output file path to this compilation process.
    pub io: CompilerIO,

    crate_types: OnceCell<Vec<CrateType>>,
    /// The `stable_crate_id` is constructed out of the crate name and all the
    /// `-C metadata` arguments passed to the compiler. Its value forms a unique
    /// global identifier for the crate. It is used to allow multiple crates
    /// with the same name to coexist. See the
    /// `rustc_symbol_mangling` crate for more information.
    pub stable_crate_id: OnceCell<StableCrateId>,

    features: OnceCell<rustc_feature::Features>,

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

    /// Tracks fuel info if `-zfuel=crate=n` is specified.
    optimization_fuel: Lock<OptimizationFuel>,

    /// Always set to zero and incremented so that we can print fuel expended by a crate.
    pub print_fuel: AtomicU64,

    /// Loaded up early on in the initialization of this `Session` to avoid
    /// false positives about a job server in our environment.
    pub jobserver: Client,

    /// Cap lint level specified by a driver specifically.
    pub driver_lint_caps: FxHashMap<lint::LintId, lint::Level>,

    /// Tracks the current behavior of the CTFE engine when an error occurs.
    /// Options range from returning the error without a backtrace to returning an error
    /// and immediately printing the backtrace to stderr.
    /// The `Lock` is only used by miri to allow setting `ctfe_backtrace` after analysis when
    /// `MIRI_BACKTRACE` is set. This makes it only apply to miri's errors and not to all CTFE
    /// errors.
    pub ctfe_backtrace: Lock<CtfeBacktrace>,

    /// This tracks where `-Zunleash-the-miri-inside-of-you` was used to get around a
    /// const check, optionally with the relevant feature gate. We use this to
    /// warn about unleashing, but with a single diagnostic instead of dozens that
    /// drown everything else in noise.
    miri_unleashed_features: Lock<Vec<(Span, Option<Symbol>)>>,

    /// Architecture to use for interpreting asm!.
    pub asm_arch: Option<InlineAsmArch>,

    /// Set of enabled features for the current target.
    pub target_features: FxIndexSet<Symbol>,

    /// Set of enabled features for the current target, including unstable ones.
    pub unstable_target_features: FxIndexSet<Symbol>,

    /// The version of the rustc process, possibly including a commit hash and description.
    pub cfg_version: &'static str,
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

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub enum MetadataKind {
    None,
    Uncompressed,
    Compressed,
}

impl Session {
    pub fn miri_unleashed_feature(&self, span: Span, feature_gate: Option<Symbol>) {
        self.miri_unleashed_features.lock().push((span, feature_gate));
    }

    pub fn local_crate_source_file(&self) -> Option<PathBuf> {
        let path = self.io.input.opt_path()?;
        Some(self.opts.file_path_mapping().map_prefix(path).0.into_owned())
    }

    fn check_miri_unleashed_features(&self) {
        let unleashed_features = self.miri_unleashed_features.lock();
        if !unleashed_features.is_empty() {
            let mut must_err = false;
            // Create a diagnostic pointing at where things got unleashed.
            self.emit_warning(errors::SkippingConstChecks {
                unleashed_features: unleashed_features
                    .iter()
                    .map(|(span, gate)| {
                        gate.map(|gate| {
                            must_err = true;
                            errors::UnleashedFeatureHelp::Named { span: *span, gate }
                        })
                        .unwrap_or(errors::UnleashedFeatureHelp::Unnamed { span: *span })
                    })
                    .collect(),
            });

            // If we should err, make sure we did.
            if must_err && self.has_errors().is_none() {
                // We have skipped a feature gate, and not run into other errors... reject.
                self.emit_err(errors::NotCircumventFeature);
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
        if !self.opts.json_future_incompat {
            return;
        }

        let diags = self.diagnostic().take_future_breakage_diagnostics();
        if diags.is_empty() {
            return;
        }
        self.parse_sess.span_diagnostic.emit_future_breakage_report(diags);
    }

    pub fn local_stable_crate_id(&self) -> StableCrateId {
        self.stable_crate_id.get().copied().unwrap()
    }

    pub fn crate_types(&self) -> &[CrateType] {
        self.crate_types.get().unwrap().as_slice()
    }

    /// Returns true if the crate is a testing one.
    pub fn is_test_crate(&self) -> bool {
        self.opts.test
    }

    pub fn needs_crate_hash(&self) -> bool {
        // Why is the crate hash needed for these configurations?
        // - debug_assertions: for the "fingerprint the result" check in
        //   `rustc_query_system::query::plumbing::execute_job`.
        // - incremental: for query lookups.
        // - needs_metadata: for putting into crate metadata.
        // - instrument_coverage: for putting into coverage data (see
        //   `hash_mir_source`).
        cfg!(debug_assertions)
            || self.opts.incremental.is_some()
            || self.needs_metadata()
            || self.instrument_coverage()
    }

    pub fn metadata_kind(&self) -> MetadataKind {
        self.crate_types()
            .iter()
            .map(|ty| match *ty {
                CrateType::Executable | CrateType::Staticlib | CrateType::Cdylib => {
                    MetadataKind::None
                }
                CrateType::Rlib => MetadataKind::Uncompressed,
                CrateType::Dylib | CrateType::ProcMacro => MetadataKind::Compressed,
            })
            .max()
            .unwrap_or(MetadataKind::None)
    }

    pub fn needs_metadata(&self) -> bool {
        self.metadata_kind() != MetadataKind::None
    }

    pub fn init_crate_types(&self, crate_types: Vec<CrateType>) {
        self.crate_types.set(crate_types).expect("`crate_types` was initialized twice")
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_warn<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ()> {
        self.diagnostic().struct_span_warn(sp, msg)
    }
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_warn_with_expectation<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
        id: lint::LintExpectationId,
    ) -> DiagnosticBuilder<'_, ()> {
        self.diagnostic().struct_span_warn_with_expectation(sp, msg, id)
    }
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_warn_with_code<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_, ()> {
        self.diagnostic().struct_span_warn_with_code(sp, msg, code)
    }
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_warn(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        self.diagnostic().struct_warn(msg)
    }
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_warn_with_expectation(
        &self,
        msg: impl Into<DiagnosticMessage>,
        id: lint::LintExpectationId,
    ) -> DiagnosticBuilder<'_, ()> {
        self.diagnostic().struct_warn_with_expectation(msg, id)
    }
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_allow<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ()> {
        self.diagnostic().struct_span_allow(sp, msg)
    }
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_allow(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, ()> {
        self.diagnostic().struct_allow(msg)
    }
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_expect(
        &self,
        msg: impl Into<DiagnosticMessage>,
        id: lint::LintExpectationId,
    ) -> DiagnosticBuilder<'_, ()> {
        self.diagnostic().struct_expect(msg, id)
    }
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_err<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        self.diagnostic().struct_span_err(sp, msg)
    }
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_err_with_code<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        self.diagnostic().struct_span_err_with_code(sp, msg, code)
    }
    // FIXME: This method should be removed (every error should have an associated error code).
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_err(
        &self,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        self.parse_sess.struct_err(msg)
    }
    #[track_caller]
    #[rustc_lint_diagnostics]
    pub fn struct_err_with_code(
        &self,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        self.diagnostic().struct_err_with_code(msg, code)
    }
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_warn_with_code(
        &self,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_, ()> {
        self.diagnostic().struct_warn_with_code(msg, code)
    }
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn struct_span_fatal<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, !> {
        self.diagnostic().struct_span_fatal(sp, msg)
    }
    #[rustc_lint_diagnostics]
    pub fn struct_span_fatal_with_code<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> DiagnosticBuilder<'_, !> {
        self.diagnostic().struct_span_fatal_with_code(sp, msg, code)
    }
    #[rustc_lint_diagnostics]
    pub fn struct_fatal(&self, msg: impl Into<DiagnosticMessage>) -> DiagnosticBuilder<'_, !> {
        self.diagnostic().struct_fatal(msg)
    }

    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_fatal<S: Into<MultiSpan>>(&self, sp: S, msg: impl Into<DiagnosticMessage>) -> ! {
        self.diagnostic().span_fatal(sp, msg)
    }
    #[rustc_lint_diagnostics]
    pub fn span_fatal_with_code<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) -> ! {
        self.diagnostic().span_fatal_with_code(sp, msg, code)
    }
    #[rustc_lint_diagnostics]
    pub fn fatal(&self, msg: impl Into<DiagnosticMessage>) -> ! {
        self.diagnostic().fatal(msg).raise()
    }
    #[rustc_lint_diagnostics]
    #[track_caller]
    pub fn span_err<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
    ) -> ErrorGuaranteed {
        self.diagnostic().span_err(sp, msg)
    }
    #[rustc_lint_diagnostics]
    pub fn span_err_with_code<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) {
        self.diagnostic().span_err_with_code(sp, msg, code)
    }
    #[rustc_lint_diagnostics]
    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub fn err(&self, msg: impl Into<DiagnosticMessage>) -> ErrorGuaranteed {
        self.diagnostic().err(msg)
    }
    #[track_caller]
    pub fn create_err<'a>(
        &'a self,
        err: impl IntoDiagnostic<'a>,
    ) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        self.parse_sess.create_err(err)
    }
    #[track_caller]
    pub fn create_feature_err<'a>(
        &'a self,
        err: impl IntoDiagnostic<'a>,
        feature: Symbol,
    ) -> DiagnosticBuilder<'a, ErrorGuaranteed> {
        let mut err = self.parse_sess.create_err(err);
        if err.code.is_none() {
            err.code = std::option::Option::Some(error_code!(E0658));
        }
        add_feature_diagnostics(&mut err, &self.parse_sess, feature);
        err
    }
    #[track_caller]
    pub fn emit_err<'a>(&'a self, err: impl IntoDiagnostic<'a>) -> ErrorGuaranteed {
        self.parse_sess.emit_err(err)
    }
    #[track_caller]
    pub fn create_warning<'a>(
        &'a self,
        err: impl IntoDiagnostic<'a, ()>,
    ) -> DiagnosticBuilder<'a, ()> {
        self.parse_sess.create_warning(err)
    }
    #[track_caller]
    pub fn emit_warning<'a>(&'a self, warning: impl IntoDiagnostic<'a, ()>) {
        self.parse_sess.emit_warning(warning)
    }
    #[track_caller]
    pub fn create_note<'a>(
        &'a self,
        note: impl IntoDiagnostic<'a, Noted>,
    ) -> DiagnosticBuilder<'a, Noted> {
        self.parse_sess.create_note(note)
    }
    #[track_caller]
    pub fn emit_note<'a>(&'a self, note: impl IntoDiagnostic<'a, Noted>) -> Noted {
        self.parse_sess.emit_note(note)
    }
    #[track_caller]
    pub fn create_fatal<'a>(
        &'a self,
        fatal: impl IntoDiagnostic<'a, !>,
    ) -> DiagnosticBuilder<'a, !> {
        self.parse_sess.create_fatal(fatal)
    }
    #[track_caller]
    pub fn emit_fatal<'a>(&'a self, fatal: impl IntoDiagnostic<'a, !>) -> ! {
        self.parse_sess.emit_fatal(fatal)
    }
    #[inline]
    pub fn err_count(&self) -> usize {
        self.diagnostic().err_count()
    }
    pub fn has_errors(&self) -> Option<ErrorGuaranteed> {
        self.diagnostic().has_errors()
    }
    pub fn has_errors_or_delayed_span_bugs(&self) -> Option<ErrorGuaranteed> {
        self.diagnostic().has_errors_or_delayed_span_bugs()
    }
    pub fn is_compilation_going_to_fail(&self) -> Option<ErrorGuaranteed> {
        self.diagnostic().is_compilation_going_to_fail()
    }
    pub fn abort_if_errors(&self) {
        self.diagnostic().abort_if_errors();
    }
    pub fn compile_status(&self) -> Result<(), ErrorGuaranteed> {
        if let Some(reported) = self.diagnostic().has_errors_or_lint_errors() {
            let _ = self.diagnostic().emit_stashed_diagnostics();
            Err(reported)
        } else {
            Ok(())
        }
    }
    // FIXME(matthewjasper) Remove this method, it should never be needed.
    pub fn track_errors<F, T>(&self, f: F) -> Result<T, ErrorGuaranteed>
    where
        F: FnOnce() -> T,
    {
        let old_count = self.err_count();
        let result = f();
        if self.err_count() == old_count {
            Ok(result)
        } else {
            Err(self.delay_span_bug(
                rustc_span::DUMMY_SP,
                "`self.err_count()` changed but an error was not emitted",
            ))
        }
    }

    #[rustc_lint_diagnostics]
    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    #[track_caller]
    pub fn span_warn<S: Into<MultiSpan>>(&self, sp: S, msg: impl Into<DiagnosticMessage>) {
        self.diagnostic().span_warn(sp, msg)
    }

    #[rustc_lint_diagnostics]
    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub fn span_warn_with_code<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
        code: DiagnosticId,
    ) {
        self.diagnostic().span_warn_with_code(sp, msg, code)
    }

    #[rustc_lint_diagnostics]
    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub fn warn(&self, msg: impl Into<DiagnosticMessage>) {
        self.diagnostic().warn(msg)
    }

    /// Ensures that compilation cannot succeed.
    ///
    /// If this function has been called but no errors have been emitted and
    /// compilation succeeds, it will cause an internal compiler error (ICE).
    ///
    /// This can be used in code paths that should never run on successful compilations.
    /// For example, it can be used to create an [`ErrorGuaranteed`]
    /// (but you should prefer threading through the [`ErrorGuaranteed`] from an error emission directly).
    ///
    /// If no span is available, use [`DUMMY_SP`].
    ///
    /// [`DUMMY_SP`]: rustc_span::DUMMY_SP
    #[track_caller]
    pub fn delay_span_bug<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
    ) -> ErrorGuaranteed {
        self.diagnostic().delay_span_bug(sp, msg)
    }

    /// Used for code paths of expensive computations that should only take place when
    /// warnings or errors are emitted. If no messages are emitted ("good path"), then
    /// it's likely a bug.
    pub fn delay_good_path_bug(&self, msg: impl Into<DiagnosticMessage>) {
        if self.opts.unstable_opts.print_type_sizes
            || self.opts.unstable_opts.query_dep_graph
            || self.opts.unstable_opts.dump_mir.is_some()
            || self.opts.unstable_opts.unpretty.is_some()
            || self.opts.output_types.contains_key(&OutputType::Mir)
            || std::env::var_os("RUSTC_LOG").is_some()
        {
            return;
        }

        self.diagnostic().delay_good_path_bug(msg)
    }

    #[rustc_lint_diagnostics]
    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub fn note_without_error(&self, msg: impl Into<DiagnosticMessage>) {
        self.diagnostic().note_without_error(msg)
    }

    #[track_caller]
    #[rustc_lint_diagnostics]
    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub fn span_note_without_error<S: Into<MultiSpan>>(
        &self,
        sp: S,
        msg: impl Into<DiagnosticMessage>,
    ) {
        self.diagnostic().span_note_without_error(sp, msg)
    }

    #[rustc_lint_diagnostics]
    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub fn struct_note_without_error(
        &self,
        msg: impl Into<DiagnosticMessage>,
    ) -> DiagnosticBuilder<'_, ()> {
        self.diagnostic().struct_note_without_error(msg)
    }

    #[inline]
    pub fn diagnostic(&self) -> &rustc_errors::Handler {
        &self.parse_sess.span_diagnostic
    }

    #[inline]
    pub fn source_map(&self) -> &SourceMap {
        self.parse_sess.source_map()
    }

    /// Returns `true` if internal lints should be added to the lint store - i.e. if
    /// `-Zunstable-options` is provided and this isn't rustdoc (internal lints can trigger errors
    /// to be emitted under rustdoc).
    pub fn enable_internal_lints(&self) -> bool {
        self.unstable_options() && !self.opts.actually_rustdoc
    }

    pub fn instrument_coverage(&self) -> bool {
        self.opts.cg.instrument_coverage() != InstrumentCoverage::Off
    }

    pub fn instrument_coverage_except_unused_generics(&self) -> bool {
        self.opts.cg.instrument_coverage() == InstrumentCoverage::ExceptUnusedGenerics
    }

    pub fn instrument_coverage_except_unused_functions(&self) -> bool {
        self.opts.cg.instrument_coverage() == InstrumentCoverage::ExceptUnusedFunctions
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

    pub fn is_sanitizer_cfi_enabled(&self) -> bool {
        self.opts.unstable_opts.sanitizer.contains(SanitizerSet::CFI)
    }

    pub fn is_sanitizer_cfi_canonical_jump_tables_disabled(&self) -> bool {
        self.opts.unstable_opts.sanitizer_cfi_canonical_jump_tables == Some(false)
    }

    pub fn is_sanitizer_cfi_canonical_jump_tables_enabled(&self) -> bool {
        self.opts.unstable_opts.sanitizer_cfi_canonical_jump_tables == Some(true)
    }

    pub fn is_sanitizer_cfi_generalize_pointers_enabled(&self) -> bool {
        self.opts.unstable_opts.sanitizer_cfi_generalize_pointers == Some(true)
    }

    pub fn is_sanitizer_cfi_normalize_integers_enabled(&self) -> bool {
        self.opts.unstable_opts.sanitizer_cfi_normalize_integers == Some(true)
    }

    pub fn is_sanitizer_kcfi_enabled(&self) -> bool {
        self.opts.unstable_opts.sanitizer.contains(SanitizerSet::KCFI)
    }

    pub fn is_split_lto_unit_enabled(&self) -> bool {
        self.opts.unstable_opts.split_lto_unit == Some(true)
    }

    /// Check whether this compile session and crate type use static crt.
    pub fn crt_static(&self, crate_type: Option<CrateType>) -> bool {
        if !self.target.crt_static_respected {
            // If the target does not opt in to crt-static support, use its default.
            return self.target.crt_static_default;
        }

        let requested_features = self.opts.cg.target_feature.split(',');
        let found_negative = requested_features.clone().any(|r| r == "-crt-static");
        let found_positive = requested_features.clone().any(|r| r == "+crt-static");

        // JUSTIFICATION: necessary use of crate_types directly (see FIXME below)
        #[allow(rustc::bad_opt_access)]
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
            self.target.crt_static_default
        }
    }

    pub fn is_wasi_reactor(&self) -> bool {
        self.target.options.os == "wasi"
            && matches!(
                self.opts.unstable_opts.wasi_exec_model,
                Some(config::WasiExecModel::Reactor)
            )
    }

    /// Returns `true` if the target can use the current split debuginfo configuration.
    pub fn target_can_use_split_dwarf(&self) -> bool {
        self.target.debuginfo_kind == DebuginfoKind::Dwarf
    }

    pub fn generate_proc_macro_decls_symbol(&self, stable_crate_id: StableCrateId) -> String {
        format!("__rustc_proc_macro_decls_{:08x}__", stable_crate_id.as_u64())
    }

    pub fn target_filesearch(&self, kind: PathKind) -> filesearch::FileSearch<'_> {
        filesearch::FileSearch::new(
            &self.sysroot,
            self.opts.target_triple.triple(),
            &self.opts.search_paths,
            &self.target_tlib_path,
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

    /// Returns a list of directories where target-specific tool binaries are located.
    pub fn get_tools_search_paths(&self, self_contained: bool) -> Vec<PathBuf> {
        let rustlib_path = rustc_target::target_rustlib_path(&self.sysroot, &config::host_triple());
        let p = PathBuf::from_iter([
            Path::new(&self.sysroot),
            Path::new(&rustlib_path),
            Path::new("bin"),
        ]);
        if self_contained { vec![p.clone(), p.join("self-contained")] } else { vec![p] }
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
        eprintln!(
            "Total time spent computing symbol hashes:      {}",
            duration_to_secs_str(*self.perf_stats.symbol_hash_time.lock())
        );
        eprintln!(
            "Total queries canonicalized:                   {}",
            self.perf_stats.queries_canonicalized.load(Ordering::Relaxed)
        );
        eprintln!(
            "normalize_generic_arg_after_erasing_regions:   {}",
            self.perf_stats.normalize_generic_arg_after_erasing_regions.load(Ordering::Relaxed)
        );
        eprintln!(
            "normalize_projection_ty:                       {}",
            self.perf_stats.normalize_projection_ty.load(Ordering::Relaxed)
        );
    }

    /// We want to know if we're allowed to do an optimization for crate foo from -z fuel=foo=n.
    /// This expends fuel if applicable, and records fuel if applicable.
    pub fn consider_optimizing(
        &self,
        get_crate_name: impl Fn() -> Symbol,
        msg: impl Fn() -> String,
    ) -> bool {
        let mut ret = true;
        if let Some((ref c, _)) = self.opts.unstable_opts.fuel {
            if c == get_crate_name().as_str() {
                assert_eq!(self.threads(), 1);
                let mut fuel = self.optimization_fuel.lock();
                ret = fuel.remaining != 0;
                if fuel.remaining == 0 && !fuel.out_of_fuel {
                    if self.diagnostic().can_emit_warnings() {
                        // We only call `msg` in case we can actually emit warnings.
                        // Otherwise, this could cause a `delay_good_path_bug` to
                        // trigger (issue #79546).
                        self.emit_warning(errors::OptimisationFuelExhausted { msg: msg() });
                    }
                    fuel.out_of_fuel = true;
                } else if fuel.remaining > 0 {
                    fuel.remaining -= 1;
                }
            }
        }
        if let Some(ref c) = self.opts.unstable_opts.print_fuel {
            if c == get_crate_name().as_str() {
                assert_eq!(self.threads(), 1);
                self.print_fuel.fetch_add(1, SeqCst);
            }
        }
        ret
    }

    /// Is this edition 2015?
    pub fn is_rust_2015(&self) -> bool {
        self.edition().is_rust_2015()
    }

    /// Are we allowed to use features from the Rust 2018 edition?
    pub fn rust_2018(&self) -> bool {
        self.edition().rust_2018()
    }

    /// Are we allowed to use features from the Rust 2021 edition?
    pub fn rust_2021(&self) -> bool {
        self.edition().rust_2021()
    }

    /// Are we allowed to use features from the Rust 2024 edition?
    pub fn rust_2024(&self) -> bool {
        self.edition().rust_2024()
    }

    /// Returns `true` if we cannot skip the PLT for shared library calls.
    pub fn needs_plt(&self) -> bool {
        // Check if the current target usually needs PLT to be enabled.
        // The user can use the command line flag to override it.
        let needs_plt = self.target.needs_plt;

        let dbg_opts = &self.opts.unstable_opts;

        let relro_level = dbg_opts.relro_level.unwrap_or(self.target.relro_level);

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
        // AddressSanitizer and KernelAddressSanitizer uses lifetimes to detect use after scope bugs.
        // MemorySanitizer uses lifetimes to detect use of uninitialized stack variables.
        // HWAddressSanitizer will use lifetimes to detect use after scope bugs in the future.
        || self.opts.unstable_opts.sanitizer.intersects(SanitizerSet::ADDRESS | SanitizerSet::KERNELADDRESS | SanitizerSet::MEMORY | SanitizerSet::HWADDRESS)
    }

    pub fn diagnostic_width(&self) -> usize {
        let default_column_width = 140;
        if let Some(width) = self.opts.diagnostic_width {
            width
        } else if self.opts.unstable_opts.ui_testing {
            default_column_width
        } else {
            termize::dimensions().map_or(default_column_width, |(w, _)| w)
        }
    }
}

// JUSTIFICATION: defn of the suggested wrapper fns
#[allow(rustc::bad_opt_access)]
impl Session {
    pub fn verbose(&self) -> bool {
        self.opts.unstable_opts.verbose
    }

    pub fn verify_llvm_ir(&self) -> bool {
        self.opts.unstable_opts.verify_llvm_ir || option_env!("RUSTC_VERIFY_LLVM_IR").is_some()
    }

    pub fn binary_dep_depinfo(&self) -> bool {
        self.opts.unstable_opts.binary_dep_depinfo
    }

    pub fn mir_opt_level(&self) -> usize {
        self.opts
            .unstable_opts
            .mir_opt_level
            .unwrap_or_else(|| if self.opts.optimize != OptLevel::No { 2 } else { 1 })
    }

    /// Calculates the flavor of LTO to use for this compilation.
    pub fn lto(&self) -> config::Lto {
        // If our target has codegen requirements ignore the command line
        if self.target.requires_lto {
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
                // The user explicitly asked for ThinLTO
                return config::Lto::Thin;
            }
        }

        // Ok at this point the target doesn't require anything and the user
        // hasn't asked for anything. Our next decision is whether or not
        // we enable "auto" ThinLTO where we use multiple codegen units and
        // then do ThinLTO over those codegen units. The logic below will
        // either return `No` or `ThinLocal`.

        // If processing command line options determined that we're incompatible
        // with ThinLTO (e.g., `-C lto --emit llvm-ir`) then return that option.
        if self.opts.cli_forced_local_thinlto_off {
            return config::Lto::No;
        }

        // If `-Z thinlto` specified process that, but note that this is mostly
        // a deprecated option now that `-C lto=thin` exists.
        if let Some(enabled) = self.opts.unstable_opts.thinlto {
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
        self.opts.cg.panic.unwrap_or(self.target.panic_strategy)
    }

    pub fn fewer_names(&self) -> bool {
        if let Some(fewer_names) = self.opts.unstable_opts.fewer_names {
            fewer_names
        } else {
            let more_names = self.opts.output_types.contains_key(&OutputType::LlvmAssembly)
                || self.opts.output_types.contains_key(&OutputType::Bitcode)
                // AddressSanitizer and MemorySanitizer use alloca name when reporting an issue.
                || self.opts.unstable_opts.sanitizer.intersects(SanitizerSet::ADDRESS | SanitizerSet::MEMORY);
            !more_names
        }
    }

    pub fn unstable_options(&self) -> bool {
        self.opts.unstable_opts.unstable_options
    }

    pub fn is_nightly_build(&self) -> bool {
        self.opts.unstable_features.is_nightly_build()
    }

    pub fn overflow_checks(&self) -> bool {
        self.opts.cg.overflow_checks.unwrap_or(self.opts.debug_assertions)
    }

    pub fn relocation_model(&self) -> RelocModel {
        self.opts.cg.relocation_model.unwrap_or(self.target.relocation_model)
    }

    pub fn code_model(&self) -> Option<CodeModel> {
        self.opts.cg.code_model.or(self.target.code_model)
    }

    pub fn tls_model(&self) -> TlsModel {
        self.opts.unstable_opts.tls_model.unwrap_or(self.target.tls_model)
    }

    pub fn split_debuginfo(&self) -> SplitDebuginfo {
        self.opts.cg.split_debuginfo.unwrap_or(self.target.split_debuginfo)
    }

    pub fn stack_protector(&self) -> StackProtector {
        if self.target.options.supports_stack_protector {
            self.opts.unstable_opts.stack_protector
        } else {
            StackProtector::None
        }
    }

    pub fn must_emit_unwind_tables(&self) -> bool {
        // This is used to control the emission of the `uwtable` attribute on
        // LLVM functions.
        //
        // Unwind tables are needed when compiling with `-C panic=unwind`, but
        // LLVM won't omit unwind tables unless the function is also marked as
        // `nounwind`, so users are allowed to disable `uwtable` emission.
        // Historically rustc always emits `uwtable` attributes by default, so
        // even they can be disabled, they're still emitted by default.
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
        self.target.requires_uwtable
            || self.opts.cg.force_unwind_tables.unwrap_or(
                self.panic_strategy() == PanicStrategy::Unwind || self.target.default_uwtable,
            )
    }

    /// Returns the number of query threads that should be used for this
    /// compilation
    #[inline]
    pub fn threads(&self) -> usize {
        self.opts.unstable_opts.threads
    }

    /// Returns the number of codegen units that should be used for this
    /// compilation
    pub fn codegen_units(&self) -> usize {
        if let Some(n) = self.opts.cli_forced_codegen_units {
            return n;
        }
        if let Some(n) = self.target.default_codegen_units {
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
        self.opts.unstable_opts.teach && self.diagnostic().must_teach(code)
    }

    pub fn edition(&self) -> Edition {
        self.opts.edition
    }

    pub fn link_dead_code(&self) -> bool {
        self.opts.cg.link_dead_code.unwrap_or(false)
    }
}

// JUSTIFICATION: part of session construction
#[allow(rustc::bad_opt_access)]
fn default_emitter(
    sopts: &config::Options,
    registry: rustc_errors::registry::Registry,
    source_map: Lrc<SourceMap>,
    bundle: Option<Lrc<FluentBundle>>,
    fallback_bundle: LazyFallbackBundle,
) -> Box<dyn Emitter + sync::Send> {
    let macro_backtrace = sopts.unstable_opts.macro_backtrace;
    let track_diagnostics = sopts.unstable_opts.track_diagnostics;
    let terminal_url = match sopts.unstable_opts.terminal_urls {
        TerminalUrl::Auto => {
            match (std::env::var("COLORTERM").as_deref(), std::env::var("TERM").as_deref()) {
                (Ok("truecolor"), Ok("xterm-256color"))
                    if sopts.unstable_features.is_nightly_build() =>
                {
                    TerminalUrl::Yes
                }
                _ => TerminalUrl::No,
            }
        }
        t => t,
    };
    match sopts.error_format {
        config::ErrorOutputType::HumanReadable(kind) => {
            let (short, color_config) = kind.unzip();

            if let HumanReadableErrorType::AnnotateSnippet(_) = kind {
                let emitter = AnnotateSnippetEmitterWriter::new(
                    Some(source_map),
                    bundle,
                    fallback_bundle,
                    short,
                    macro_backtrace,
                );
                Box::new(emitter.ui_testing(sopts.unstable_opts.ui_testing))
            } else {
                let emitter = EmitterWriter::stderr(
                    color_config,
                    Some(source_map),
                    bundle,
                    fallback_bundle,
                    short,
                    sopts.unstable_opts.teach,
                    sopts.diagnostic_width,
                    macro_backtrace,
                    track_diagnostics,
                    terminal_url,
                );
                Box::new(emitter.ui_testing(sopts.unstable_opts.ui_testing))
            }
        }
        config::ErrorOutputType::Json { pretty, json_rendered } => Box::new(
            JsonEmitter::stderr(
                Some(registry),
                source_map,
                bundle,
                fallback_bundle,
                pretty,
                json_rendered,
                sopts.diagnostic_width,
                macro_backtrace,
                track_diagnostics,
                terminal_url,
            )
            .ui_testing(sopts.unstable_opts.ui_testing),
        ),
    }
}

// JUSTIFICATION: literally session construction
#[allow(rustc::bad_opt_access)]
pub fn build_session(
    sopts: config::Options,
    io: CompilerIO,
    bundle: Option<Lrc<rustc_errors::FluentBundle>>,
    registry: rustc_errors::registry::Registry,
    fluent_resources: Vec<&'static str>,
    driver_lint_caps: FxHashMap<lint::LintId, lint::Level>,
    file_loader: Option<Box<dyn FileLoader + Send + Sync + 'static>>,
    target_override: Option<Target>,
    cfg_version: &'static str,
) -> Session {
    // FIXME: This is not general enough to make the warning lint completely override
    // normal diagnostic warnings, since the warning lint can also be denied and changed
    // later via the source code.
    let warnings_allow = sopts
        .lint_opts
        .iter()
        .rfind(|&(key, _)| *key == "warnings")
        .is_some_and(|&(_, level)| level == lint::Allow);
    let cap_lints_allow = sopts.lint_cap.is_some_and(|cap| cap == lint::Allow);
    let can_emit_warnings = !(warnings_allow || cap_lints_allow);

    let sysroot = match &sopts.maybe_sysroot {
        Some(sysroot) => sysroot.clone(),
        None => filesearch::get_or_default_sysroot().expect("Failed finding sysroot"),
    };

    let target_cfg = config::build_target_config(&sopts, target_override, &sysroot);
    let host_triple = TargetTriple::from_triple(config::host_triple());
    let (host, target_warnings) = Target::search(&host_triple, &sysroot).unwrap_or_else(|e| {
        early_error(sopts.error_format, format!("Error loading host specification: {e}"))
    });
    for warning in target_warnings.warning_messages() {
        early_warn(sopts.error_format, warning)
    }

    let loader = file_loader.unwrap_or_else(|| Box::new(RealFileLoader));
    let hash_kind = sopts.unstable_opts.src_hash_algorithm.unwrap_or_else(|| {
        if target_cfg.is_like_msvc {
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

    let fallback_bundle = fallback_fluent_bundle(
        fluent_resources,
        sopts.unstable_opts.translate_directionality_markers,
    );
    let emitter = default_emitter(&sopts, registry, source_map.clone(), bundle, fallback_bundle);

    let span_diagnostic = rustc_errors::Handler::with_emitter_and_flags(
        emitter,
        sopts.unstable_opts.diagnostic_handler_flags(can_emit_warnings),
    );

    let self_profiler = if let SwitchWithOptPath::Enabled(ref d) = sopts.unstable_opts.self_profile
    {
        let directory =
            if let Some(ref directory) = d { directory } else { std::path::Path::new(".") };

        let profiler = SelfProfiler::new(
            directory,
            sopts.crate_name.as_deref(),
            sopts.unstable_opts.self_profile_events.as_deref(),
            &sopts.unstable_opts.self_profile_counter,
        );
        match profiler {
            Ok(profiler) => Some(Arc::new(profiler)),
            Err(e) => {
                early_warn(sopts.error_format, format!("failed to create profiler: {e}"));
                None
            }
        }
    } else {
        None
    };

    let mut parse_sess = ParseSess::with_span_handler(span_diagnostic, source_map);
    parse_sess.assume_incomplete_release = sopts.unstable_opts.assume_incomplete_release;

    let host_triple = config::host_triple();
    let target_triple = sopts.target_triple.triple();
    let host_tlib_path = Lrc::new(SearchPath::from_sysroot_and_triple(&sysroot, host_triple));
    let target_tlib_path = if host_triple == target_triple {
        // Use the same `SearchPath` if host and target triple are identical to avoid unnecessary
        // rescanning of the target lib path and an unnecessary allocation.
        host_tlib_path.clone()
    } else {
        Lrc::new(SearchPath::from_sysroot_and_triple(&sysroot, target_triple))
    };

    let optimization_fuel = Lock::new(OptimizationFuel {
        remaining: sopts.unstable_opts.fuel.as_ref().map_or(0, |&(_, i)| i),
        out_of_fuel: false,
    });
    let print_fuel = AtomicU64::new(0);

    let cgu_reuse_tracker = if sopts.unstable_opts.query_dep_graph {
        CguReuseTracker::new()
    } else {
        CguReuseTracker::new_disabled()
    };

    let prof = SelfProfilerRef::new(
        self_profiler,
        sopts.unstable_opts.time_passes.then(|| sopts.unstable_opts.time_passes_format),
    );

    let ctfe_backtrace = Lock::new(match env::var("RUSTC_CTFE_BACKTRACE") {
        Ok(ref val) if val == "immediate" => CtfeBacktrace::Immediate,
        Ok(ref val) if val != "0" => CtfeBacktrace::Capture,
        _ => CtfeBacktrace::Disabled,
    });

    let asm_arch =
        if target_cfg.allow_asm { InlineAsmArch::from_str(&target_cfg.arch).ok() } else { None };

    let sess = Session {
        target: target_cfg,
        host,
        opts: sopts,
        host_tlib_path,
        target_tlib_path,
        parse_sess,
        sysroot,
        io,
        crate_types: OnceCell::new(),
        stable_crate_id: OnceCell::new(),
        features: OnceCell::new(),
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
        optimization_fuel,
        print_fuel,
        jobserver: jobserver::client(),
        driver_lint_caps,
        ctfe_backtrace,
        miri_unleashed_features: Lock::new(Default::default()),
        asm_arch,
        target_features: Default::default(),
        unstable_target_features: Default::default(),
        cfg_version,
    };

    validate_commandline_args_with_session_available(&sess);

    sess
}

/// Validate command line arguments with a `Session`.
///
/// If it is useful to have a Session available already for validating a commandline argument, you
/// can do so here.
// JUSTIFICATION: needs to access args to validate them
#[allow(rustc::bad_opt_access)]
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
        && sess.target.is_like_windows
    {
        sess.emit_err(errors::LinkerPluginToWindowsNotSupported);
    }

    // Make sure that any given profiling data actually exists so LLVM can't
    // decide to silently skip PGO.
    if let Some(ref path) = sess.opts.cg.profile_use {
        if !path.exists() {
            sess.emit_err(errors::ProfileUseFileDoesNotExist { path });
        }
    }

    // Do the same for sample profile data.
    if let Some(ref path) = sess.opts.unstable_opts.profile_sample_use {
        if !path.exists() {
            sess.emit_err(errors::ProfileSampleUseFileDoesNotExist { path });
        }
    }

    // Unwind tables cannot be disabled if the target requires them.
    if let Some(include_uwtables) = sess.opts.cg.force_unwind_tables {
        if sess.target.requires_uwtable && !include_uwtables {
            sess.emit_err(errors::TargetRequiresUnwindTables);
        }
    }

    // Sanitizers can only be used on platforms that we know have working sanitizer codegen.
    let supported_sanitizers = sess.target.options.supported_sanitizers;
    let unsupported_sanitizers = sess.opts.unstable_opts.sanitizer - supported_sanitizers;
    match unsupported_sanitizers.into_iter().count() {
        0 => {}
        1 => {
            sess.emit_err(errors::SanitizerNotSupported { us: unsupported_sanitizers.to_string() });
        }
        _ => {
            sess.emit_err(errors::SanitizersNotSupported {
                us: unsupported_sanitizers.to_string(),
            });
        }
    }
    // Cannot mix and match sanitizers.
    let mut sanitizer_iter = sess.opts.unstable_opts.sanitizer.into_iter();
    if let (Some(first), Some(second)) = (sanitizer_iter.next(), sanitizer_iter.next()) {
        sess.emit_err(errors::CannotMixAndMatchSanitizers {
            first: first.to_string(),
            second: second.to_string(),
        });
    }

    // Cannot enable crt-static with sanitizers on Linux
    if sess.crt_static(None) && !sess.opts.unstable_opts.sanitizer.is_empty() {
        sess.emit_err(errors::CannotEnableCrtStaticLinux);
    }

    // LLVM CFI requires LTO.
    if sess.is_sanitizer_cfi_enabled()
        && !(sess.lto() == config::Lto::Fat
            || sess.lto() == config::Lto::Thin
            || sess.opts.cg.linker_plugin_lto.enabled())
    {
        sess.emit_err(errors::SanitizerCfiRequiresLto);
    }

    // LLVM CFI is incompatible with LLVM KCFI.
    if sess.is_sanitizer_cfi_enabled() && sess.is_sanitizer_kcfi_enabled() {
        sess.emit_err(errors::CannotMixAndMatchSanitizers {
            first: "cfi".to_string(),
            second: "kcfi".to_string(),
        });
    }

    // Canonical jump tables requires CFI.
    if sess.is_sanitizer_cfi_canonical_jump_tables_disabled() {
        if !sess.is_sanitizer_cfi_enabled() {
            sess.emit_err(errors::SanitizerCfiCanonicalJumpTablesRequiresCfi);
        }
    }

    // LLVM CFI pointer generalization requires CFI or KCFI.
    if sess.is_sanitizer_cfi_generalize_pointers_enabled() {
        if !(sess.is_sanitizer_cfi_enabled() || sess.is_sanitizer_kcfi_enabled()) {
            sess.emit_err(errors::SanitizerCfiGeneralizePointersRequiresCfi);
        }
    }

    // LLVM CFI integer normalization requires CFI or KCFI.
    if sess.is_sanitizer_cfi_normalize_integers_enabled() {
        if !(sess.is_sanitizer_cfi_enabled() || sess.is_sanitizer_kcfi_enabled()) {
            sess.emit_err(errors::SanitizerCfiNormalizeIntegersRequiresCfi);
        }
    }

    // LTO unit splitting requires LTO.
    if sess.is_split_lto_unit_enabled()
        && !(sess.lto() == config::Lto::Fat
            || sess.lto() == config::Lto::Thin
            || sess.opts.cg.linker_plugin_lto.enabled())
    {
        sess.emit_err(errors::SplitLtoUnitRequiresLto);
    }

    // VFE requires LTO.
    if sess.lto() != config::Lto::Fat {
        if sess.opts.unstable_opts.virtual_function_elimination {
            sess.emit_err(errors::UnstableVirtualFunctionElimination);
        }
    }

    if sess.opts.unstable_opts.stack_protector != StackProtector::None {
        if !sess.target.options.supports_stack_protector {
            sess.emit_warning(errors::StackProtectorNotSupportedForTarget {
                stack_protector: sess.opts.unstable_opts.stack_protector,
                target_triple: &sess.opts.target_triple,
            });
        }
    }

    if sess.opts.unstable_opts.branch_protection.is_some() && sess.target.arch != "aarch64" {
        sess.emit_err(errors::BranchProtectionRequiresAArch64);
    }

    if let Some(dwarf_version) = sess.opts.unstable_opts.dwarf_version {
        if dwarf_version > 5 {
            sess.emit_err(errors::UnsupportedDwarfVersion { dwarf_version });
        }
    }

    if !sess.target.options.supported_split_debuginfo.contains(&sess.split_debuginfo())
        && !sess.opts.unstable_opts.unstable_options
    {
        sess.emit_err(errors::SplitDebugInfoUnstablePlatform { debuginfo: sess.split_debuginfo() });
    }

    if sess.opts.unstable_opts.instrument_xray.is_some() && !sess.target.options.supports_xray {
        sess.emit_err(errors::InstrumentationNotSupported { us: "XRay".to_string() });
    }

    if let Some(flavor) = sess.opts.cg.linker_flavor {
        if let Some(compatible_list) = sess.target.linker_flavor.check_compatibility(flavor) {
            let flavor = flavor.desc();
            sess.emit_err(errors::IncompatibleLinkerFlavor { flavor, compatible_list });
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

fn early_error_handler(output: config::ErrorOutputType) -> rustc_errors::Handler {
    // FIXME(#100717): early errors aren't translated at the moment, so this is fine, but it will
    // need to reference every crate that might emit an early error for translation to work.
    let fallback_bundle =
        fallback_fluent_bundle(vec![rustc_errors::DEFAULT_LOCALE_RESOURCE], false);
    let emitter: Box<dyn Emitter + sync::Send> = match output {
        config::ErrorOutputType::HumanReadable(kind) => {
            let (short, color_config) = kind.unzip();
            Box::new(EmitterWriter::stderr(
                color_config,
                None,
                None,
                fallback_bundle,
                short,
                false,
                None,
                false,
                false,
                TerminalUrl::No,
            ))
        }
        config::ErrorOutputType::Json { pretty, json_rendered } => Box::new(JsonEmitter::basic(
            pretty,
            json_rendered,
            None,
            fallback_bundle,
            None,
            false,
            false,
            TerminalUrl::No,
        )),
    };
    rustc_errors::Handler::with_emitter(true, None, emitter)
}

#[allow(rustc::untranslatable_diagnostic)]
#[allow(rustc::diagnostic_outside_of_impl)]
#[must_use = "ErrorGuaranteed must be returned from `run_compiler` in order to exit with a non-zero status code"]
pub fn early_error_no_abort(
    output: config::ErrorOutputType,
    msg: impl Into<DiagnosticMessage>,
) -> ErrorGuaranteed {
    early_error_handler(output).struct_err(msg).emit()
}

#[allow(rustc::untranslatable_diagnostic)]
#[allow(rustc::diagnostic_outside_of_impl)]
pub fn early_error(output: config::ErrorOutputType, msg: impl Into<DiagnosticMessage>) -> ! {
    early_error_handler(output).struct_fatal(msg).emit()
}

#[allow(rustc::untranslatable_diagnostic)]
#[allow(rustc::diagnostic_outside_of_impl)]
pub fn early_warn(output: config::ErrorOutputType, msg: impl Into<DiagnosticMessage>) {
    early_error_handler(output).struct_warn(msg).emit()
}
