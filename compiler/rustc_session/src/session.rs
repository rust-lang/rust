use std::any::Any;
use std::ops::{Div, Mul};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::{env, fmt, io};

use rand::{RngCore, rng};
use rustc_data_structures::base_n::{CASE_INSENSITIVE, ToBaseN};
use rustc_data_structures::flock;
use rustc_data_structures::fx::{FxHashMap, FxIndexSet};
use rustc_data_structures::profiling::{SelfProfiler, SelfProfilerRef};
use rustc_data_structures::sync::{DynSend, DynSync, Lock, MappedReadGuard, ReadGuard, RwLock};
use rustc_errors::annotate_snippet_emitter_writer::AnnotateSnippetEmitter;
use rustc_errors::codes::*;
use rustc_errors::emitter::{
    DynEmitter, HumanEmitter, HumanReadableErrorType, OutputTheme, stderr_destination,
};
use rustc_errors::json::JsonEmitter;
use rustc_errors::timings::TimingSectionHandler;
use rustc_errors::translation::Translator;
use rustc_errors::{
    Diag, DiagCtxt, DiagCtxtHandle, DiagMessage, Diagnostic, ErrorGuaranteed, FatalAbort,
    TerminalUrl, fallback_fluent_bundle,
};
use rustc_macros::HashStable_Generic;
pub use rustc_span::def_id::StableCrateId;
use rustc_span::edition::Edition;
use rustc_span::source_map::{FilePathMapping, SourceMap};
use rustc_span::{FileNameDisplayPreference, RealFileName, Span, Symbol};
use rustc_target::asm::InlineAsmArch;
use rustc_target::spec::{
    CodeModel, DebuginfoKind, PanicStrategy, RelocModel, RelroLevel, SanitizerSet,
    SmallDataThresholdSupport, SplitDebuginfo, StackProtector, SymbolVisibility, Target,
    TargetTuple, TlsModel, apple,
};

use crate::code_stats::CodeStats;
pub use crate::code_stats::{DataTypeKind, FieldInfo, FieldKind, SizeKind, VariantInfo};
use crate::config::{
    self, CoverageLevel, CrateType, DebugInfo, ErrorOutputType, FunctionReturn, Input,
    InstrumentCoverage, OptLevel, OutFileName, OutputType, RemapPathScopeComponents,
    SwitchWithOptPath,
};
use crate::filesearch::FileSearch;
use crate::parse::{ParseSess, add_feature_diagnostics};
use crate::search_paths::SearchPath;
use crate::{errors, filesearch, lint};

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

    /// Create a new unlimited limit.
    pub fn unlimited() -> Self {
        Limit(usize::MAX)
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

impl rustc_errors::IntoDiagArg for Limit {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        self.to_string().into_diag_arg(&mut None)
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
    /// The maximum pattern complexity allowed (internal only).
    pub pattern_complexity_limit: Limit,
}

pub struct CompilerIO {
    pub input: Input,
    pub output_dir: Option<PathBuf>,
    pub output_file: Option<OutFileName>,
    pub temps_dir: Option<PathBuf>,
}

pub trait LintStoreMarker: Any + DynSync + DynSend {}

/// Represents the data associated with a compilation
/// session for a single crate.
pub struct Session {
    pub target: Target,
    pub host: Target,
    pub opts: config::Options,
    pub target_tlib_path: Arc<SearchPath>,
    pub psess: ParseSess,
    /// Input, input file path and output file path to this compilation process.
    pub io: CompilerIO,

    incr_comp_session: RwLock<IncrCompSession>,

    /// Used by `-Z self-profile`.
    pub prof: SelfProfilerRef,

    /// Used to emit section timings events (enabled by `--json=timings`).
    pub timings: TimingSectionHandler,

    /// Data about code being compiled, gathered during compilation.
    pub code_stats: CodeStats,

    /// This only ever stores a `LintStore` but we don't want a dependency on that type here.
    pub lint_store: Option<Arc<dyn LintStoreMarker>>,

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

    /// The inner atomic value is set to true when a feature marked as `internal` is
    /// enabled. Makes it so that "please report a bug" is hidden, as ICEs with
    /// internal features are wontfix, and they are usually the cause of the ICEs.
    /// None signifies that this is not tracked.
    pub using_internal_features: &'static AtomicBool,

    /// All commandline args used to invoke the compiler, with @file args fully expanded.
    /// This will only be used within debug info, e.g. in the pdb file on windows
    /// This is mainly useful for other tools that reads that debuginfo to figure out
    /// how to call the compiler with the same arguments.
    pub expanded_args: Vec<String>,

    target_filesearch: FileSearch,
    host_filesearch: FileSearch,

    /// A random string generated per invocation of rustc.
    ///
    /// This is prepended to all temporary files so that they do not collide
    /// during concurrent invocations of rustc, or past invocations that were
    /// preserved with a flag like `-C save-temps`, since these files may be
    /// hard linked.
    pub invocation_temp: Option<String>,
}

#[derive(Clone, Copy)]
pub enum CodegenUnits {
    /// Specified by the user. In this case we try fairly hard to produce the
    /// number of CGUs requested.
    User(usize),

    /// A default value, i.e. not specified by the user. In this case we take
    /// more liberties about CGU formation, e.g. avoid producing very small
    /// CGUs.
    Default(usize),
}

impl CodegenUnits {
    pub fn as_usize(self) -> usize {
        match self {
            CodegenUnits::User(n) => n,
            CodegenUnits::Default(n) => n,
        }
    }
}

impl Session {
    pub fn miri_unleashed_feature(&self, span: Span, feature_gate: Option<Symbol>) {
        self.miri_unleashed_features.lock().push((span, feature_gate));
    }

    pub fn local_crate_source_file(&self) -> Option<RealFileName> {
        Some(self.source_map().path_mapping().to_real_filename(self.io.input.opt_path()?))
    }

    fn check_miri_unleashed_features(&self) -> Option<ErrorGuaranteed> {
        let mut guar = None;
        let unleashed_features = self.miri_unleashed_features.lock();
        if !unleashed_features.is_empty() {
            let mut must_err = false;
            // Create a diagnostic pointing at where things got unleashed.
            self.dcx().emit_warn(errors::SkippingConstChecks {
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
            if must_err && self.dcx().has_errors().is_none() {
                // We have skipped a feature gate, and not run into other errors... reject.
                guar = Some(self.dcx().emit_err(errors::NotCircumventFeature));
            }
        }
        guar
    }

    /// Invoked all the way at the end to finish off diagnostics printing.
    pub fn finish_diagnostics(&self) -> Option<ErrorGuaranteed> {
        let mut guar = None;
        guar = guar.or(self.check_miri_unleashed_features());
        guar = guar.or(self.dcx().emit_stashed_diagnostics());
        self.dcx().print_error_count();
        if self.opts.json_future_incompat {
            self.dcx().emit_future_breakage_report();
        }
        guar
    }

    /// Returns true if the crate is a testing one.
    pub fn is_test_crate(&self) -> bool {
        self.opts.test
    }

    /// `feature` must be a language feature.
    #[track_caller]
    pub fn create_feature_err<'a>(&'a self, err: impl Diagnostic<'a>, feature: Symbol) -> Diag<'a> {
        let mut err = self.dcx().create_err(err);
        if err.code.is_none() {
            #[allow(rustc::diagnostic_outside_of_impl)]
            err.code(E0658);
        }
        add_feature_diagnostics(&mut err, self, feature);
        err
    }

    /// Record the fact that we called `trimmed_def_paths`, and do some
    /// checking about whether its cost was justified.
    pub fn record_trimmed_def_paths(&self) {
        if self.opts.unstable_opts.print_type_sizes
            || self.opts.unstable_opts.query_dep_graph
            || self.opts.unstable_opts.dump_mir.is_some()
            || self.opts.unstable_opts.unpretty.is_some()
            || self.opts.output_types.contains_key(&OutputType::Mir)
            || std::env::var_os("RUSTC_LOG").is_some()
        {
            return;
        }

        self.dcx().set_must_produce_diag()
    }

    #[inline]
    pub fn dcx(&self) -> DiagCtxtHandle<'_> {
        self.psess.dcx()
    }

    #[inline]
    pub fn source_map(&self) -> &SourceMap {
        self.psess.source_map()
    }

    /// Returns `true` if internal lints should be added to the lint store - i.e. if
    /// `-Zunstable-options` is provided and this isn't rustdoc (internal lints can trigger errors
    /// to be emitted under rustdoc).
    pub fn enable_internal_lints(&self) -> bool {
        self.unstable_options() && !self.opts.actually_rustdoc
    }

    pub fn instrument_coverage(&self) -> bool {
        self.opts.cg.instrument_coverage() != InstrumentCoverage::No
    }

    pub fn instrument_coverage_branch(&self) -> bool {
        self.instrument_coverage()
            && self.opts.unstable_opts.coverage_options.level >= CoverageLevel::Branch
    }

    pub fn instrument_coverage_condition(&self) -> bool {
        self.instrument_coverage()
            && self.opts.unstable_opts.coverage_options.level >= CoverageLevel::Condition
    }

    pub fn instrument_coverage_mcdc(&self) -> bool {
        self.instrument_coverage()
            && self.opts.unstable_opts.coverage_options.level >= CoverageLevel::Mcdc
    }

    /// True if `-Zcoverage-options=no-mir-spans` was passed.
    pub fn coverage_no_mir_spans(&self) -> bool {
        self.opts.unstable_opts.coverage_options.no_mir_spans
    }

    /// True if `-Zcoverage-options=discard-all-spans-in-codegen` was passed.
    pub fn coverage_discard_all_spans_in_codegen(&self) -> bool {
        self.opts.unstable_opts.coverage_options.discard_all_spans_in_codegen
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

    pub fn is_sanitizer_kcfi_arity_enabled(&self) -> bool {
        self.opts.unstable_opts.sanitizer_kcfi_arity == Some(true)
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

    pub fn target_filesearch(&self) -> &filesearch::FileSearch {
        &self.target_filesearch
    }
    pub fn host_filesearch(&self) -> &filesearch::FileSearch {
        &self.host_filesearch
    }

    /// Returns a list of directories where target-specific tool binaries are located. Some fallback
    /// directories are also returned, for example if `--sysroot` is used but tools are missing
    /// (#125246): we also add the bin directories to the sysroot where rustc is located.
    pub fn get_tools_search_paths(&self, self_contained: bool) -> Vec<PathBuf> {
        let search_paths = self
            .opts
            .sysroot
            .all_paths()
            .map(|sysroot| filesearch::make_target_bin_path(&sysroot, config::host_tuple()));

        if self_contained {
            // The self-contained tools are expected to be e.g. in `bin/self-contained` in the
            // sysroot's `rustlib` path, so we add such a subfolder to the bin path, and the
            // fallback paths.
            search_paths.flat_map(|path| [path.clone(), path.join("self-contained")]).collect()
        } else {
            search_paths.collect()
        }
    }

    pub fn init_incr_comp_session(&self, session_dir: PathBuf, lock_file: flock::Lock) {
        let mut incr_comp_session = self.incr_comp_session.borrow_mut();

        if let IncrCompSession::NotInitialized = *incr_comp_session {
        } else {
            panic!("Trying to initialize IncrCompSession `{:?}`", *incr_comp_session)
        }

        *incr_comp_session =
            IncrCompSession::Active { session_directory: session_dir, _lock_file: lock_file };
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

    pub fn incr_comp_session_dir(&self) -> MappedReadGuard<'_, PathBuf> {
        let incr_comp_session = self.incr_comp_session.borrow();
        ReadGuard::map(incr_comp_session, |incr_comp_session| match *incr_comp_session {
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

    pub fn incr_comp_session_dir_opt(&self) -> Option<MappedReadGuard<'_, PathBuf>> {
        self.opts.incremental.as_ref().map(|_| self.incr_comp_session_dir())
    }

    /// Is this edition 2015?
    pub fn is_rust_2015(&self) -> bool {
        self.edition().is_rust_2015()
    }

    /// Are we allowed to use features from the Rust 2018 edition?
    pub fn at_least_rust_2018(&self) -> bool {
        self.edition().at_least_rust_2018()
    }

    /// Are we allowed to use features from the Rust 2021 edition?
    pub fn at_least_rust_2021(&self) -> bool {
        self.edition().at_least_rust_2021()
    }

    /// Are we allowed to use features from the Rust 2024 edition?
    pub fn at_least_rust_2024(&self) -> bool {
        self.edition().at_least_rust_2024()
    }

    /// Returns `true` if we should use the PLT for shared library calls.
    pub fn needs_plt(&self) -> bool {
        // Check if the current target usually wants PLT to be enabled.
        // The user can use the command line flag to override it.
        let want_plt = self.target.plt_by_default;

        let dbg_opts = &self.opts.unstable_opts;

        let relro_level = self.opts.cg.relro_level.unwrap_or(self.target.relro_level);

        // Only enable this optimization by default if full relro is also enabled.
        // In this case, lazy binding was already unavailable, so nothing is lost.
        // This also ensures `-Wl,-z,now` is supported by the linker.
        let full_relro = RelroLevel::Full == relro_level;

        // If user didn't explicitly forced us to use / skip the PLT,
        // then use it unless the target doesn't want it by default or the full relro forces it on.
        dbg_opts.plt.unwrap_or(want_plt || !full_relro)
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

    /// Returns the default symbol visibility.
    pub fn default_visibility(&self) -> SymbolVisibility {
        self.opts
            .unstable_opts
            .default_visibility
            .or(self.target.options.default_visibility)
            .unwrap_or(SymbolVisibility::Interposable)
    }

    pub fn staticlib_components(&self, verbatim: bool) -> (&str, &str) {
        if verbatim {
            ("", "")
        } else {
            (&*self.target.staticlib_prefix, &*self.target.staticlib_suffix)
        }
    }
}

// JUSTIFICATION: defn of the suggested wrapper fns
#[allow(rustc::bad_opt_access)]
impl Session {
    pub fn verbose_internals(&self) -> bool {
        self.opts.unstable_opts.verbose_internals
    }

    pub fn print_llvm_stats(&self) -> bool {
        self.opts.unstable_opts.print_codegen_stats
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
        if self.codegen_units().as_usize() == 1 {
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

    pub fn ub_checks(&self) -> bool {
        self.opts.unstable_opts.ub_checks.unwrap_or(self.opts.debug_assertions)
    }

    pub fn contract_checks(&self) -> bool {
        self.opts.unstable_opts.contract_checks.unwrap_or(false)
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

    pub fn direct_access_external_data(&self) -> Option<bool> {
        self.opts
            .unstable_opts
            .direct_access_external_data
            .or(self.target.direct_access_external_data)
    }

    pub fn split_debuginfo(&self) -> SplitDebuginfo {
        self.opts.cg.split_debuginfo.unwrap_or(self.target.split_debuginfo)
    }

    /// Returns the DWARF version passed on the CLI or the default for the target.
    pub fn dwarf_version(&self) -> u32 {
        self.opts
            .cg
            .dwarf_version
            .or(self.opts.unstable_opts.dwarf_version)
            .unwrap_or(self.target.default_dwarf_version)
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
    pub fn codegen_units(&self) -> CodegenUnits {
        if let Some(n) = self.opts.cli_forced_codegen_units {
            return CodegenUnits::User(n);
        }
        if let Some(n) = self.target.default_codegen_units {
            return CodegenUnits::Default(n as usize);
        }

        // If incremental compilation is turned on, we default to a high number
        // codegen units in order to reduce the "collateral damage" small
        // changes cause.
        if self.opts.incremental.is_some() {
            return CodegenUnits::Default(256);
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
        CodegenUnits::Default(16)
    }

    pub fn teach(&self, code: ErrCode) -> bool {
        self.opts.unstable_opts.teach && self.dcx().must_teach(code)
    }

    pub fn edition(&self) -> Edition {
        self.opts.edition
    }

    pub fn link_dead_code(&self) -> bool {
        self.opts.cg.link_dead_code.unwrap_or(false)
    }

    pub fn filename_display_preference(
        &self,
        scope: RemapPathScopeComponents,
    ) -> FileNameDisplayPreference {
        assert!(
            scope.bits().count_ones() == 1,
            "one and only one scope should be passed to `Session::filename_display_preference`"
        );
        if self.opts.unstable_opts.remap_path_scope.contains(scope) {
            FileNameDisplayPreference::Remapped
        } else {
            FileNameDisplayPreference::Local
        }
    }

    /// Get the deployment target on Apple platforms based on the standard environment variables,
    /// or fall back to the minimum version supported by `rustc`.
    ///
    /// This should be guarded behind `if sess.target.is_like_darwin`.
    pub fn apple_deployment_target(&self) -> apple::OSVersion {
        let min = apple::OSVersion::minimum_deployment_target(&self.target);
        let env_var = apple::deployment_target_env_var(&self.target.os);

        // FIXME(madsmtm): Track changes to this.
        if let Ok(deployment_target) = env::var(env_var) {
            match apple::OSVersion::from_str(&deployment_target) {
                Ok(version) => {
                    let os_min = apple::OSVersion::os_minimum_deployment_target(&self.target.os);
                    // It is common that the deployment target is set a bit too low, for example on
                    // macOS Aarch64 to also target older x86_64. So we only want to warn when variable
                    // is lower than the minimum OS supported by rustc, not when the variable is lower
                    // than the minimum for a specific target.
                    if version < os_min {
                        self.dcx().emit_warn(errors::AppleDeploymentTarget::TooLow {
                            env_var,
                            version: version.fmt_pretty().to_string(),
                            os_min: os_min.fmt_pretty().to_string(),
                        });
                    }

                    // Raise the deployment target to the minimum supported.
                    version.max(min)
                }
                Err(error) => {
                    self.dcx().emit_err(errors::AppleDeploymentTarget::Invalid { env_var, error });
                    min
                }
            }
        } else {
            // If no deployment target variable is set, default to the minimum found above.
            min
        }
    }
}

// JUSTIFICATION: part of session construction
#[allow(rustc::bad_opt_access)]
fn default_emitter(
    sopts: &config::Options,
    source_map: Arc<SourceMap>,
    translator: Translator,
) -> Box<DynEmitter> {
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

    let source_map = if sopts.unstable_opts.link_only { None } else { Some(source_map) };

    match sopts.error_format {
        config::ErrorOutputType::HumanReadable { kind, color_config } => {
            let short = kind.short();

            if let HumanReadableErrorType::AnnotateSnippet = kind {
                let emitter =
                    AnnotateSnippetEmitter::new(source_map, translator, short, macro_backtrace);
                Box::new(emitter.ui_testing(sopts.unstable_opts.ui_testing))
            } else {
                let emitter = HumanEmitter::new(stderr_destination(color_config), translator)
                    .sm(source_map)
                    .short_message(short)
                    .diagnostic_width(sopts.diagnostic_width)
                    .macro_backtrace(macro_backtrace)
                    .track_diagnostics(track_diagnostics)
                    .terminal_url(terminal_url)
                    .theme(if let HumanReadableErrorType::Unicode = kind {
                        OutputTheme::Unicode
                    } else {
                        OutputTheme::Ascii
                    })
                    .ignored_directories_in_source_blocks(
                        sopts.unstable_opts.ignore_directory_in_diagnostics_source_blocks.clone(),
                    );
                Box::new(emitter.ui_testing(sopts.unstable_opts.ui_testing))
            }
        }
        config::ErrorOutputType::Json { pretty, json_rendered, color_config } => Box::new(
            JsonEmitter::new(
                Box::new(io::BufWriter::new(io::stderr())),
                source_map,
                translator,
                pretty,
                json_rendered,
                color_config,
            )
            .ui_testing(sopts.unstable_opts.ui_testing)
            .ignored_directories_in_source_blocks(
                sopts.unstable_opts.ignore_directory_in_diagnostics_source_blocks.clone(),
            )
            .diagnostic_width(sopts.diagnostic_width)
            .macro_backtrace(macro_backtrace)
            .track_diagnostics(track_diagnostics)
            .terminal_url(terminal_url),
        ),
    }
}

// JUSTIFICATION: literally session construction
#[allow(rustc::bad_opt_access)]
#[allow(rustc::untranslatable_diagnostic)] // FIXME: make this translatable
pub fn build_session(
    sopts: config::Options,
    io: CompilerIO,
    fluent_bundle: Option<Arc<rustc_errors::FluentBundle>>,
    registry: rustc_errors::registry::Registry,
    fluent_resources: Vec<&'static str>,
    driver_lint_caps: FxHashMap<lint::LintId, lint::Level>,
    target: Target,
    cfg_version: &'static str,
    ice_file: Option<PathBuf>,
    using_internal_features: &'static AtomicBool,
    expanded_args: Vec<String>,
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

    let translator = Translator {
        fluent_bundle,
        fallback_fluent_bundle: fallback_fluent_bundle(
            fluent_resources,
            sopts.unstable_opts.translate_directionality_markers,
        ),
    };
    let source_map = rustc_span::source_map::get_source_map().unwrap();
    let emitter = default_emitter(&sopts, Arc::clone(&source_map), translator);

    let mut dcx = DiagCtxt::new(emitter)
        .with_flags(sopts.unstable_opts.dcx_flags(can_emit_warnings))
        .with_registry(registry);
    if let Some(ice_file) = ice_file {
        dcx = dcx.with_ice_file(ice_file);
    }

    let host_triple = TargetTuple::from_tuple(config::host_tuple());
    let (host, target_warnings) = Target::search(&host_triple, sopts.sysroot.path())
        .unwrap_or_else(|e| dcx.handle().fatal(format!("Error loading host specification: {e}")));
    for warning in target_warnings.warning_messages() {
        dcx.handle().warn(warning)
    }

    let self_profiler = if let SwitchWithOptPath::Enabled(ref d) = sopts.unstable_opts.self_profile
    {
        let directory = if let Some(directory) = d { directory } else { std::path::Path::new(".") };

        let profiler = SelfProfiler::new(
            directory,
            sopts.crate_name.as_deref(),
            sopts.unstable_opts.self_profile_events.as_deref(),
            &sopts.unstable_opts.self_profile_counter,
        );
        match profiler {
            Ok(profiler) => Some(Arc::new(profiler)),
            Err(e) => {
                dcx.handle().emit_warn(errors::FailedToCreateProfiler { err: e.to_string() });
                None
            }
        }
    } else {
        None
    };

    let mut psess = ParseSess::with_dcx(dcx, source_map);
    psess.assume_incomplete_release = sopts.unstable_opts.assume_incomplete_release;

    let host_triple = config::host_tuple();
    let target_triple = sopts.target_triple.tuple();
    // FIXME use host sysroot?
    let host_tlib_path =
        Arc::new(SearchPath::from_sysroot_and_triple(sopts.sysroot.path(), host_triple));
    let target_tlib_path = if host_triple == target_triple {
        // Use the same `SearchPath` if host and target triple are identical to avoid unnecessary
        // rescanning of the target lib path and an unnecessary allocation.
        Arc::clone(&host_tlib_path)
    } else {
        Arc::new(SearchPath::from_sysroot_and_triple(sopts.sysroot.path(), target_triple))
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

    let asm_arch = if target.allow_asm { InlineAsmArch::from_str(&target.arch).ok() } else { None };
    let target_filesearch =
        filesearch::FileSearch::new(&sopts.search_paths, &target_tlib_path, &target);
    let host_filesearch = filesearch::FileSearch::new(&sopts.search_paths, &host_tlib_path, &host);

    let invocation_temp = sopts
        .incremental
        .as_ref()
        .map(|_| rng().next_u32().to_base_fixed_len(CASE_INSENSITIVE).to_string());

    let timings = TimingSectionHandler::new(sopts.json_timings);

    let sess = Session {
        target,
        host,
        opts: sopts,
        target_tlib_path,
        psess,
        io,
        incr_comp_session: RwLock::new(IncrCompSession::NotInitialized),
        prof,
        timings,
        code_stats: Default::default(),
        lint_store: None,
        driver_lint_caps,
        ctfe_backtrace,
        miri_unleashed_features: Lock::new(Default::default()),
        asm_arch,
        target_features: Default::default(),
        unstable_target_features: Default::default(),
        cfg_version,
        using_internal_features,
        expanded_args,
        target_filesearch,
        host_filesearch,
        invocation_temp,
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
        sess.dcx().emit_err(errors::LinkerPluginToWindowsNotSupported);
    }

    // Make sure that any given profiling data actually exists so LLVM can't
    // decide to silently skip PGO.
    if let Some(ref path) = sess.opts.cg.profile_use {
        if !path.exists() {
            sess.dcx().emit_err(errors::ProfileUseFileDoesNotExist { path });
        }
    }

    // Do the same for sample profile data.
    if let Some(ref path) = sess.opts.unstable_opts.profile_sample_use {
        if !path.exists() {
            sess.dcx().emit_err(errors::ProfileSampleUseFileDoesNotExist { path });
        }
    }

    // Unwind tables cannot be disabled if the target requires them.
    if let Some(include_uwtables) = sess.opts.cg.force_unwind_tables {
        if sess.target.requires_uwtable && !include_uwtables {
            sess.dcx().emit_err(errors::TargetRequiresUnwindTables);
        }
    }

    // Sanitizers can only be used on platforms that we know have working sanitizer codegen.
    let supported_sanitizers = sess.target.options.supported_sanitizers;
    let mut unsupported_sanitizers = sess.opts.unstable_opts.sanitizer - supported_sanitizers;
    // Niche: if `fixed-x18`, or effectively switching on `reserved-x18` flag, is enabled
    // we should allow Shadow Call Stack sanitizer.
    if sess.opts.unstable_opts.fixed_x18 && sess.target.arch == "aarch64" {
        unsupported_sanitizers -= SanitizerSet::SHADOWCALLSTACK;
    }
    match unsupported_sanitizers.into_iter().count() {
        0 => {}
        1 => {
            sess.dcx()
                .emit_err(errors::SanitizerNotSupported { us: unsupported_sanitizers.to_string() });
        }
        _ => {
            sess.dcx().emit_err(errors::SanitizersNotSupported {
                us: unsupported_sanitizers.to_string(),
            });
        }
    }

    // Cannot mix and match mutually-exclusive sanitizers.
    if let Some((first, second)) = sess.opts.unstable_opts.sanitizer.mutually_exclusive() {
        sess.dcx().emit_err(errors::CannotMixAndMatchSanitizers {
            first: first.to_string(),
            second: second.to_string(),
        });
    }

    // Cannot enable crt-static with sanitizers on Linux
    if sess.crt_static(None)
        && !sess.opts.unstable_opts.sanitizer.is_empty()
        && !sess.target.is_like_msvc
    {
        sess.dcx().emit_err(errors::CannotEnableCrtStaticLinux);
    }

    // LLVM CFI requires LTO.
    if sess.is_sanitizer_cfi_enabled()
        && !(sess.lto() == config::Lto::Fat || sess.opts.cg.linker_plugin_lto.enabled())
    {
        sess.dcx().emit_err(errors::SanitizerCfiRequiresLto);
    }

    // KCFI requires panic=abort
    if sess.is_sanitizer_kcfi_enabled() && sess.panic_strategy() != PanicStrategy::Abort {
        sess.dcx().emit_err(errors::SanitizerKcfiRequiresPanicAbort);
    }

    // LLVM CFI using rustc LTO requires a single codegen unit.
    if sess.is_sanitizer_cfi_enabled()
        && sess.lto() == config::Lto::Fat
        && (sess.codegen_units().as_usize() != 1)
    {
        sess.dcx().emit_err(errors::SanitizerCfiRequiresSingleCodegenUnit);
    }

    // Canonical jump tables requires CFI.
    if sess.is_sanitizer_cfi_canonical_jump_tables_disabled() {
        if !sess.is_sanitizer_cfi_enabled() {
            sess.dcx().emit_err(errors::SanitizerCfiCanonicalJumpTablesRequiresCfi);
        }
    }

    // KCFI arity indicator requires KCFI.
    if sess.is_sanitizer_kcfi_arity_enabled() && !sess.is_sanitizer_kcfi_enabled() {
        sess.dcx().emit_err(errors::SanitizerKcfiArityRequiresKcfi);
    }

    // LLVM CFI pointer generalization requires CFI or KCFI.
    if sess.is_sanitizer_cfi_generalize_pointers_enabled() {
        if !(sess.is_sanitizer_cfi_enabled() || sess.is_sanitizer_kcfi_enabled()) {
            sess.dcx().emit_err(errors::SanitizerCfiGeneralizePointersRequiresCfi);
        }
    }

    // LLVM CFI integer normalization requires CFI or KCFI.
    if sess.is_sanitizer_cfi_normalize_integers_enabled() {
        if !(sess.is_sanitizer_cfi_enabled() || sess.is_sanitizer_kcfi_enabled()) {
            sess.dcx().emit_err(errors::SanitizerCfiNormalizeIntegersRequiresCfi);
        }
    }

    // LTO unit splitting requires LTO.
    if sess.is_split_lto_unit_enabled()
        && !(sess.lto() == config::Lto::Fat
            || sess.lto() == config::Lto::Thin
            || sess.opts.cg.linker_plugin_lto.enabled())
    {
        sess.dcx().emit_err(errors::SplitLtoUnitRequiresLto);
    }

    // VFE requires LTO.
    if sess.lto() != config::Lto::Fat {
        if sess.opts.unstable_opts.virtual_function_elimination {
            sess.dcx().emit_err(errors::UnstableVirtualFunctionElimination);
        }
    }

    if sess.opts.unstable_opts.stack_protector != StackProtector::None {
        if !sess.target.options.supports_stack_protector {
            sess.dcx().emit_warn(errors::StackProtectorNotSupportedForTarget {
                stack_protector: sess.opts.unstable_opts.stack_protector,
                target_triple: &sess.opts.target_triple,
            });
        }
    }

    if sess.opts.unstable_opts.small_data_threshold.is_some() {
        if sess.target.small_data_threshold_support() == SmallDataThresholdSupport::None {
            sess.dcx().emit_warn(errors::SmallDataThresholdNotSupportedForTarget {
                target_triple: &sess.opts.target_triple,
            })
        }
    }

    if sess.opts.unstable_opts.branch_protection.is_some() && sess.target.arch != "aarch64" {
        sess.dcx().emit_err(errors::BranchProtectionRequiresAArch64);
    }

    if let Some(dwarf_version) =
        sess.opts.cg.dwarf_version.or(sess.opts.unstable_opts.dwarf_version)
    {
        // DWARF 1 is not supported by LLVM and DWARF 6 is not yet finalized.
        if dwarf_version < 2 || dwarf_version > 5 {
            sess.dcx().emit_err(errors::UnsupportedDwarfVersion { dwarf_version });
        }
    }

    if !sess.target.options.supported_split_debuginfo.contains(&sess.split_debuginfo())
        && !sess.opts.unstable_opts.unstable_options
    {
        sess.dcx()
            .emit_err(errors::SplitDebugInfoUnstablePlatform { debuginfo: sess.split_debuginfo() });
    }

    if sess.opts.unstable_opts.embed_source {
        let dwarf_version = sess.dwarf_version();

        if dwarf_version < 5 {
            sess.dcx().emit_warn(errors::EmbedSourceInsufficientDwarfVersion { dwarf_version });
        }

        if sess.opts.debuginfo == DebugInfo::None {
            sess.dcx().emit_warn(errors::EmbedSourceRequiresDebugInfo);
        }
    }

    if sess.opts.unstable_opts.instrument_xray.is_some() && !sess.target.options.supports_xray {
        sess.dcx().emit_err(errors::InstrumentationNotSupported { us: "XRay".to_string() });
    }

    if let Some(flavor) = sess.opts.cg.linker_flavor {
        if let Some(compatible_list) = sess.target.linker_flavor.check_compatibility(flavor) {
            let flavor = flavor.desc();
            sess.dcx().emit_err(errors::IncompatibleLinkerFlavor { flavor, compatible_list });
        }
    }

    if sess.opts.unstable_opts.function_return != FunctionReturn::default() {
        if sess.target.arch != "x86" && sess.target.arch != "x86_64" {
            sess.dcx().emit_err(errors::FunctionReturnRequiresX86OrX8664);
        }
    }

    if let Some(regparm) = sess.opts.unstable_opts.regparm {
        if regparm > 3 {
            sess.dcx().emit_err(errors::UnsupportedRegparm { regparm });
        }
        if sess.target.arch != "x86" {
            sess.dcx().emit_err(errors::UnsupportedRegparmArch);
        }
    }
    if sess.opts.unstable_opts.reg_struct_return {
        if sess.target.arch != "x86" {
            sess.dcx().emit_err(errors::UnsupportedRegStructReturnArch);
        }
    }

    // The code model check applies to `thunk` and `thunk-extern`, but not `thunk-inline`, so it is
    // kept as a `match` to force a change if new ones are added, even if we currently only support
    // `thunk-extern` like Clang.
    match sess.opts.unstable_opts.function_return {
        FunctionReturn::Keep => (),
        FunctionReturn::ThunkExtern => {
            // FIXME: In principle, the inherited base LLVM target code model could be large,
            // but this only checks whether we were passed one explicitly (like Clang does).
            if let Some(code_model) = sess.code_model()
                && code_model == CodeModel::Large
            {
                sess.dcx().emit_err(errors::FunctionReturnThunkExternRequiresNonLargeCodeModel);
            }
        }
    }

    if sess.opts.cg.soft_float {
        if sess.target.arch == "arm" {
            sess.dcx().emit_warn(errors::SoftFloatDeprecated);
        } else {
            // All `use_softfp` does is the equivalent of `-mfloat-abi` in GCC/clang, which only exists on ARM targets.
            // We document this flag to only affect `*eabihf` targets, so let's show a warning for all other targets.
            sess.dcx().emit_warn(errors::SoftFloatIgnored);
        }
    }
}

/// Holds data on the current incremental compilation session, if there is one.
#[derive(Debug)]
enum IncrCompSession {
    /// This is the state the session will be in until the incr. comp. dir is
    /// needed.
    NotInitialized,
    /// This is the state during which the session directory is private and can
    /// be modified. `_lock_file` is never directly used, but its presence
    /// alone has an effect, because the file will unlock when the session is
    /// dropped.
    Active { session_directory: PathBuf, _lock_file: flock::Lock },
    /// This is the state after the session directory has been finalized. In this
    /// state, the contents of the directory must not be modified any more.
    Finalized { session_directory: PathBuf },
    /// This is an error state that is reached when some compilation error has
    /// occurred. It indicates that the contents of the session directory must
    /// not be used, since they might be invalid.
    InvalidBecauseOfErrors { session_directory: PathBuf },
}

/// A wrapper around an [`DiagCtxt`] that is used for early error emissions.
pub struct EarlyDiagCtxt {
    dcx: DiagCtxt,
}

impl EarlyDiagCtxt {
    pub fn new(output: ErrorOutputType) -> Self {
        let emitter = mk_emitter(output);
        Self { dcx: DiagCtxt::new(emitter) }
    }

    /// Swap out the underlying dcx once we acquire the user's preference on error emission
    /// format. If `early_err` was previously called this will panic.
    pub fn set_error_format(&mut self, output: ErrorOutputType) {
        assert!(self.dcx.handle().has_errors().is_none());

        let emitter = mk_emitter(output);
        self.dcx = DiagCtxt::new(emitter);
    }

    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub fn early_note(&self, msg: impl Into<DiagMessage>) {
        self.dcx.handle().note(msg)
    }

    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub fn early_help(&self, msg: impl Into<DiagMessage>) {
        self.dcx.handle().struct_help(msg).emit()
    }

    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    #[must_use = "raise_fatal must be called on the returned ErrorGuaranteed in order to exit with a non-zero status code"]
    pub fn early_err(&self, msg: impl Into<DiagMessage>) -> ErrorGuaranteed {
        self.dcx.handle().err(msg)
    }

    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub fn early_fatal(&self, msg: impl Into<DiagMessage>) -> ! {
        self.dcx.handle().fatal(msg)
    }

    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub fn early_struct_fatal(&self, msg: impl Into<DiagMessage>) -> Diag<'_, FatalAbort> {
        self.dcx.handle().struct_fatal(msg)
    }

    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub fn early_warn(&self, msg: impl Into<DiagMessage>) {
        self.dcx.handle().warn(msg)
    }

    #[allow(rustc::untranslatable_diagnostic)]
    #[allow(rustc::diagnostic_outside_of_impl)]
    pub fn early_struct_warn(&self, msg: impl Into<DiagMessage>) -> Diag<'_, ()> {
        self.dcx.handle().struct_warn(msg)
    }
}

fn mk_emitter(output: ErrorOutputType) -> Box<DynEmitter> {
    // FIXME(#100717): early errors aren't translated at the moment, so this is fine, but it will
    // need to reference every crate that might emit an early error for translation to work.
    let translator =
        Translator::with_fallback_bundle(vec![rustc_errors::DEFAULT_LOCALE_RESOURCE], false);
    let emitter: Box<DynEmitter> = match output {
        config::ErrorOutputType::HumanReadable { kind, color_config } => {
            let short = kind.short();
            Box::new(
                HumanEmitter::new(stderr_destination(color_config), translator)
                    .theme(if let HumanReadableErrorType::Unicode = kind {
                        OutputTheme::Unicode
                    } else {
                        OutputTheme::Ascii
                    })
                    .short_message(short),
            )
        }
        config::ErrorOutputType::Json { pretty, json_rendered, color_config } => {
            Box::new(JsonEmitter::new(
                Box::new(io::BufWriter::new(io::stderr())),
                Some(Arc::new(SourceMap::new(FilePathMapping::empty()))),
                translator,
                pretty,
                json_rendered,
                color_config,
            ))
        }
    };
    emitter
}

pub trait RemapFileNameExt {
    type Output<'a>
    where
        Self: 'a;

    /// Returns a possibly remapped filename based on the passed scope and remap cli options.
    ///
    /// One and only one scope should be passed to this method, it will panic otherwise.
    fn for_scope(&self, sess: &Session, scope: RemapPathScopeComponents) -> Self::Output<'_>;
}

impl RemapFileNameExt for rustc_span::FileName {
    type Output<'a> = rustc_span::FileNameDisplay<'a>;

    fn for_scope(&self, sess: &Session, scope: RemapPathScopeComponents) -> Self::Output<'_> {
        assert!(
            scope.bits().count_ones() == 1,
            "one and only one scope should be passed to for_scope"
        );
        if sess.opts.unstable_opts.remap_path_scope.contains(scope) {
            self.prefer_remapped_unconditionaly()
        } else {
            self.prefer_local()
        }
    }
}

impl RemapFileNameExt for rustc_span::RealFileName {
    type Output<'a> = &'a Path;

    fn for_scope(&self, sess: &Session, scope: RemapPathScopeComponents) -> Self::Output<'_> {
        assert!(
            scope.bits().count_ones() == 1,
            "one and only one scope should be passed to for_scope"
        );
        if sess.opts.unstable_opts.remap_path_scope.contains(scope) {
            self.remapped_path_if_available()
        } else {
            self.local_path_if_available()
        }
    }
}
