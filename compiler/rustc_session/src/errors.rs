use std::num::{NonZero, ParseIntError};

use rustc_ast::token;
use rustc_ast::util::literal::LitError;
use rustc_errors::codes::*;
use rustc_errors::{
    Diag, DiagCtxtHandle, DiagMessage, Diagnostic, EmissionGuarantee, ErrorGuaranteed, Level,
    MultiSpan,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_span::{Span, Symbol};
use rustc_target::spec::{SplitDebuginfo, StackProtector, TargetTuple};

use crate::parse::ParseSess;

#[derive(Diagnostic)]
pub(crate) enum AppleDeploymentTarget {
    #[diag("failed to parse deployment target specified in {$env_var}: {$error}")]
    Invalid { env_var: &'static str, error: ParseIntError },
    #[diag(
        "deployment target in {$env_var} was set to {$version}, but the minimum supported by `rustc` is {$os_min}"
    )]
    TooLow { env_var: &'static str, version: String, os_min: String },
}

pub(crate) struct FeatureGateError {
    pub(crate) span: MultiSpan,
    pub(crate) explain: DiagMessage,
}

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for FeatureGateError {
    #[track_caller]
    fn into_diag(self, dcx: DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        Diag::new(dcx, level, self.explain).with_span(self.span).with_code(E0658)
    }
}

#[derive(Subdiagnostic)]
#[note("see issue #{$n} <https://github.com/rust-lang/rust/issues/{$n}> for more information")]
pub(crate) struct FeatureDiagnosticForIssue {
    pub(crate) n: NonZero<u32>,
}

#[derive(Subdiagnostic)]
#[note("this compiler was built on {$date}; consider upgrading it if it is out of date")]
pub(crate) struct SuggestUpgradeCompiler {
    date: &'static str,
}

impl SuggestUpgradeCompiler {
    pub(crate) fn ui_testing() -> Self {
        Self { date: "YYYY-MM-DD" }
    }

    pub(crate) fn new() -> Option<Self> {
        let date = option_env!("CFG_VER_DATE")?;

        Some(Self { date })
    }
}

#[derive(Subdiagnostic)]
#[help("add `#![feature({$feature})]` to the crate attributes to enable")]
pub(crate) struct FeatureDiagnosticHelp {
    pub(crate) feature: Symbol,
}

#[derive(Subdiagnostic)]
#[suggestion(
    "add `#![feature({$feature})]` to the crate attributes to enable",
    applicability = "maybe-incorrect",
    code = "#![feature({feature})]\n"
)]
pub struct FeatureDiagnosticSuggestion {
    pub feature: Symbol,
    #[primary_span]
    pub span: Span,
}

#[derive(Subdiagnostic)]
#[help("add `-Zcrate-attr=\"feature({$feature})\"` to the command-line options to enable")]
pub(crate) struct CliFeatureDiagnosticHelp {
    pub(crate) feature: Symbol,
}

#[derive(Diagnostic)]
#[diag("must be a name of an associated function")]
pub struct MustBeNameOfAssociatedFunction {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "`-Zunleash-the-miri-inside-of-you` may not be used to circumvent feature gates, except when testing error paths in the CTFE engine"
)]
pub(crate) struct NotCircumventFeature;

#[derive(Diagnostic)]
#[diag(
    "linker plugin based LTO is not supported together with `-C prefer-dynamic` when targeting Windows-like targets"
)]
pub(crate) struct LinkerPluginToWindowsNotSupported;

#[derive(Diagnostic)]
#[diag("file `{$path}` passed to `-C profile-use` does not exist")]
pub(crate) struct ProfileUseFileDoesNotExist<'a> {
    pub(crate) path: &'a std::path::Path,
}

#[derive(Diagnostic)]
#[diag("file `{$path}` passed to `-C profile-sample-use` does not exist")]
pub(crate) struct ProfileSampleUseFileDoesNotExist<'a> {
    pub(crate) path: &'a std::path::Path,
}

#[derive(Diagnostic)]
#[diag("target requires unwind tables, they cannot be disabled with `-C force-unwind-tables=no`")]
pub(crate) struct TargetRequiresUnwindTables;

#[derive(Diagnostic)]
#[diag("{$us} instrumentation is not supported for this target")]
pub(crate) struct InstrumentationNotSupported {
    pub(crate) us: String,
}

#[derive(Diagnostic)]
#[diag("{$us} sanitizer is not supported for this target")]
pub(crate) struct SanitizerNotSupported {
    pub(crate) us: String,
}

#[derive(Diagnostic)]
#[diag("{$us} sanitizers are not supported for this target")]
pub(crate) struct SanitizersNotSupported {
    pub(crate) us: String,
}

#[derive(Diagnostic)]
#[diag("`-Zsanitizer={$first}` is incompatible with `-Zsanitizer={$second}`")]
pub(crate) struct CannotMixAndMatchSanitizers {
    pub(crate) first: String,
    pub(crate) second: String,
}

#[derive(Diagnostic)]
#[diag(
    "sanitizer is incompatible with statically linked libc, disable it using `-C target-feature=-crt-static`"
)]
pub(crate) struct CannotEnableCrtStaticLinux;

#[derive(Diagnostic)]
#[diag("`-Zsanitizer=cfi` requires `-Clto` or `-Clinker-plugin-lto`")]
pub(crate) struct SanitizerCfiRequiresLto;

#[derive(Diagnostic)]
#[diag("`-Zsanitizer=cfi` with `-Clto` requires `-Ccodegen-units=1`")]
pub(crate) struct SanitizerCfiRequiresSingleCodegenUnit;

#[derive(Diagnostic)]
#[diag("`-Zsanitizer-cfi-canonical-jump-tables` requires `-Zsanitizer=cfi`")]
pub(crate) struct SanitizerCfiCanonicalJumpTablesRequiresCfi;

#[derive(Diagnostic)]
#[diag("`-Zsanitizer-cfi-generalize-pointers` requires `-Zsanitizer=cfi` or `-Zsanitizer=kcfi`")]
pub(crate) struct SanitizerCfiGeneralizePointersRequiresCfi;

#[derive(Diagnostic)]
#[diag("`-Zsanitizer-cfi-normalize-integers` requires `-Zsanitizer=cfi` or `-Zsanitizer=kcfi`")]
pub(crate) struct SanitizerCfiNormalizeIntegersRequiresCfi;

#[derive(Diagnostic)]
#[diag("`-Zsanitizer-kcfi-arity` requires `-Zsanitizer=kcfi`")]
pub(crate) struct SanitizerKcfiArityRequiresKcfi;

#[derive(Diagnostic)]
#[diag("`-Z sanitizer=kcfi` requires `-C panic=abort`")]
pub(crate) struct SanitizerKcfiRequiresPanicAbort;

#[derive(Diagnostic)]
#[diag("`-Zsplit-lto-unit` requires `-Clto`, `-Clto=thin`, or `-Clinker-plugin-lto`")]
pub(crate) struct SplitLtoUnitRequiresLto;

#[derive(Diagnostic)]
#[diag("`-Zvirtual-function-elimination` requires `-Clto`")]
pub(crate) struct UnstableVirtualFunctionElimination;

#[derive(Diagnostic)]
#[diag("requested DWARF version {$dwarf_version} is not supported")]
#[help("supported DWARF versions are 2, 3, 4 and 5")]
pub(crate) struct UnsupportedDwarfVersion {
    pub(crate) dwarf_version: u32,
}

#[derive(Diagnostic)]
#[diag(
    "`-Zembed-source=y` requires at least `-Z dwarf-version=5` but DWARF version is {$dwarf_version}"
)]
pub(crate) struct EmbedSourceInsufficientDwarfVersion {
    pub(crate) dwarf_version: u32,
}

#[derive(Diagnostic)]
#[diag("`-Zembed-source=y` requires debug information to be enabled")]
pub(crate) struct EmbedSourceRequiresDebugInfo;

#[derive(Diagnostic)]
#[diag(
    "`-Z stack-protector={$stack_protector}` is not supported for target {$target_triple} and will be ignored"
)]
pub(crate) struct StackProtectorNotSupportedForTarget<'a> {
    pub(crate) stack_protector: StackProtector,
    pub(crate) target_triple: &'a TargetTuple,
}

#[derive(Diagnostic)]
#[diag(
    "`-Z small-data-threshold` is not supported for target {$target_triple} and will be ignored"
)]
pub(crate) struct SmallDataThresholdNotSupportedForTarget<'a> {
    pub(crate) target_triple: &'a TargetTuple,
}

#[derive(Diagnostic)]
#[diag("`-Zbranch-protection` is only supported on aarch64")]
pub(crate) struct BranchProtectionRequiresAArch64;

#[derive(Diagnostic)]
#[diag("`-Csplit-debuginfo={$debuginfo}` is unstable on this platform")]
pub(crate) struct SplitDebugInfoUnstablePlatform {
    pub(crate) debuginfo: SplitDebuginfo,
}

#[derive(Diagnostic)]
#[diag("output file {$file} is not writeable -- check its permissions")]
pub(crate) struct FileIsNotWriteable<'a> {
    pub(crate) file: &'a std::path::Path,
}

#[derive(Diagnostic)]
#[diag("failed to write `{$path}` due to error `{$err}`")]
pub(crate) struct FileWriteFail<'a> {
    pub(crate) path: &'a std::path::Path,
    pub(crate) err: String,
}

#[derive(Diagnostic)]
#[diag("crate name must not be empty")]
pub(crate) struct CrateNameEmpty {
    #[primary_span]
    pub(crate) span: Option<Span>,
}

#[derive(Diagnostic)]
#[diag("invalid character {$character} in crate name: `{$crate_name}`")]
pub(crate) struct InvalidCharacterInCrateName {
    #[primary_span]
    pub(crate) span: Option<Span>,
    pub(crate) character: char,
    pub(crate) crate_name: Symbol,
}

#[derive(Subdiagnostic)]
#[multipart_suggestion(
    "parentheses are required to parse this as an expression",
    applicability = "machine-applicable"
)]
pub struct ExprParenthesesNeeded {
    #[suggestion_part(code = "(")]
    left: Span,
    #[suggestion_part(code = ")")]
    right: Span,
}

impl ExprParenthesesNeeded {
    pub fn surrounding(s: Span) -> Self {
        ExprParenthesesNeeded { left: s.shrink_to_lo(), right: s.shrink_to_hi() }
    }
}

#[derive(Diagnostic)]
#[diag("skipping const checks")]
pub(crate) struct SkippingConstChecks {
    #[subdiagnostic]
    pub(crate) unleashed_features: Vec<UnleashedFeatureHelp>,
}

#[derive(Subdiagnostic)]
pub(crate) enum UnleashedFeatureHelp {
    #[help("skipping check for `{$gate}` feature")]
    Named {
        #[primary_span]
        span: Span,
        gate: Symbol,
    },
    #[help("skipping check that does not even have a feature gate")]
    Unnamed {
        #[primary_span]
        span: Span,
    },
}

#[derive(Diagnostic)]
#[diag("suffixes on {$kind} literals are invalid")]
struct InvalidLiteralSuffix<'a> {
    #[primary_span]
    #[label("invalid suffix `{$suffix}`")]
    span: Span,
    // FIXME(#100717)
    kind: &'a str,
    suffix: Symbol,
}

#[derive(Diagnostic)]
#[diag("invalid width `{$width}` for integer literal")]
#[help("valid widths are 8, 16, 32, 64 and 128")]
struct InvalidIntLiteralWidth {
    #[primary_span]
    span: Span,
    width: String,
}

#[derive(Diagnostic)]
#[diag("invalid base prefix for number literal")]
#[note("base prefixes (`0xff`, `0b1010`, `0o755`) are lowercase")]
struct InvalidNumLiteralBasePrefix {
    #[primary_span]
    #[suggestion(
        "try making the prefix lowercase",
        applicability = "maybe-incorrect",
        code = "{fixed}"
    )]
    span: Span,
    fixed: String,
}

#[derive(Diagnostic)]
#[diag("invalid suffix `{$suffix}` for number literal")]
#[help("the suffix must be one of the numeric types (`u32`, `isize`, `f32`, etc.)")]
struct InvalidNumLiteralSuffix {
    #[primary_span]
    #[label("invalid suffix `{$suffix}`")]
    span: Span,
    suffix: String,
}

#[derive(Diagnostic)]
#[diag("invalid width `{$width}` for float literal")]
#[help("valid widths are 32 and 64")]
struct InvalidFloatLiteralWidth {
    #[primary_span]
    span: Span,
    width: String,
}

#[derive(Diagnostic)]
#[diag("invalid suffix `{$suffix}` for float literal")]
#[help("valid suffixes are `f32` and `f64`")]
struct InvalidFloatLiteralSuffix {
    #[primary_span]
    #[label("invalid suffix `{$suffix}`")]
    span: Span,
    suffix: String,
}

#[derive(Diagnostic)]
#[diag("integer literal is too large")]
#[note("value exceeds limit of `{$limit}`")]
struct IntLiteralTooLarge {
    #[primary_span]
    span: Span,
    limit: String,
}

#[derive(Diagnostic)]
#[diag("hexadecimal float literal is not supported")]
struct HexadecimalFloatLiteralNotSupported {
    #[primary_span]
    #[label("not supported")]
    span: Span,
}

#[derive(Diagnostic)]
#[diag("octal float literal is not supported")]
struct OctalFloatLiteralNotSupported {
    #[primary_span]
    #[label("not supported")]
    span: Span,
}

#[derive(Diagnostic)]
#[diag("binary float literal is not supported")]
struct BinaryFloatLiteralNotSupported {
    #[primary_span]
    #[label("not supported")]
    span: Span,
}

pub fn report_lit_error(
    psess: &ParseSess,
    err: LitError,
    lit: token::Lit,
    span: Span,
) -> ErrorGuaranteed {
    create_lit_error(psess, err, lit, span).emit()
}

pub fn create_lit_error(psess: &ParseSess, err: LitError, lit: token::Lit, span: Span) -> Diag<'_> {
    // Checks if `s` looks like i32 or u1234 etc.
    fn looks_like_width_suffix(first_chars: &[char], s: &str) -> bool {
        s.len() > 1 && s.starts_with(first_chars) && s[1..].chars().all(|c| c.is_ascii_digit())
    }

    // Try to lowercase the prefix if the prefix and suffix are valid.
    fn fix_base_capitalisation(prefix: &str, suffix: &str) -> Option<String> {
        let mut chars = suffix.chars();

        let base_char = chars.next().unwrap();
        let base = match base_char {
            'B' => 2,
            'O' => 8,
            'X' => 16,
            _ => return None,
        };

        // check that the suffix contains only base-appropriate characters
        let valid = prefix == "0"
            && chars
                .filter(|c| *c != '_')
                .take_while(|c| *c != 'i' && *c != 'u')
                .all(|c| c.to_digit(base).is_some());

        valid.then(|| format!("0{}{}", base_char.to_ascii_lowercase(), &suffix[1..]))
    }

    let dcx = psess.dcx();
    match err {
        LitError::InvalidSuffix(suffix) => {
            dcx.create_err(InvalidLiteralSuffix { span, kind: lit.kind.descr(), suffix })
        }
        LitError::InvalidIntSuffix(suffix) => {
            let suf = suffix.as_str();
            if looks_like_width_suffix(&['i', 'u'], suf) {
                // If it looks like a width, try to be helpful.
                dcx.create_err(InvalidIntLiteralWidth { span, width: suf[1..].into() })
            } else if let Some(fixed) = fix_base_capitalisation(lit.symbol.as_str(), suf) {
                dcx.create_err(InvalidNumLiteralBasePrefix { span, fixed })
            } else {
                dcx.create_err(InvalidNumLiteralSuffix { span, suffix: suf.to_string() })
            }
        }
        LitError::InvalidFloatSuffix(suffix) => {
            let suf = suffix.as_str();
            if looks_like_width_suffix(&['f'], suf) {
                // If it looks like a width, try to be helpful.
                dcx.create_err(InvalidFloatLiteralWidth { span, width: suf[1..].to_string() })
            } else {
                dcx.create_err(InvalidFloatLiteralSuffix { span, suffix: suf.to_string() })
            }
        }
        LitError::NonDecimalFloat(base) => match base {
            16 => dcx.create_err(HexadecimalFloatLiteralNotSupported { span }),
            8 => dcx.create_err(OctalFloatLiteralNotSupported { span }),
            2 => dcx.create_err(BinaryFloatLiteralNotSupported { span }),
            _ => unreachable!(),
        },
        LitError::IntTooLarge(base) => {
            let max = u128::MAX;
            let limit = match base {
                2 => format!("{max:#b}"),
                8 => format!("{max:#o}"),
                16 => format!("{max:#x}"),
                _ => format!("{max}"),
            };
            dcx.create_err(IntLiteralTooLarge { span, limit })
        }
    }
}

#[derive(Diagnostic)]
#[diag("linker flavor `{$flavor}` is incompatible with the current target")]
#[note("compatible flavors are: {$compatible_list}")]
pub(crate) struct IncompatibleLinkerFlavor {
    pub(crate) flavor: &'static str,
    pub(crate) compatible_list: String,
}

#[derive(Diagnostic)]
#[diag("`-Zfunction-return` (except `keep`) is only supported on x86 and x86_64")]
pub(crate) struct FunctionReturnRequiresX86OrX8664;

#[derive(Diagnostic)]
#[diag("`-Zfunction-return=thunk-extern` is only supported on non-large code models")]
pub(crate) struct FunctionReturnThunkExternRequiresNonLargeCodeModel;

#[derive(Diagnostic)]
#[diag("`-Zindirect-branch-cs-prefix` is only supported on x86 and x86_64")]
pub(crate) struct IndirectBranchCsPrefixRequiresX86OrX8664;

#[derive(Diagnostic)]
#[diag("`-Zregparm={$regparm}` is unsupported (valid values 0-3)")]
pub(crate) struct UnsupportedRegparm {
    pub(crate) regparm: u32,
}

#[derive(Diagnostic)]
#[diag("`-Zregparm=N` is only supported on x86")]
pub(crate) struct UnsupportedRegparmArch;

#[derive(Diagnostic)]
#[diag("`-Zreg-struct-return` is only supported on x86")]
pub(crate) struct UnsupportedRegStructReturnArch;

#[derive(Diagnostic)]
#[diag("failed to create profiler: {$err}")]
pub(crate) struct FailedToCreateProfiler {
    pub(crate) err: String,
}

#[derive(Diagnostic)]
#[diag("`-Csoft-float` is ignored on this target; it only has an effect on *eabihf targets")]
#[note("this may become a hard error in a future version of Rust")]
pub(crate) struct SoftFloatIgnored;

#[derive(Diagnostic)]
#[diag("`-Csoft-float` is unsound and deprecated; use a corresponding *eabi target instead")]
#[note("it will be removed or ignored in a future version of Rust")]
#[note("see issue #129893 <https://github.com/rust-lang/rust/issues/129893> for more information")]
pub(crate) struct SoftFloatDeprecated;

#[derive(LintDiagnostic)]
#[diag("unexpected `--cfg {$cfg}` flag")]
#[note("config `{$cfg_name}` is only supposed to be controlled by `{$controlled_by}`")]
#[note("manually setting a built-in cfg can and does create incoherent behaviors")]
pub(crate) struct UnexpectedBuiltinCfg {
    pub(crate) cfg: String,
    pub(crate) cfg_name: Symbol,
    pub(crate) controlled_by: &'static str,
}

#[derive(Diagnostic)]
#[diag("ThinLTO is not supported by the codegen backend")]
pub(crate) struct ThinLtoNotSupportedByBackend;
