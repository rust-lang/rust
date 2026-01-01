//! Code for dealing with `--print` requests.

use std::fmt;
use std::sync::LazyLock;

use rustc_data_structures::fx::FxHashSet;

use crate::EarlyDiagCtxt;
use crate::config::{
    CodegenOptions, OutFileName, UnstableOptions, nightly_options, split_out_file_name,
};
use crate::macros::AllVariants;

#[derive(Clone, PartialEq, Debug)]
pub struct PrintRequest {
    pub kind: PrintKind,
    pub out: OutFileName,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[derive(AllVariants)]
pub enum PrintKind {
    // tidy-alphabetical-start
    AllTargetSpecsJson,
    BackendHasZstd,
    CallingConventions,
    Cfg,
    CheckCfg,
    CodeModels,
    CrateName,
    CrateRootLintLevels,
    DeploymentTarget,
    FileNames,
    HostTuple,
    LinkArgs,
    NativeStaticLibs,
    RelocationModels,
    SplitDebuginfo,
    StackProtectorStrategies,
    SupportedCrateTypes,
    Sysroot,
    TargetCPUs,
    TargetFeatures,
    TargetLibdir,
    TargetList,
    TargetSpecJson,
    TargetSpecJsonSchema,
    TlsModels,
    // tidy-alphabetical-end
}

impl PrintKind {
    /// FIXME: rust-analyzer doesn't support `#![feature(macro_derive)]` yet
    /// (<https://github.com/rust-lang/rust-analyzer/issues/21043>), which breaks autocomplete.
    /// Work around that by aliasing the trait constant to a regular constant.
    const ALL_VARIANTS: &[Self] = <Self as AllVariants>::ALL_VARIANTS;

    fn name(self) -> &'static str {
        use PrintKind::*;
        match self {
            // tidy-alphabetical-start
            AllTargetSpecsJson => "all-target-specs-json",
            BackendHasZstd => "backend-has-zstd",
            CallingConventions => "calling-conventions",
            Cfg => "cfg",
            CheckCfg => "check-cfg",
            CodeModels => "code-models",
            CrateName => "crate-name",
            CrateRootLintLevels => "crate-root-lint-levels",
            DeploymentTarget => "deployment-target",
            FileNames => "file-names",
            HostTuple => "host-tuple",
            LinkArgs => "link-args",
            NativeStaticLibs => "native-static-libs",
            RelocationModels => "relocation-models",
            SplitDebuginfo => "split-debuginfo",
            StackProtectorStrategies => "stack-protector-strategies",
            SupportedCrateTypes => "supported-crate-types",
            Sysroot => "sysroot",
            TargetCPUs => "target-cpus",
            TargetFeatures => "target-features",
            TargetLibdir => "target-libdir",
            TargetList => "target-list",
            TargetSpecJson => "target-spec-json",
            TargetSpecJsonSchema => "target-spec-json-schema",
            TlsModels => "tls-models",
            // tidy-alphabetical-end
        }
    }

    fn is_stable(self) -> bool {
        use PrintKind::*;
        match self {
            // Stable values:
            CallingConventions
            | Cfg
            | CodeModels
            | CrateName
            | DeploymentTarget
            | FileNames
            | HostTuple
            | LinkArgs
            | NativeStaticLibs
            | RelocationModels
            | SplitDebuginfo
            | StackProtectorStrategies
            | Sysroot
            | TargetCPUs
            | TargetFeatures
            | TargetLibdir
            | TargetList
            | TlsModels => true,

            // Unstable values:
            AllTargetSpecsJson => false,
            BackendHasZstd => false, // (perma-unstable, for use by compiletest)
            CheckCfg => false,
            CrateRootLintLevels => false,
            SupportedCrateTypes => false,
            TargetSpecJson => false,
            TargetSpecJsonSchema => false,
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        Self::ALL_VARIANTS.iter().find(|kind| kind.name() == s).copied()
    }
}

impl fmt::Display for PrintKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.name().fmt(f)
    }
}

pub(crate) static PRINT_HELP: LazyLock<String> = LazyLock::new(|| {
    let print_kinds =
        PrintKind::ALL_VARIANTS.iter().map(|kind| kind.name()).collect::<Vec<_>>().join("|");
    format!(
        "Compiler information to print on stdout (or to a file)\n\
        INFO may be one of <{print_kinds}>.",
    )
});

pub(crate) fn collect_print_requests(
    early_dcx: &EarlyDiagCtxt,
    cg: &mut CodegenOptions,
    unstable_opts: &UnstableOptions,
    matches: &getopts::Matches,
) -> Vec<PrintRequest> {
    let mut prints = Vec::<PrintRequest>::new();
    if cg.target_cpu.as_deref() == Some("help") {
        prints.push(PrintRequest { kind: PrintKind::TargetCPUs, out: OutFileName::Stdout });
        cg.target_cpu = None;
    };
    if cg.target_feature == "help" {
        prints.push(PrintRequest { kind: PrintKind::TargetFeatures, out: OutFileName::Stdout });
        cg.target_feature = String::new();
    }

    // We disallow reusing the same path in multiple prints, such as `--print
    // cfg=output.txt --print link-args=output.txt`, because outputs are printed
    // by disparate pieces of the compiler, and keeping track of which files
    // need to be overwritten vs appended to is annoying.
    let mut printed_paths = FxHashSet::default();

    prints.extend(matches.opt_strs("print").into_iter().map(|req| {
        let (req, out) = split_out_file_name(&req);

        let kind = if let Some(print_kind) = PrintKind::from_str(req) {
            check_print_request_stability(early_dcx, unstable_opts, print_kind);
            print_kind
        } else {
            let is_nightly = nightly_options::match_is_nightly_build(matches);
            emit_unknown_print_request_help(early_dcx, req, is_nightly)
        };

        let out = out.unwrap_or(OutFileName::Stdout);
        if let OutFileName::Real(path) = &out {
            if !printed_paths.insert(path.clone()) {
                early_dcx.early_fatal(format!(
                    "cannot print multiple outputs to the same path: {}",
                    path.display(),
                ));
            }
        }

        PrintRequest { kind, out }
    }));

    prints
}

fn check_print_request_stability(
    early_dcx: &EarlyDiagCtxt,
    unstable_opts: &UnstableOptions,
    print_kind: PrintKind,
) {
    if !print_kind.is_stable() && !unstable_opts.unstable_options {
        early_dcx.early_fatal(format!(
            "the `-Z unstable-options` flag must also be passed to enable the `{print_kind}` print option"
        ));
    }
}

fn emit_unknown_print_request_help(early_dcx: &EarlyDiagCtxt, req: &str, is_nightly: bool) -> ! {
    let prints = PrintKind::ALL_VARIANTS
        .iter()
        // If we're not on nightly, we don't want to print unstable options
        .filter(|kind| is_nightly || kind.is_stable())
        .map(|kind| format!("`{kind}`"))
        .collect::<Vec<_>>()
        .join(", ");

    let mut diag = early_dcx.early_struct_fatal(format!("unknown print request: `{req}`"));
    #[allow(rustc::diagnostic_outside_of_impl)]
    diag.help(format!("valid print requests are: {prints}"));

    if req == "lints" {
        diag.help(format!("use `-Whelp` to print a list of lints"));
    }

    diag.help(format!("for more information, see the rustc book: https://doc.rust-lang.org/rustc/command-line-arguments.html#--print-print-compiler-information"));
    diag.emit()
}
