//! Code for dealing with `--print` requests.

use std::str::FromStr;
use std::sync::LazyLock;

use rustc_data_structures::fx::FxHashSet;
use rustc_data_structures::string_enum;

use crate::EarlyDiagCtxt;
use crate::config::{
    CodegenOptions, OutFileName, UnstableOptions, build_unknown_arg_value_diag, nightly_options,
    split_out_file_name,
};

#[derive(Clone, PartialEq, Debug)]
pub struct PrintRequest {
    pub kind: PrintKind,
    pub out: OutFileName,
    pub arg: Option<String>,
}

string_enum! {
    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub enum PrintKind {
        // tidy-alphabetical-start
        AllTargetSpecsJson => "all-target-specs-json",
        BackendHasMnemonic => "backend-has-mnemonic",
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

impl PrintKind {
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
            BackendHasMnemonic => false, // (perma-unstable, for use by compiletest)
            BackendHasZstd => false,     // (perma-unstable, for use by compiletest)
            CheckCfg => false,
            CrateRootLintLevels => false,
            SupportedCrateTypes => false,
            TargetSpecJson => false,
            TargetSpecJsonSchema => false,
        }
    }
}

pub(crate) static PRINT_HELP: LazyLock<String> = LazyLock::new(|| {
    let print_kinds: String =
        PrintKind::VARIANTS.iter().map(PrintKind::to_str).intersperse("|").collect();
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
        prints.push(PrintRequest {
            kind: PrintKind::TargetCPUs,
            out: OutFileName::Stdout,
            arg: None,
        });
        cg.target_cpu = None;
    };
    if cg.target_feature == "help" {
        prints.push(PrintRequest {
            kind: PrintKind::TargetFeatures,
            out: OutFileName::Stdout,
            arg: None,
        });
        cg.target_feature = String::new();
    }

    // We disallow reusing the same path in multiple prints, such as `--print
    // cfg=output.txt --print link-args=output.txt`, because outputs are printed
    // by disparate pieces of the compiler, and keeping track of which files
    // need to be overwritten vs appended to is annoying.
    let mut printed_paths = FxHashSet::default();

    prints.extend(matches.opt_strs("print").into_iter().map(|req| {
        let (req, out) = split_out_file_name(&req);

        let (kind, arg) = if let Some(mnemonic) = req.strip_prefix("backend-has-mnemonic") {
            check_print_request_stability(early_dcx, unstable_opts, PrintKind::BackendHasMnemonic);
            // BackendHasMnemonic requires a mnemonic argument
            if let Some(mnemonic) = mnemonic.strip_prefix(':')
                && !mnemonic.is_empty()
            {
                (PrintKind::BackendHasMnemonic, Some(mnemonic.to_string()))
            } else {
                early_dcx.early_fatal(
                    "expected mnemonic name after `--print=backend-has-mnemonic:`, \
                    for example: `--print=backend-has-mnemonic:RET`",
                );
            }
        } else if let Ok(print_kind) = PrintKind::from_str(req) {
            check_print_request_stability(early_dcx, unstable_opts, print_kind);
            (print_kind, None)
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

        PrintRequest { kind, out, arg }
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
    let valid: Vec<&str> = PrintKind::VARIANTS
        .iter()
        .filter(|kind| is_nightly || kind.is_stable())
        .map(|kind| kind.to_str())
        .collect();

    let mut diag = build_unknown_arg_value_diag(early_dcx, "print request", req, &valid);

    if req == "lints" {
        diag.help("use `-Whelp` to print a list of lints");
    }

    diag.help(
        "for more information, see the rustc book: \
         https://doc.rust-lang.org/rustc/command-line-arguments.html#--print-print-compiler-information",
    );
    diag.emit()
}
