//! Code for dealing with `--print` requests.

use std::sync::LazyLock;

use rustc_data_structures::fx::FxHashSet;

use crate::EarlyDiagCtxt;
use crate::config::{
    CodegenOptions, OutFileName, UnstableOptions, nightly_options, split_out_file_name,
};

const PRINT_KINDS: &[(&str, PrintKind)] = &[
    // tidy-alphabetical-start
    ("all-target-specs-json", PrintKind::AllTargetSpecsJson),
    ("calling-conventions", PrintKind::CallingConventions),
    ("cfg", PrintKind::Cfg),
    ("check-cfg", PrintKind::CheckCfg),
    ("code-models", PrintKind::CodeModels),
    ("crate-name", PrintKind::CrateName),
    ("crate-root-lint-levels", PrintKind::CrateRootLintLevels),
    ("deployment-target", PrintKind::DeploymentTarget),
    ("file-names", PrintKind::FileNames),
    ("host-tuple", PrintKind::HostTuple),
    ("link-args", PrintKind::LinkArgs),
    ("native-static-libs", PrintKind::NativeStaticLibs),
    ("relocation-models", PrintKind::RelocationModels),
    ("split-debuginfo", PrintKind::SplitDebuginfo),
    ("stack-protector-strategies", PrintKind::StackProtectorStrategies),
    ("supported-crate-types", PrintKind::SupportedCrateTypes),
    ("sysroot", PrintKind::Sysroot),
    ("target-cpus", PrintKind::TargetCPUs),
    ("target-features", PrintKind::TargetFeatures),
    ("target-libdir", PrintKind::TargetLibdir),
    ("target-list", PrintKind::TargetList),
    ("target-spec-json", PrintKind::TargetSpecJson),
    ("target-spec-json-schema", PrintKind::TargetSpecJsonSchema),
    ("tls-models", PrintKind::TlsModels),
    // tidy-alphabetical-end
];

#[derive(Clone, PartialEq, Debug)]
pub struct PrintRequest {
    pub kind: PrintKind,
    pub out: OutFileName,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum PrintKind {
    // tidy-alphabetical-start
    AllTargetSpecsJson,
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

pub(crate) static PRINT_HELP: LazyLock<String> = LazyLock::new(|| {
    format!(
        "Compiler information to print on stdout (or to a file)\n\
        INFO may be one of <{}>.",
        PRINT_KINDS.iter().map(|(name, _)| format!("{name}")).collect::<Vec<_>>().join("|")
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

        let kind = if let Some((print_name, print_kind)) =
            PRINT_KINDS.iter().find(|&&(name, _)| name == req)
        {
            check_print_request_stability(early_dcx, unstable_opts, (print_name, *print_kind));
            *print_kind
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
    (print_name, print_kind): (&str, PrintKind),
) {
    if !is_print_request_stable(print_kind) && !unstable_opts.unstable_options {
        early_dcx.early_fatal(format!(
            "the `-Z unstable-options` flag must also be passed to enable the `{print_name}` \
                print option"
        ));
    }
}

fn is_print_request_stable(print_kind: PrintKind) -> bool {
    match print_kind {
        PrintKind::AllTargetSpecsJson
        | PrintKind::CheckCfg
        | PrintKind::CrateRootLintLevels
        | PrintKind::SupportedCrateTypes
        | PrintKind::TargetSpecJson
        | PrintKind::TargetSpecJsonSchema => false,
        _ => true,
    }
}

fn emit_unknown_print_request_help(early_dcx: &EarlyDiagCtxt, req: &str, is_nightly: bool) -> ! {
    let prints = PRINT_KINDS
        .iter()
        .filter_map(|(name, kind)| {
            // If we're not on nightly, we don't want to print unstable options
            if !is_nightly && !is_print_request_stable(*kind) {
                None
            } else {
                Some(format!("`{name}`"))
            }
        })
        .collect::<Vec<_>>();
    let prints = prints.join(", ");

    let mut diag = early_dcx.early_struct_fatal(format!("unknown print request: `{req}`"));
    #[allow(rustc::diagnostic_outside_of_impl)]
    diag.help(format!("valid print requests are: {prints}"));

    if req == "lints" {
        diag.help(format!("use `-Whelp` to print a list of lints"));
    }

    diag.help(format!("for more information, see the rustc book: https://doc.rust-lang.org/rustc/command-line-arguments.html#--print-print-compiler-information"));
    diag.emit()
}
