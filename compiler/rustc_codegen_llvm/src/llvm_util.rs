use std::collections::VecDeque;
use std::ffi::{CStr, CString};
use std::fmt::Write;
use std::path::Path;
use std::sync::Once;
use std::{ptr, slice, str};

use libc::c_int;
use rustc_codegen_ssa::base::wants_wasm_eh;
use rustc_codegen_ssa::codegen_attrs::check_tied_features;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_data_structures::unord::UnordSet;
use rustc_fs_util::path_to_c_string;
use rustc_middle::bug;
use rustc_session::Session;
use rustc_session::config::{PrintKind, PrintRequest};
use rustc_span::symbol::Symbol;
use rustc_target::spec::{MergeFunctions, PanicStrategy, SmallDataThresholdSupport};
use rustc_target::target_features::{RUSTC_SPECIAL_FEATURES, RUSTC_SPECIFIC_FEATURES, Stability};

use crate::back::write::create_informational_target_machine;
use crate::errors::{
    FixedX18InvalidArch, ForbiddenCTargetFeature, InvalidTargetFeaturePrefix, PossibleFeature,
    UnknownCTargetFeature, UnknownCTargetFeaturePrefix, UnstableCTargetFeature,
};
use crate::llvm;

static INIT: Once = Once::new();

pub(crate) fn init(sess: &Session) {
    unsafe {
        // Before we touch LLVM, make sure that multithreading is enabled.
        if llvm::LLVMIsMultithreaded() != 1 {
            bug!("LLVM compiled without support for threads");
        }
        INIT.call_once(|| {
            configure_llvm(sess);
        });
    }
}

fn require_inited() {
    if !INIT.is_completed() {
        bug!("LLVM is not initialized");
    }
}

unsafe fn configure_llvm(sess: &Session) {
    let n_args = sess.opts.cg.llvm_args.len() + sess.target.llvm_args.len();
    let mut llvm_c_strs = Vec::with_capacity(n_args + 1);
    let mut llvm_args = Vec::with_capacity(n_args + 1);

    unsafe {
        llvm::LLVMRustInstallErrorHandlers();
    }
    // On Windows, an LLVM assertion will open an Abort/Retry/Ignore dialog
    // box for the purpose of launching a debugger. However, on CI this will
    // cause it to hang until it times out, which can take several hours.
    if std::env::var_os("CI").is_some() {
        unsafe {
            llvm::LLVMRustDisableSystemDialogsOnCrash();
        }
    }

    fn llvm_arg_to_arg_name(full_arg: &str) -> &str {
        full_arg.trim().split(|c: char| c == '=' || c.is_whitespace()).next().unwrap_or("")
    }

    let cg_opts = sess.opts.cg.llvm_args.iter().map(AsRef::as_ref);
    let tg_opts = sess.target.llvm_args.iter().map(AsRef::as_ref);
    let sess_args = cg_opts.chain(tg_opts);

    let user_specified_args: FxHashSet<_> =
        sess_args.clone().map(|s| llvm_arg_to_arg_name(s)).filter(|s| !s.is_empty()).collect();

    {
        // This adds the given argument to LLVM. Unless `force` is true
        // user specified arguments are *not* overridden.
        let mut add = |arg: &str, force: bool| {
            if force || !user_specified_args.contains(llvm_arg_to_arg_name(arg)) {
                let s = CString::new(arg).unwrap();
                llvm_args.push(s.as_ptr());
                llvm_c_strs.push(s);
            }
        };
        // Set the llvm "program name" to make usage and invalid argument messages more clear.
        add("rustc -Cllvm-args=\"...\" with", true);
        if sess.opts.unstable_opts.time_llvm_passes {
            add("-time-passes", false);
        }
        if sess.opts.unstable_opts.print_llvm_passes {
            add("-debug-pass=Structure", false);
        }
        if sess.target.generate_arange_section
            && !sess.opts.unstable_opts.no_generate_arange_section
        {
            add("-generate-arange-section", false);
        }

        match sess.opts.unstable_opts.merge_functions.unwrap_or(sess.target.merge_functions) {
            MergeFunctions::Disabled | MergeFunctions::Trampolines => {}
            MergeFunctions::Aliases => {
                add("-mergefunc-use-aliases", false);
            }
        }

        if wants_wasm_eh(sess) {
            add("-wasm-enable-eh", false);
        }

        if sess.target.os == "emscripten" && sess.panic_strategy() == PanicStrategy::Unwind {
            add("-enable-emscripten-cxx-exceptions", false);
        }

        // HACK(eddyb) LLVM inserts `llvm.assume` calls to preserve align attributes
        // during inlining. Unfortunately these may block other optimizations.
        add("-preserve-alignment-assumptions-during-inlining=false", false);

        // Use non-zero `import-instr-limit` multiplier for cold callsites.
        add("-import-cold-multiplier=0.1", false);

        if sess.print_llvm_stats() {
            add("-stats", false);
        }

        for arg in sess_args {
            add(&(*arg), true);
        }

        match (
            sess.opts.unstable_opts.small_data_threshold,
            sess.target.small_data_threshold_support(),
        ) {
            // Set up the small-data optimization limit for architectures that use
            // an LLVM argument to control this.
            (Some(threshold), SmallDataThresholdSupport::LlvmArg(arg)) => {
                add(&format!("--{arg}={threshold}"), false)
            }
            _ => (),
        };
    }

    if sess.opts.unstable_opts.llvm_time_trace {
        unsafe { llvm::LLVMRustTimeTraceProfilerInitialize() };
    }

    rustc_llvm::initialize_available_targets();

    unsafe { llvm::LLVMRustSetLLVMOptions(llvm_args.len() as c_int, llvm_args.as_ptr()) };
}

pub(crate) fn time_trace_profiler_finish(file_name: &Path) {
    unsafe {
        let file_name = path_to_c_string(file_name);
        llvm::LLVMRustTimeTraceProfilerFinish(file_name.as_ptr());
    }
}

enum TargetFeatureFoldStrength<'a> {
    // The feature is only tied when enabling the feature, disabling
    // this feature shouldn't disable the tied feature.
    EnableOnly(&'a str),
    // The feature is tied for both enabling and disabling this feature.
    Both(&'a str),
}

impl<'a> TargetFeatureFoldStrength<'a> {
    fn as_str(&self) -> &'a str {
        match self {
            TargetFeatureFoldStrength::EnableOnly(feat) => feat,
            TargetFeatureFoldStrength::Both(feat) => feat,
        }
    }
}

pub(crate) struct LLVMFeature<'a> {
    llvm_feature_name: &'a str,
    dependency: Option<TargetFeatureFoldStrength<'a>>,
}

impl<'a> LLVMFeature<'a> {
    fn new(llvm_feature_name: &'a str) -> Self {
        Self { llvm_feature_name, dependency: None }
    }

    fn with_dependency(
        llvm_feature_name: &'a str,
        dependency: TargetFeatureFoldStrength<'a>,
    ) -> Self {
        Self { llvm_feature_name, dependency: Some(dependency) }
    }

    fn contains(&self, feat: &str) -> bool {
        self.iter().any(|dep| dep == feat)
    }

    fn iter(&'a self) -> impl Iterator<Item = &'a str> {
        let dependencies = self.dependency.iter().map(|feat| feat.as_str());
        std::iter::once(self.llvm_feature_name).chain(dependencies)
    }
}

impl<'a> IntoIterator for LLVMFeature<'a> {
    type Item = &'a str;
    type IntoIter = impl Iterator<Item = &'a str>;

    fn into_iter(self) -> Self::IntoIter {
        let dependencies = self.dependency.into_iter().map(|feat| feat.as_str());
        std::iter::once(self.llvm_feature_name).chain(dependencies)
    }
}

// WARNING: the features after applying `to_llvm_features` must be known
// to LLVM or the feature detection code will walk past the end of the feature
// array, leading to crashes.
//
// To find a list of LLVM's names, see llvm-project/llvm/lib/Target/{ARCH}/*.td
// where `{ARCH}` is the architecture name. Look for instances of `SubtargetFeature`.
//
// Check the current rustc fork of LLVM in the repo at https://github.com/rust-lang/llvm-project/.
// The commit in use can be found via the `llvm-project` submodule in
// https://github.com/rust-lang/rust/tree/master/src Though note that Rust can also be build with
// an external precompiled version of LLVM which might lead to failures if the oldest tested /
// supported LLVM version doesn't yet support the relevant intrinsics.
pub(crate) fn to_llvm_features<'a>(sess: &Session, s: &'a str) -> Option<LLVMFeature<'a>> {
    let arch = if sess.target.arch == "x86_64" {
        "x86"
    } else if sess.target.arch == "arm64ec" {
        "aarch64"
    } else if sess.target.arch == "sparc64" {
        "sparc"
    } else {
        &*sess.target.arch
    };
    match (arch, s) {
        ("x86", "sse4.2") => Some(LLVMFeature::with_dependency(
            "sse4.2",
            TargetFeatureFoldStrength::EnableOnly("crc32"),
        )),
        ("x86", "pclmulqdq") => Some(LLVMFeature::new("pclmul")),
        ("x86", "rdrand") => Some(LLVMFeature::new("rdrnd")),
        ("x86", "bmi1") => Some(LLVMFeature::new("bmi")),
        ("x86", "cmpxchg16b") => Some(LLVMFeature::new("cx16")),
        ("x86", "lahfsahf") => Some(LLVMFeature::new("sahf")),
        ("aarch64", "rcpc2") => Some(LLVMFeature::new("rcpc-immo")),
        ("aarch64", "dpb") => Some(LLVMFeature::new("ccpp")),
        ("aarch64", "dpb2") => Some(LLVMFeature::new("ccdp")),
        ("aarch64", "frintts") => Some(LLVMFeature::new("fptoint")),
        ("aarch64", "fcma") => Some(LLVMFeature::new("complxnum")),
        ("aarch64", "pmuv3") => Some(LLVMFeature::new("perfmon")),
        ("aarch64", "paca") => Some(LLVMFeature::new("pauth")),
        ("aarch64", "pacg") => Some(LLVMFeature::new("pauth")),
        ("aarch64", "pauth-lr") if get_version().0 < 19 => None,
        // Before LLVM 20 those two features were packaged together as b16b16
        ("aarch64", "sve-b16b16") if get_version().0 < 20 => Some(LLVMFeature::new("b16b16")),
        ("aarch64", "sme-b16b16") if get_version().0 < 20 => Some(LLVMFeature::new("b16b16")),
        ("aarch64", "flagm2") => Some(LLVMFeature::new("altnzcv")),
        // Rust ties fp and neon together.
        ("aarch64", "neon") => {
            Some(LLVMFeature::with_dependency("neon", TargetFeatureFoldStrength::Both("fp-armv8")))
        }
        // In LLVM neon implicitly enables fp, but we manually enable
        // neon when a feature only implicitly enables fp
        ("aarch64", "fhm") => Some(LLVMFeature::new("fp16fml")),
        ("aarch64", "fp16") => Some(LLVMFeature::new("fullfp16")),
        // Filter out features that are not supported by the current LLVM version
        ("aarch64", "fpmr") if get_version().0 != 18 => None,
        // In LLVM 18, `unaligned-scalar-mem` was merged with `unaligned-vector-mem` into a single
        // feature called `fast-unaligned-access`. In LLVM 19, it was split back out.
        ("riscv32" | "riscv64", "unaligned-scalar-mem") if get_version().0 == 18 => {
            Some(LLVMFeature::new("fast-unaligned-access"))
        }
        // Filter out features that are not supported by the current LLVM version
        ("riscv32" | "riscv64", "zaamo") if get_version().0 < 19 => None,
        ("riscv32" | "riscv64", "zabha") if get_version().0 < 19 => None,
        ("riscv32" | "riscv64", "zalrsc") if get_version().0 < 19 => None,
        // Enable the evex512 target feature if an avx512 target feature is enabled.
        ("x86", s) if s.starts_with("avx512") => {
            Some(LLVMFeature::with_dependency(s, TargetFeatureFoldStrength::EnableOnly("evex512")))
        }
        // Support for `wide-arithmetic` will first land in LLVM 20 as part of
        // llvm/llvm-project#111598
        ("wasm32" | "wasm64", "wide-arithmetic") if get_version() < (20, 0, 0) => None,
        ("sparc", "leoncasa") => Some(LLVMFeature::new("hasleoncasa")),
        // In LLVM 19, there is no `v8plus` feature and `v9` means "SPARC-V9 instruction available and SPARC-V8+ ABI used".
        // https://github.com/llvm/llvm-project/blob/llvmorg-19.1.0/llvm/lib/Target/Sparc/MCTargetDesc/SparcELFObjectWriter.cpp#L27-L28
        // Before LLVM 19, there is no `v8plus` feature and `v9` means "SPARC-V9 instruction available".
        // https://github.com/llvm/llvm-project/blob/llvmorg-18.1.0/llvm/lib/Target/Sparc/MCTargetDesc/SparcELFObjectWriter.cpp#L26
        ("sparc", "v8plus") if get_version().0 == 19 => Some(LLVMFeature::new("v9")),
        ("sparc", "v8plus") if get_version().0 < 19 => None,
        (_, s) => Some(LLVMFeature::new(s)),
    }
}

/// Used to generate cfg variables and apply features.
/// Must express features in the way Rust understands them.
///
/// We do not have to worry about RUSTC_SPECIFIC_FEATURES here, those are handled outside codegen.
pub fn target_features(sess: &Session, allow_unstable: bool) -> Vec<Symbol> {
    let mut features: FxHashSet<Symbol> = Default::default();

    // Add base features for the target.
    // We do *not* add the -Ctarget-features there, and instead duplicate the logic for that below.
    // The reason is that if LLVM considers a feature implied but we do not, we don't want that to
    // show up in `cfg`. That way, `cfg` is entirely under our control -- except for the handling of
    // the target CPU, that is still expanded to target features (with all their implied features) by
    // LLVM.
    let target_machine = create_informational_target_machine(sess, true);
    // Compute which of the known target features are enabled in the 'base' target machine.
    // We only consider "supported" features; "forbidden" features are not reflected in `cfg` as of now.
    features.extend(
        sess.target
            .rust_target_features()
            .iter()
            .filter(|(_, gate, _)| gate.is_supported())
            .filter(|(feature, _, _)| {
                // skip checking special features, as LLVM may not understand them
                if RUSTC_SPECIAL_FEATURES.contains(feature) {
                    return true;
                }
                // check that all features in a given smallvec are enabled
                if let Some(feat) = to_llvm_features(sess, feature) {
                    for llvm_feature in feat {
                        let cstr = SmallCStr::new(llvm_feature);
                        if !unsafe { llvm::LLVMRustHasFeature(&target_machine, cstr.as_ptr()) } {
                            return false;
                        }
                    }
                    true
                } else {
                    false
                }
            })
            .map(|(feature, _, _)| Symbol::intern(feature)),
    );

    // Add enabled features
    for (enabled, feature) in
        sess.opts.cg.target_feature.split(',').filter_map(|s| match s.chars().next() {
            Some('+') => Some((true, Symbol::intern(&s[1..]))),
            Some('-') => Some((false, Symbol::intern(&s[1..]))),
            _ => None,
        })
    {
        if enabled {
            // Also add all transitively implied features.
            features.extend(sess.target.implied_target_features(std::iter::once(feature)));
        } else {
            // Remove transitively reverse-implied features.

            // We don't care about the order in `features` since the only thing we use it for is the
            // `features.contains` below.
            #[allow(rustc::potential_query_instability)]
            features.retain(|f| {
                if sess.target.implied_target_features(std::iter::once(*f)).contains(&feature) {
                    // If `f` if implies `feature`, then `!feature` implies `!f`, so we have to
                    // remove `f`. (This is the standard logical contraposition principle.)
                    false
                } else {
                    // We can keep `f`.
                    true
                }
            });
        }
    }

    // Filter enabled features based on feature gates
    sess.target
        .rust_target_features()
        .iter()
        .filter(|(_, gate, _)| gate.is_supported())
        .filter_map(|&(feature, gate, _)| {
            if sess.is_nightly_build() || allow_unstable || gate.is_stable() {
                Some(feature)
            } else {
                None
            }
        })
        .filter(|feature| features.contains(&Symbol::intern(feature)))
        .map(|feature| Symbol::intern(feature))
        .collect()
}

pub(crate) fn print_version() {
    let (major, minor, patch) = get_version();
    println!("LLVM version: {major}.{minor}.{patch}");
}

pub(crate) fn get_version() -> (u32, u32, u32) {
    // Can be called without initializing LLVM
    unsafe {
        (llvm::LLVMRustVersionMajor(), llvm::LLVMRustVersionMinor(), llvm::LLVMRustVersionPatch())
    }
}

pub(crate) fn print_passes() {
    // Can be called without initializing LLVM
    unsafe {
        llvm::LLVMRustPrintPasses();
    }
}

fn llvm_target_features(tm: &llvm::TargetMachine) -> Vec<(&str, &str)> {
    let len = unsafe { llvm::LLVMRustGetTargetFeaturesCount(tm) };
    let mut ret = Vec::with_capacity(len);
    for i in 0..len {
        unsafe {
            let mut feature = ptr::null();
            let mut desc = ptr::null();
            llvm::LLVMRustGetTargetFeature(tm, i, &mut feature, &mut desc);
            if feature.is_null() || desc.is_null() {
                bug!("LLVM returned a `null` target feature string");
            }
            let feature = CStr::from_ptr(feature).to_str().unwrap_or_else(|e| {
                bug!("LLVM returned a non-utf8 feature string: {}", e);
            });
            let desc = CStr::from_ptr(desc).to_str().unwrap_or_else(|e| {
                bug!("LLVM returned a non-utf8 feature string: {}", e);
            });
            ret.push((feature, desc));
        }
    }
    ret
}

pub(crate) fn print(req: &PrintRequest, out: &mut String, sess: &Session) {
    require_inited();
    let tm = create_informational_target_machine(sess, false);
    match req.kind {
        PrintKind::TargetCPUs => print_target_cpus(sess, &tm, out),
        PrintKind::TargetFeatures => print_target_features(sess, &tm, out),
        _ => bug!("rustc_codegen_llvm can't handle print request: {:?}", req),
    }
}

fn print_target_cpus(sess: &Session, tm: &llvm::TargetMachine, out: &mut String) {
    let cpu_names = llvm::build_string(|s| unsafe {
        llvm::LLVMRustPrintTargetCPUs(&tm, s);
    })
    .unwrap();

    struct Cpu<'a> {
        cpu_name: &'a str,
        remark: String,
    }
    // Compare CPU against current target to label the default.
    let target_cpu = handle_native(&sess.target.cpu);
    let make_remark = |cpu_name| {
        if cpu_name == target_cpu {
            // FIXME(#132514): This prints the LLVM target string, which can be
            // different from the Rust target string. Is that intended?
            let target = &sess.target.llvm_target;
            format!(
                " - This is the default target CPU for the current build target (currently {target})."
            )
        } else {
            "".to_owned()
        }
    };
    let mut cpus = cpu_names
        .lines()
        .map(|cpu_name| Cpu { cpu_name, remark: make_remark(cpu_name) })
        .collect::<VecDeque<_>>();

    // Only print the "native" entry when host and target are the same arch,
    // since otherwise it could be wrong or misleading.
    if sess.host.arch == sess.target.arch {
        let host = get_host_cpu_name();
        cpus.push_front(Cpu {
            cpu_name: "native",
            remark: format!(" - Select the CPU of the current host (currently {host})."),
        });
    }

    let max_name_width = cpus.iter().map(|cpu| cpu.cpu_name.len()).max().unwrap_or(0);
    writeln!(out, "Available CPUs for this target:").unwrap();
    for Cpu { cpu_name, remark } in cpus {
        // Only pad the CPU name if there's a remark to print after it.
        let width = if remark.is_empty() { 0 } else { max_name_width };
        writeln!(out, "    {cpu_name:<width$}{remark}").unwrap();
    }
}

fn print_target_features(sess: &Session, tm: &llvm::TargetMachine, out: &mut String) {
    let mut llvm_target_features = llvm_target_features(tm);
    let mut known_llvm_target_features = FxHashSet::<&'static str>::default();
    let mut rustc_target_features = sess
        .target
        .rust_target_features()
        .iter()
        .filter_map(|(feature, gate, _implied)| {
            if !gate.is_supported() {
                // Only list (experimentally) supported features.
                return None;
            }
            // LLVM asserts that these are sorted. LLVM and Rust both use byte comparison for these
            // strings.
            let llvm_feature = to_llvm_features(sess, *feature)?.llvm_feature_name;
            let desc =
                match llvm_target_features.binary_search_by_key(&llvm_feature, |(f, _d)| f).ok() {
                    Some(index) => {
                        known_llvm_target_features.insert(llvm_feature);
                        llvm_target_features[index].1
                    }
                    None => "",
                };

            Some((*feature, desc))
        })
        .collect::<Vec<_>>();

    // Since we add this at the end ...
    rustc_target_features.extend_from_slice(&[(
        "crt-static",
        "Enables C Run-time Libraries to be statically linked",
    )]);
    // ... we need to sort the list again.
    rustc_target_features.sort();

    llvm_target_features.retain(|(f, _d)| !known_llvm_target_features.contains(f));

    let max_feature_len = llvm_target_features
        .iter()
        .chain(rustc_target_features.iter())
        .map(|(feature, _desc)| feature.len())
        .max()
        .unwrap_or(0);

    writeln!(out, "Features supported by rustc for this target:").unwrap();
    for (feature, desc) in &rustc_target_features {
        writeln!(out, "    {feature:max_feature_len$} - {desc}.").unwrap();
    }
    writeln!(out, "\nCode-generation features supported by LLVM for this target:").unwrap();
    for (feature, desc) in &llvm_target_features {
        writeln!(out, "    {feature:max_feature_len$} - {desc}.").unwrap();
    }
    if llvm_target_features.is_empty() {
        writeln!(out, "    Target features listing is not supported by this LLVM version.")
            .unwrap();
    }
    writeln!(out, "\nUse +feature to enable a feature, or -feature to disable it.").unwrap();
    writeln!(out, "For example, rustc -C target-cpu=mycpu -C target-feature=+feature1,-feature2\n")
        .unwrap();
    writeln!(out, "Code-generation features cannot be used in cfg or #[target_feature],").unwrap();
    writeln!(out, "and may be renamed or removed in a future version of LLVM or rustc.\n").unwrap();
}

/// Returns the host CPU name, according to LLVM.
fn get_host_cpu_name() -> &'static str {
    let mut len = 0;
    // SAFETY: The underlying C++ global function returns a `StringRef` that
    // isn't tied to any particular backing buffer, so it must be 'static.
    let slice: &'static [u8] = unsafe {
        let ptr = llvm::LLVMRustGetHostCPUName(&mut len);
        assert!(!ptr.is_null());
        slice::from_raw_parts(ptr, len)
    };
    str::from_utf8(slice).expect("host CPU name should be UTF-8")
}

/// If the given string is `"native"`, returns the host CPU name according to
/// LLVM. Otherwise, the string is returned as-is.
fn handle_native(cpu_name: &str) -> &str {
    match cpu_name {
        "native" => get_host_cpu_name(),
        _ => cpu_name,
    }
}

pub(crate) fn target_cpu(sess: &Session) -> &str {
    let cpu_name = sess.opts.cg.target_cpu.as_deref().unwrap_or_else(|| &sess.target.cpu);
    handle_native(cpu_name)
}

/// The list of LLVM features computed from CLI flags (`-Ctarget-cpu`, `-Ctarget-feature`,
/// `--target` and similar).
pub(crate) fn global_llvm_features(
    sess: &Session,
    diagnostics: bool,
    only_base_features: bool,
) -> Vec<String> {
    // Features that come earlier are overridden by conflicting features later in the string.
    // Typically we'll want more explicit settings to override the implicit ones, so:
    //
    // * Features from -Ctarget-cpu=*; are overridden by [^1]
    // * Features implied by --target; are overridden by
    // * Features from -Ctarget-feature; are overridden by
    // * function specific features.
    //
    // [^1]: target-cpu=native is handled here, other target-cpu values are handled implicitly
    // through LLVM TargetMachine implementation.
    //
    // FIXME(nagisa): it isn't clear what's the best interaction between features implied by
    // `-Ctarget-cpu` and `--target` are. On one hand, you'd expect CLI arguments to always
    // override anything that's implicit, so e.g. when there's no `--target` flag, features implied
    // the host target are overridden by `-Ctarget-cpu=*`. On the other hand, what about when both
    // `--target` and `-Ctarget-cpu=*` are specified? Both then imply some target features and both
    // flags are specified by the user on the CLI. It isn't as clear-cut which order of precedence
    // should be taken in cases like these.
    let mut features = vec![];

    // -Ctarget-cpu=native
    match sess.opts.cg.target_cpu {
        Some(ref s) if s == "native" => {
            // We have already figured out the actual CPU name with `LLVMRustGetHostCPUName` and set
            // that for LLVM, so the features implied by that CPU name will be available everywhere.
            // However, that is not sufficient: e.g. `skylake` alone is not sufficient to tell if
            // some of the instructions are available or not. So we have to also explicitly ask for
            // the exact set of features available on the host, and enable all of them.
            let features_string = unsafe {
                let ptr = llvm::LLVMGetHostCPUFeatures();
                let features_string = if !ptr.is_null() {
                    CStr::from_ptr(ptr)
                        .to_str()
                        .unwrap_or_else(|e| {
                            bug!("LLVM returned a non-utf8 features string: {}", e);
                        })
                        .to_owned()
                } else {
                    bug!("could not allocate host CPU features, LLVM returned a `null` string");
                };

                llvm::LLVMDisposeMessage(ptr);

                features_string
            };
            features.extend(features_string.split(',').map(String::from));
        }
        Some(_) | None => {}
    };

    // Features implied by an implicit or explicit `--target`.
    features.extend(
        sess.target
            .features
            .split(',')
            .filter(|v| !v.is_empty() && backend_feature_name(sess, v).is_some())
            // Drop +v8plus feature introduced in LLVM 20.
            .filter(|v| *v != "+v8plus" || get_version() >= (20, 0, 0))
            .map(String::from),
    );

    if wants_wasm_eh(sess) && sess.panic_strategy() == PanicStrategy::Unwind {
        features.push("+exception-handling".into());
    }

    // -Ctarget-features
    if !only_base_features {
        let known_features = sess.target.rust_target_features();
        let mut featsmap = FxHashMap::default();

        // insert implied features
        let mut all_rust_features = vec![];
        for feature in sess.opts.cg.target_feature.split(',') {
            match feature.strip_prefix('+') {
                Some(feature) => all_rust_features.extend(
                    UnordSet::from(
                        sess.target
                            .implied_target_features(std::iter::once(Symbol::intern(feature))),
                    )
                    .to_sorted_stable_ord()
                    .iter()
                    .map(|s| format!("+{}", s.as_str())),
                ),
                _ => all_rust_features.push(feature.to_string()),
            }
        }

        let feats = all_rust_features
            .iter()
            .filter_map(|s| {
                let enable_disable = match s.chars().next() {
                    None => return None,
                    Some(c @ ('+' | '-')) => c,
                    Some(_) => {
                        if diagnostics {
                            sess.dcx().emit_warn(UnknownCTargetFeaturePrefix { feature: s });
                        }
                        return None;
                    }
                };

                // Get the backend feature name, if any.
                // This excludes rustc-specific features, which do not get passed to LLVM.
                let feature = backend_feature_name(sess, s)?;
                // Warn against use of LLVM specific feature names and unstable features on the CLI.
                if diagnostics {
                    let feature_state = known_features.iter().find(|&&(v, _, _)| v == feature);
                    match feature_state {
                        None => {
                            let rust_feature =
                                known_features.iter().find_map(|&(rust_feature, _, _)| {
                                    let llvm_features = to_llvm_features(sess, rust_feature)?;
                                    if llvm_features.contains(feature)
                                        && !llvm_features.contains(rust_feature)
                                    {
                                        Some(rust_feature)
                                    } else {
                                        None
                                    }
                                });
                            let unknown_feature = if let Some(rust_feature) = rust_feature {
                                UnknownCTargetFeature {
                                    feature,
                                    rust_feature: PossibleFeature::Some { rust_feature },
                                }
                            } else {
                                UnknownCTargetFeature {
                                    feature,
                                    rust_feature: PossibleFeature::None,
                                }
                            };
                            sess.dcx().emit_warn(unknown_feature);
                        }
                        Some((_, Stability::Stable, _)) => {}
                        Some((_, Stability::Unstable(_), _)) => {
                            // An unstable feature. Warn about using it.
                            sess.dcx().emit_warn(UnstableCTargetFeature { feature });
                        }
                        Some((_, Stability::Forbidden { reason }, _)) => {
                            sess.dcx().emit_warn(ForbiddenCTargetFeature { feature, reason });
                        }
                    }

                    // FIXME(nagisa): figure out how to not allocate a full hashset here.
                    featsmap.insert(feature, enable_disable == '+');
                }

                // We run through `to_llvm_features` when
                // passing requests down to LLVM. This means that all in-language
                // features also work on the command line instead of having two
                // different names when the LLVM name and the Rust name differ.
                let llvm_feature = to_llvm_features(sess, feature)?;

                Some(
                    std::iter::once(format!(
                        "{}{}",
                        enable_disable, llvm_feature.llvm_feature_name
                    ))
                    .chain(llvm_feature.dependency.into_iter().filter_map(
                        move |feat| match (enable_disable, feat) {
                            ('-' | '+', TargetFeatureFoldStrength::Both(f))
                            | ('+', TargetFeatureFoldStrength::EnableOnly(f)) => {
                                Some(format!("{enable_disable}{f}"))
                            }
                            _ => None,
                        },
                    )),
                )
            })
            .flatten();
        features.extend(feats);

        if diagnostics && let Some(f) = check_tied_features(sess, &featsmap) {
            sess.dcx().emit_err(rustc_codegen_ssa::errors::TargetFeatureDisableOrEnable {
                features: f,
                span: None,
                missing_features: None,
            });
        }
    }

    // -Zfixed-x18
    if sess.opts.unstable_opts.fixed_x18 {
        if sess.target.arch != "aarch64" {
            sess.dcx().emit_fatal(FixedX18InvalidArch { arch: &sess.target.arch });
        } else {
            features.push("+reserve-x18".into());
        }
    }

    features
}

/// Returns a feature name for the given `+feature` or `-feature` string.
///
/// Only allows features that are backend specific (i.e. not [`RUSTC_SPECIFIC_FEATURES`].)
fn backend_feature_name<'a>(sess: &Session, s: &'a str) -> Option<&'a str> {
    // features must start with a `+` or `-`.
    let feature = s
        .strip_prefix(&['+', '-'][..])
        .unwrap_or_else(|| sess.dcx().emit_fatal(InvalidTargetFeaturePrefix { feature: s }));
    // Rustc-specific feature requests like `+crt-static` or `-crt-static`
    // are not passed down to LLVM.
    if s.is_empty() || RUSTC_SPECIFIC_FEATURES.contains(&feature) {
        return None;
    }
    Some(feature)
}

pub(crate) fn tune_cpu(sess: &Session) -> Option<&str> {
    let name = sess.opts.unstable_opts.tune_cpu.as_ref()?;
    Some(handle_native(name))
}
