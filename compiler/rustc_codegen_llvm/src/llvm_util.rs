use crate::back::write::create_informational_target_machine;
use crate::errors::{
    PossibleFeature, TargetFeatureDisableOrEnable, UnknownCTargetFeature,
    UnknownCTargetFeaturePrefix,
};
use crate::llvm;
use libc::c_int;
use rustc_codegen_ssa::target_features::{
    supported_target_features, tied_target_features, RUSTC_SPECIFIC_FEATURES,
};
use rustc_codegen_ssa::traits::PrintBackendInfo;
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_fs_util::path_to_c_string;
use rustc_middle::bug;
use rustc_session::config::{PrintKind, PrintRequest};
use rustc_session::Session;
use rustc_span::symbol::Symbol;
use rustc_target::spec::{MergeFunctions, PanicStrategy};

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::Path;
use std::ptr;
use std::slice;
use std::str;
use std::sync::Once;

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

    llvm::LLVMRustInstallFatalErrorHandler();
    // On Windows, an LLVM assertion will open an Abort/Retry/Ignore dialog
    // box for the purpose of launching a debugger. However, on CI this will
    // cause it to hang until it times out, which can take several hours.
    if std::env::var_os("CI").is_some() {
        llvm::LLVMRustDisableSystemDialogsOnCrash();
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
    }

    if sess.opts.unstable_opts.llvm_time_trace {
        llvm::LLVMTimeTraceProfilerInitialize();
    }

    rustc_llvm::initialize_available_targets();

    llvm::LLVMRustSetLLVMOptions(llvm_args.len() as c_int, llvm_args.as_ptr());
}

pub fn time_trace_profiler_finish(file_name: &Path) {
    unsafe {
        let file_name = path_to_c_string(file_name);
        llvm::LLVMTimeTraceProfilerFinish(file_name.as_ptr());
    }
}

pub enum TargetFeatureFoldStrength<'a> {
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

pub struct LLVMFeature<'a> {
    pub llvm_feature_name: &'a str,
    pub dependency: Option<TargetFeatureFoldStrength<'a>>,
}

impl<'a> LLVMFeature<'a> {
    pub fn new(llvm_feature_name: &'a str) -> Self {
        Self { llvm_feature_name, dependency: None }
    }

    pub fn with_dependency(
        llvm_feature_name: &'a str,
        dependency: TargetFeatureFoldStrength<'a>,
    ) -> Self {
        Self { llvm_feature_name, dependency: Some(dependency) }
    }

    pub fn contains(&self, feat: &str) -> bool {
        self.iter().any(|dep| dep == feat)
    }

    pub fn iter(&'a self) -> impl Iterator<Item = &'a str> {
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
// To find a list of LLVM's names, check llvm-project/llvm/include/llvm/Support/*TargetParser.def
// where the * matches the architecture's name
//
// For targets not present in the above location, see llvm-project/llvm/lib/Target/{ARCH}/*.td
// where `{ARCH}` is the architecture name. Look for instances of `SubtargetFeature`.
//
// Beware to not use the llvm github project for this, but check the git submodule
// found in src/llvm-project
// Though note that Rust can also be build with an external precompiled version of LLVM
// which might lead to failures if the oldest tested / supported LLVM version
// doesn't yet support the relevant intrinsics
pub fn to_llvm_features<'a>(sess: &Session, s: &'a str) -> LLVMFeature<'a> {
    let arch = if sess.target.arch == "x86_64" { "x86" } else { &*sess.target.arch };
    match (arch, s) {
        ("x86", "sse4.2") => {
            LLVMFeature::with_dependency("sse4.2", TargetFeatureFoldStrength::EnableOnly("crc32"))
        }
        ("x86", "pclmulqdq") => LLVMFeature::new("pclmul"),
        ("x86", "rdrand") => LLVMFeature::new("rdrnd"),
        ("x86", "bmi1") => LLVMFeature::new("bmi"),
        ("x86", "cmpxchg16b") => LLVMFeature::new("cx16"),
        ("aarch64", "rcpc2") => LLVMFeature::new("rcpc-immo"),
        ("aarch64", "dpb") => LLVMFeature::new("ccpp"),
        ("aarch64", "dpb2") => LLVMFeature::new("ccdp"),
        ("aarch64", "frintts") => LLVMFeature::new("fptoint"),
        ("aarch64", "fcma") => LLVMFeature::new("complxnum"),
        ("aarch64", "pmuv3") => LLVMFeature::new("perfmon"),
        ("aarch64", "paca") => LLVMFeature::new("pauth"),
        ("aarch64", "pacg") => LLVMFeature::new("pauth"),
        // Rust ties fp and neon together.
        ("aarch64", "neon") => {
            LLVMFeature::with_dependency("neon", TargetFeatureFoldStrength::Both("fp-armv8"))
        }
        // In LLVM neon implicitly enables fp, but we manually enable
        // neon when a feature only implicitly enables fp
        ("aarch64", "f32mm") => {
            LLVMFeature::with_dependency("f32mm", TargetFeatureFoldStrength::EnableOnly("neon"))
        }
        ("aarch64", "f64mm") => {
            LLVMFeature::with_dependency("f64mm", TargetFeatureFoldStrength::EnableOnly("neon"))
        }
        ("aarch64", "fhm") => {
            LLVMFeature::with_dependency("fp16fml", TargetFeatureFoldStrength::EnableOnly("neon"))
        }
        ("aarch64", "fp16") => {
            LLVMFeature::with_dependency("fullfp16", TargetFeatureFoldStrength::EnableOnly("neon"))
        }
        ("aarch64", "jsconv") => {
            LLVMFeature::with_dependency("jsconv", TargetFeatureFoldStrength::EnableOnly("neon"))
        }
        ("aarch64", "sve") => {
            LLVMFeature::with_dependency("sve", TargetFeatureFoldStrength::EnableOnly("neon"))
        }
        ("aarch64", "sve2") => {
            LLVMFeature::with_dependency("sve2", TargetFeatureFoldStrength::EnableOnly("neon"))
        }
        ("aarch64", "sve2-aes") => {
            LLVMFeature::with_dependency("sve2-aes", TargetFeatureFoldStrength::EnableOnly("neon"))
        }
        ("aarch64", "sve2-sm4") => {
            LLVMFeature::with_dependency("sve2-sm4", TargetFeatureFoldStrength::EnableOnly("neon"))
        }
        ("aarch64", "sve2-sha3") => {
            LLVMFeature::with_dependency("sve2-sha3", TargetFeatureFoldStrength::EnableOnly("neon"))
        }
        ("aarch64", "sve2-bitperm") => LLVMFeature::with_dependency(
            "sve2-bitperm",
            TargetFeatureFoldStrength::EnableOnly("neon"),
        ),
        (_, s) => LLVMFeature::new(s),
    }
}

/// Given a map from target_features to whether they are enabled or disabled,
/// ensure only valid combinations are allowed.
pub fn check_tied_features(
    sess: &Session,
    features: &FxHashMap<&str, bool>,
) -> Option<&'static [&'static str]> {
    if !features.is_empty() {
        for tied in tied_target_features(sess) {
            // Tied features must be set to the same value, or not set at all
            let mut tied_iter = tied.iter();
            let enabled = features.get(tied_iter.next().unwrap());
            if tied_iter.any(|f| enabled != features.get(f)) {
                return Some(tied);
            }
        }
    }
    return None;
}

/// Used to generate cfg variables and apply features
/// Must express features in the way Rust understands them
pub fn target_features(sess: &Session, allow_unstable: bool) -> Vec<Symbol> {
    let target_machine = create_informational_target_machine(sess);
    supported_target_features(sess)
        .iter()
        .filter_map(|&(feature, gate)| {
            if sess.is_nightly_build() || allow_unstable || gate.is_none() {
                Some(feature)
            } else {
                None
            }
        })
        .filter(|feature| {
            // check that all features in a given smallvec are enabled
            for llvm_feature in to_llvm_features(sess, feature) {
                let cstr = SmallCStr::new(llvm_feature);
                if !unsafe { llvm::LLVMRustHasFeature(target_machine, cstr.as_ptr()) } {
                    return false;
                }
            }
            true
        })
        .map(|feature| Symbol::intern(feature))
        .collect()
}

pub fn print_version() {
    let (major, minor, patch) = get_version();
    println!("LLVM version: {}.{}.{}", major, minor, patch);
}

pub fn get_version() -> (u32, u32, u32) {
    // Can be called without initializing LLVM
    unsafe {
        (llvm::LLVMRustVersionMajor(), llvm::LLVMRustVersionMinor(), llvm::LLVMRustVersionPatch())
    }
}

pub fn print_passes() {
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

fn print_target_features(out: &mut dyn PrintBackendInfo, sess: &Session, tm: &llvm::TargetMachine) {
    let mut llvm_target_features = llvm_target_features(tm);
    let mut known_llvm_target_features = FxHashSet::<&'static str>::default();
    let mut rustc_target_features = supported_target_features(sess)
        .iter()
        .map(|(feature, _gate)| {
            // LLVM asserts that these are sorted. LLVM and Rust both use byte comparison for these strings.
            let llvm_feature = to_llvm_features(sess, *feature).llvm_feature_name;
            let desc =
                match llvm_target_features.binary_search_by_key(&llvm_feature, |(f, _d)| f).ok() {
                    Some(index) => {
                        known_llvm_target_features.insert(llvm_feature);
                        llvm_target_features[index].1
                    }
                    None => "",
                };

            (*feature, desc)
        })
        .collect::<Vec<_>>();
    rustc_target_features.extend_from_slice(&[(
        "crt-static",
        "Enables C Run-time Libraries to be statically linked",
    )]);
    llvm_target_features.retain(|(f, _d)| !known_llvm_target_features.contains(f));

    let max_feature_len = llvm_target_features
        .iter()
        .chain(rustc_target_features.iter())
        .map(|(feature, _desc)| feature.len())
        .max()
        .unwrap_or(0);

    writeln!(out, "Features supported by rustc for this target:");
    for (feature, desc) in &rustc_target_features {
        writeln!(out, "    {1:0$} - {2}.", max_feature_len, feature, desc);
    }
    writeln!(out, "\nCode-generation features supported by LLVM for this target:");
    for (feature, desc) in &llvm_target_features {
        writeln!(out, "    {1:0$} - {2}.", max_feature_len, feature, desc);
    }
    if llvm_target_features.is_empty() {
        writeln!(out, "    Target features listing is not supported by this LLVM version.");
    }
    writeln!(out, "\nUse +feature to enable a feature, or -feature to disable it.");
    writeln!(out, "For example, rustc -C target-cpu=mycpu -C target-feature=+feature1,-feature2\n");
    writeln!(out, "Code-generation features cannot be used in cfg or #[target_feature],");
    writeln!(out, "and may be renamed or removed in a future version of LLVM or rustc.\n");
}

pub(crate) fn print(req: &PrintRequest, mut out: &mut dyn PrintBackendInfo, sess: &Session) {
    require_inited();
    let tm = create_informational_target_machine(sess);
    match req.kind {
        PrintKind::TargetCPUs => {
            // SAFETY generate a C compatible string from a byte slice to pass
            // the target CPU name into LLVM, the lifetime of the reference is
            // at least as long as the C function
            let cpu_cstring = CString::new(handle_native(sess.target.cpu.as_ref()))
                .unwrap_or_else(|e| bug!("failed to convert to cstring: {}", e));
            unsafe extern "C" fn callback(out: *mut c_void, string: *const c_char, len: usize) {
                let out = &mut *(out as *mut &mut dyn PrintBackendInfo);
                let bytes = slice::from_raw_parts(string as *const u8, len);
                write!(out, "{}", String::from_utf8_lossy(bytes));
            }
            unsafe {
                llvm::LLVMRustPrintTargetCPUs(
                    tm,
                    cpu_cstring.as_ptr(),
                    callback,
                    &mut out as *mut &mut dyn PrintBackendInfo as *mut c_void,
                );
            }
        }
        PrintKind::TargetFeatures => print_target_features(out, sess, tm),
        _ => bug!("rustc_codegen_llvm can't handle print request: {:?}", req),
    }
}

fn handle_native(name: &str) -> &str {
    if name != "native" {
        return name;
    }

    unsafe {
        let mut len = 0;
        let ptr = llvm::LLVMRustGetHostCPUName(&mut len);
        str::from_utf8(slice::from_raw_parts(ptr as *const u8, len)).unwrap()
    }
}

pub fn target_cpu(sess: &Session) -> &str {
    match sess.opts.cg.target_cpu {
        Some(ref name) => handle_native(name),
        None => handle_native(sess.target.cpu.as_ref()),
    }
}

/// The list of LLVM features computed from CLI flags (`-Ctarget-cpu`, `-Ctarget-feature`,
/// `--target` and similar).
pub(crate) fn global_llvm_features(sess: &Session, diagnostics: bool) -> Vec<String> {
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
            .filter(|v| !v.is_empty() && backend_feature_name(v).is_some())
            // Drop +atomics-32 feature introduced in LLVM 15.
            .filter(|v| *v != "+atomics-32" || get_version() >= (15, 0, 0))
            .map(String::from),
    );

    // -Ctarget-features
    let supported_features = supported_target_features(sess);
    let mut featsmap = FxHashMap::default();
    let feats = sess
        .opts
        .cg
        .target_feature
        .split(',')
        .filter_map(|s| {
            let enable_disable = match s.chars().next() {
                None => return None,
                Some(c @ ('+' | '-')) => c,
                Some(_) => {
                    if diagnostics {
                        sess.emit_warning(UnknownCTargetFeaturePrefix { feature: s });
                    }
                    return None;
                }
            };

            let feature = backend_feature_name(s)?;
            // Warn against use of LLVM specific feature names on the CLI.
            if diagnostics && !supported_features.iter().any(|&(v, _)| v == feature) {
                let rust_feature = supported_features.iter().find_map(|&(rust_feature, _)| {
                    let llvm_features = to_llvm_features(sess, rust_feature);
                    if llvm_features.contains(&feature) && !llvm_features.contains(&rust_feature) {
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
                    UnknownCTargetFeature { feature, rust_feature: PossibleFeature::None }
                };
                sess.emit_warning(unknown_feature);
            }

            if diagnostics {
                // FIXME(nagisa): figure out how to not allocate a full hashset here.
                featsmap.insert(feature, enable_disable == '+');
            }

            // rustc-specific features do not get passed down to LLVM…
            if RUSTC_SPECIFIC_FEATURES.contains(&feature) {
                return None;
            }
            // ... otherwise though we run through `to_llvm_features` when
            // passing requests down to LLVM. This means that all in-language
            // features also work on the command line instead of having two
            // different names when the LLVM name and the Rust name differ.
            let llvm_feature = to_llvm_features(sess, feature);

            Some(
                std::iter::once(format!("{}{}", enable_disable, llvm_feature.llvm_feature_name))
                    .chain(llvm_feature.dependency.into_iter().filter_map(move |feat| {
                        match (enable_disable, feat) {
                            ('-' | '+', TargetFeatureFoldStrength::Both(f))
                            | ('+', TargetFeatureFoldStrength::EnableOnly(f)) => {
                                Some(format!("{}{}", enable_disable, f))
                            }
                            _ => None,
                        }
                    })),
            )
        })
        .flatten();
    features.extend(feats);

    if diagnostics && let Some(f) = check_tied_features(sess, &featsmap) {
        sess.emit_err(TargetFeatureDisableOrEnable {
            features: f,
            span: None,
            missing_features: None,
        });
    }

    features
}

/// Returns a feature name for the given `+feature` or `-feature` string.
///
/// Only allows features that are backend specific (i.e. not [`RUSTC_SPECIFIC_FEATURES`].)
fn backend_feature_name(s: &str) -> Option<&str> {
    // features must start with a `+` or `-`.
    let feature = s.strip_prefix(&['+', '-'][..]).unwrap_or_else(|| {
        bug!("target feature `{}` must begin with a `+` or `-`", s);
    });
    // Rustc-specific feature requests like `+crt-static` or `-crt-static`
    // are not passed down to LLVM.
    if RUSTC_SPECIFIC_FEATURES.contains(&feature) {
        return None;
    }
    Some(feature)
}

pub fn tune_cpu(sess: &Session) -> Option<&str> {
    let name = sess.opts.unstable_opts.tune_cpu.as_ref()?;
    Some(handle_native(name))
}
