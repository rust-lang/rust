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
use rustc_data_structures::fx::{FxHashMap, FxHashSet};
use rustc_data_structures::small_c_str::SmallCStr;
use rustc_fs_util::path_to_c_string;
use rustc_middle::bug;
use rustc_session::config::PrintRequest;
use rustc_session::Session;
use rustc_span::symbol::Symbol;
use rustc_target::spec::{MergeFunctions, PanicStrategy};
use smallvec::{smallvec, SmallVec};
use std::ffi::{CStr, CString};

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
pub fn to_llvm_features<'a>(sess: &Session, s: &'a str) -> SmallVec<[&'a str; 2]> {
    let arch = if sess.target.arch == "x86_64" { "x86" } else { &*sess.target.arch };
    match (arch, s) {
        ("x86", "sse4.2") => smallvec!["sse4.2", "crc32"],
        ("x86", "pclmulqdq") => smallvec!["pclmul"],
        ("x86", "rdrand") => smallvec!["rdrnd"],
        ("x86", "bmi1") => smallvec!["bmi"],
        ("x86", "cmpxchg16b") => smallvec!["cx16"],
        // FIXME: These aliases are misleading, and should be removed before avx512_target_feature is
        // stabilized. They must remain until std::arch switches off them.
        // rust#100752
        ("x86", "avx512vaes") => smallvec!["vaes"],
        ("x86", "avx512gfni") => smallvec!["gfni"],
        ("x86", "avx512vpclmulqdq") => smallvec!["vpclmulqdq"],
        ("aarch64", "rcpc2") => smallvec!["rcpc-immo"],
        ("aarch64", "dpb") => smallvec!["ccpp"],
        ("aarch64", "dpb2") => smallvec!["ccdp"],
        ("aarch64", "frintts") => smallvec!["fptoint"],
        ("aarch64", "fcma") => smallvec!["complxnum"],
        ("aarch64", "pmuv3") => smallvec!["perfmon"],
        ("aarch64", "paca") => smallvec!["pauth"],
        ("aarch64", "pacg") => smallvec!["pauth"],
        // Rust ties fp and neon together. In LLVM neon implicitly enables fp,
        // but we manually enable neon when a feature only implicitly enables fp
        ("aarch64", "f32mm") => smallvec!["f32mm", "neon"],
        ("aarch64", "f64mm") => smallvec!["f64mm", "neon"],
        ("aarch64", "fhm") => smallvec!["fp16fml", "neon"],
        ("aarch64", "fp16") => smallvec!["fullfp16", "neon"],
        ("aarch64", "jsconv") => smallvec!["jsconv", "neon"],
        ("aarch64", "sve") => smallvec!["sve", "neon"],
        ("aarch64", "sve2") => smallvec!["sve2", "neon"],
        ("aarch64", "sve2-aes") => smallvec!["sve2-aes", "neon"],
        ("aarch64", "sve2-sm4") => smallvec!["sve2-sm4", "neon"],
        ("aarch64", "sve2-sha3") => smallvec!["sve2-sha3", "neon"],
        ("aarch64", "sve2-bitperm") => smallvec!["sve2-bitperm", "neon"],
        (_, s) => smallvec![s],
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

fn print_target_features(sess: &Session, tm: &llvm::TargetMachine) {
    let mut llvm_target_features = llvm_target_features(tm);
    let mut known_llvm_target_features = FxHashSet::<&'static str>::default();
    let mut rustc_target_features = supported_target_features(sess)
        .iter()
        .map(|(feature, _gate)| {
            let desc = if let Some(llvm_feature) = to_llvm_features(sess, *feature).first() {
                // LLVM asserts that these are sorted. LLVM and Rust both use byte comparison for these strings.
                match llvm_target_features.binary_search_by_key(&llvm_feature, |(f, _d)| f).ok() {
                    Some(index) => {
                        known_llvm_target_features.insert(llvm_feature);
                        llvm_target_features[index].1
                    }
                    None => "",
                }
            } else {
                ""
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

    println!("Features supported by rustc for this target:");
    for (feature, desc) in &rustc_target_features {
        println!("    {1:0$} - {2}.", max_feature_len, feature, desc);
    }
    println!("\nCode-generation features supported by LLVM for this target:");
    for (feature, desc) in &llvm_target_features {
        println!("    {1:0$} - {2}.", max_feature_len, feature, desc);
    }
    if llvm_target_features.is_empty() {
        println!("    Target features listing is not supported by this LLVM version.");
    }
    println!("\nUse +feature to enable a feature, or -feature to disable it.");
    println!("For example, rustc -C target-cpu=mycpu -C target-feature=+feature1,-feature2\n");
    println!("Code-generation features cannot be used in cfg or #[target_feature],");
    println!("and may be renamed or removed in a future version of LLVM or rustc.\n");
}

pub(crate) fn print(req: PrintRequest, sess: &Session) {
    require_inited();
    let tm = create_informational_target_machine(sess);
    match req {
        PrintRequest::TargetCPUs => unsafe { llvm::LLVMRustPrintTargetCPUs(tm) },
        PrintRequest::TargetFeatures => print_target_features(sess, tm),
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
                Some(c @ '+' | c @ '-') => c,
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
            Some(
                to_llvm_features(sess, feature)
                    .into_iter()
                    .map(move |f| format!("{}{}", enable_disable, f)),
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
