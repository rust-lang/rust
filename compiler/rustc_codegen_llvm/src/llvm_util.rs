use crate::back::write::create_informational_target_machine;
use crate::{llvm, llvm_util};
use libc::c_int;
use rustc_codegen_ssa::target_features::supported_target_features;
use rustc_data_structures::fx::FxHashSet;
use rustc_metadata::dynamic_lib::DynamicLibrary;
use rustc_middle::bug;
use rustc_session::config::PrintRequest;
use rustc_session::Session;
use rustc_span::symbol::Symbol;
use rustc_target::spec::{MergeFunctions, PanicStrategy};
use std::ffi::{CStr, CString};
use tracing::debug;

use std::mem;
use std::path::Path;
use std::ptr;
use std::slice;
use std::str;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;

static POISONED: AtomicBool = AtomicBool::new(false);
static INIT: Once = Once::new();

pub(crate) fn init(sess: &Session) {
    unsafe {
        // Before we touch LLVM, make sure that multithreading is enabled.
        INIT.call_once(|| {
            if llvm::LLVMStartMultithreaded() != 1 {
                // use an extra bool to make sure that all future usage of LLVM
                // cannot proceed despite the Once not running more than once.
                POISONED.store(true, Ordering::SeqCst);
            }

            configure_llvm(sess);
        });

        if POISONED.load(Ordering::SeqCst) {
            bug!("couldn't enable multi-threaded LLVM");
        }
    }
}

fn require_inited() {
    INIT.call_once(|| bug!("llvm is not initialized"));
    if POISONED.load(Ordering::SeqCst) {
        bug!("couldn't enable multi-threaded LLVM");
    }
}

unsafe fn configure_llvm(sess: &Session) {
    let n_args = sess.opts.cg.llvm_args.len() + sess.target.llvm_args.len();
    let mut llvm_c_strs = Vec::with_capacity(n_args + 1);
    let mut llvm_args = Vec::with_capacity(n_args + 1);

    llvm::LLVMRustInstallFatalErrorHandler();

    fn llvm_arg_to_arg_name(full_arg: &str) -> &str {
        full_arg.trim().split(|c: char| c == '=' || c.is_whitespace()).next().unwrap_or("")
    }

    let cg_opts = sess.opts.cg.llvm_args.iter();
    let tg_opts = sess.target.llvm_args.iter();
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
        if sess.time_llvm_passes() {
            add("-time-passes", false);
        }
        if sess.print_llvm_passes() {
            add("-debug-pass=Structure", false);
        }
        if !sess.opts.debugging_opts.no_generate_arange_section {
            add("-generate-arange-section", false);
        }

        // Disable the machine outliner by default in LLVM versions 11 and LLVM
        // version 12, where it leads to miscompilation.
        //
        // Ref:
        // - https://github.com/rust-lang/rust/issues/85351
        // - https://reviews.llvm.org/D103167
        let llvm_version = llvm_util::get_version();
        if llvm_version >= (11, 0, 0) && llvm_version < (13, 0, 0) {
            add("-enable-machine-outliner=never", false);
        }

        match sess.opts.debugging_opts.merge_functions.unwrap_or(sess.target.merge_functions) {
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

    if sess.opts.debugging_opts.llvm_time_trace {
        // time-trace is not thread safe and running it in parallel will cause seg faults.
        if !sess.opts.debugging_opts.no_parallel_llvm {
            bug!("`-Z llvm-time-trace` requires `-Z no-parallel-llvm")
        }

        llvm::LLVMTimeTraceProfilerInitialize();
    }

    llvm::LLVMInitializePasses();

    for plugin in &sess.opts.debugging_opts.llvm_plugins {
        let path = Path::new(plugin);
        let res = DynamicLibrary::open(path);
        match res {
            Ok(_) => debug!("LLVM plugin loaded succesfully {} ({})", path.display(), plugin),
            Err(e) => bug!("couldn't load plugin: {}", e),
        }
        mem::forget(res);
    }

    rustc_llvm::initialize_available_targets();

    llvm::LLVMRustSetLLVMOptions(llvm_args.len() as c_int, llvm_args.as_ptr());
}

pub fn time_trace_profiler_finish(file_name: &str) {
    unsafe {
        let file_name = CString::new(file_name).unwrap();
        llvm::LLVMTimeTraceProfilerFinish(file_name.as_ptr());
    }
}

// WARNING: the features after applying `to_llvm_feature` must be known
// to LLVM or the feature detection code will walk past the end of the feature
// array, leading to crashes.
// To find a list of LLVM's names, check llvm-project/llvm/include/llvm/Support/*TargetParser.def
// where the * matches the architecture's name
// Beware to not use the llvm github project for this, but check the git submodule
// found in src/llvm-project
// Though note that Rust can also be build with an external precompiled version of LLVM
// which might lead to failures if the oldest tested / supported LLVM version
// doesn't yet support the relevant intrinsics
pub fn to_llvm_feature<'a>(sess: &Session, s: &'a str) -> Vec<&'a str> {
    let arch = if sess.target.arch == "x86_64" { "x86" } else { &*sess.target.arch };
    match (arch, s) {
        ("x86", "sse4.2") => {
            if get_version() >= (14, 0, 0) {
                vec!["sse4.2", "crc32"]
            } else {
                vec!["sse4.2"]
            }
        }
        ("x86", "pclmulqdq") => vec!["pclmul"],
        ("x86", "rdrand") => vec!["rdrnd"],
        ("x86", "bmi1") => vec!["bmi"],
        ("x86", "cmpxchg16b") => vec!["cx16"],
        ("x86", "avx512vaes") => vec!["vaes"],
        ("x86", "avx512gfni") => vec!["gfni"],
        ("x86", "avx512vpclmulqdq") => vec!["vpclmulqdq"],
        ("aarch64", "fp") => vec!["fp-armv8"],
        ("aarch64", "fp16") => vec!["fullfp16"],
        ("aarch64", "fhm") => vec!["fp16fml"],
        ("aarch64", "rcpc2") => vec!["rcpc-immo"],
        ("aarch64", "dpb") => vec!["ccpp"],
        ("aarch64", "dpb2") => vec!["ccdp"],
        ("aarch64", "frintts") => vec!["fptoint"],
        ("aarch64", "fcma") => vec!["complxnum"],
        (_, s) => vec![s],
    }
}

pub fn target_features(sess: &Session) -> Vec<Symbol> {
    let target_machine = create_informational_target_machine(sess);
    supported_target_features(sess)
        .iter()
        .filter_map(
            |&(feature, gate)| {
                if sess.is_nightly_build() || gate.is_none() { Some(feature) } else { None }
            },
        )
        .filter(|feature| {
            for llvm_feature in to_llvm_feature(sess, feature) {
                let cstr = CString::new(llvm_feature).unwrap();
                if unsafe { llvm::LLVMRustHasFeature(target_machine, cstr.as_ptr()) } {
                    return true;
                }
            }
            false
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
    let mut target_features = llvm_target_features(tm);
    let mut rustc_target_features = supported_target_features(sess)
        .iter()
        .filter_map(|(feature, _gate)| {
            for llvm_feature in to_llvm_feature(sess, *feature) {
                // LLVM asserts that these are sorted. LLVM and Rust both use byte comparison for these strings.
                match target_features.binary_search_by_key(&llvm_feature, |(f, _d)| (*f)).ok().map(
                    |index| {
                        let (_f, desc) = target_features.remove(index);
                        (*feature, desc)
                    },
                ) {
                    Some(v) => return Some(v),
                    None => {}
                }
            }
            None
        })
        .collect::<Vec<_>>();
    rustc_target_features.extend_from_slice(&[(
        "crt-static",
        "Enables C Run-time Libraries to be statically linked",
    )]);
    let max_feature_len = target_features
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
    for (feature, desc) in &target_features {
        println!("    {1:0$} - {2}.", max_feature_len, feature, desc);
    }
    if target_features.is_empty() {
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
    let name = sess.opts.cg.target_cpu.as_ref().unwrap_or(&sess.target.cpu);
    handle_native(name)
}

/// The list of LLVM features computed from CLI flags (`-Ctarget-cpu`, `-Ctarget-feature`,
/// `--target` and similar).
// FIXME(nagisa): Cache the output of this somehow? Maybe make this a query? We're calling this
// for every function that has `#[target_feature]` on it. The global features won't change between
// the functions; only crates, maybe…
pub fn llvm_global_features(sess: &Session) -> Vec<String> {
    // FIXME(nagisa): this should definitely be available more centrally and to other codegen backends.
    /// These features control behaviour of rustc rather than llvm.
    const RUSTC_SPECIFIC_FEATURES: &[&str] = &["crt-static"];

    // Features that come earlier are overriden by conflicting features later in the string.
    // Typically we'll want more explicit settings to override the implicit ones, so:
    //
    // * Features from -Ctarget-cpu=*; are overriden by [^1]
    // * Features implied by --target; are overriden by
    // * Features from -Ctarget-feature; are overriden by
    // * function specific features.
    //
    // [^1]: target-cpu=native is handled here, other target-cpu values are handled implicitly
    // through LLVM TargetMachine implementation.
    //
    // FIXME(nagisa): it isn't clear what's the best interaction between features implied by
    // `-Ctarget-cpu` and `--target` are. On one hand, you'd expect CLI arguments to always
    // override anything that's implicit, so e.g. when there's no `--target` flag, features implied
    // the host target are overriden by `-Ctarget-cpu=*`. On the other hand, what about when both
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

    let filter = |s: &str| {
        if s.is_empty() {
            return vec![];
        }
        let feature = if s.starts_with('+') || s.starts_with('-') {
            &s[1..]
        } else {
            return vec![s.to_string()];
        };
        // Rustc-specific feature requests like `+crt-static` or `-crt-static`
        // are not passed down to LLVM.
        if RUSTC_SPECIFIC_FEATURES.contains(&feature) {
            return vec![];
        }
        // ... otherwise though we run through `to_llvm_feature` feature when
        // passing requests down to LLVM. This means that all in-language
        // features also work on the command line instead of having two
        // different names when the LLVM name and the Rust name differ.
        to_llvm_feature(sess, feature).iter().map(|f| format!("{}{}", &s[..1], f)).collect()
    };

    // Features implied by an implicit or explicit `--target`.
    features.extend(sess.target.features.split(',').flat_map(&filter));

    // -Ctarget-features
    features.extend(sess.opts.cg.target_feature.split(',').flat_map(&filter));

    // FIXME: Move outline-atomics to target definition when earliest supported LLVM is 12.
    if get_version() >= (12, 0, 0) && sess.target.llvm_target.contains("aarch64-unknown-linux") {
        features.push("+outline-atomics".to_string());
    }

    features
}

pub fn tune_cpu(sess: &Session) -> Option<&str> {
    let name = sess.opts.debugging_opts.tune_cpu.as_ref()?;
    Some(handle_native(name))
}
