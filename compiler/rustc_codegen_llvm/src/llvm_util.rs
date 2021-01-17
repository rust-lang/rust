use crate::back::write::create_informational_target_machine;
use crate::llvm;
use libc::c_int;
use rustc_codegen_ssa::target_features::supported_target_features;
use rustc_data_structures::fx::FxHashSet;
use rustc_middle::bug;
use rustc_session::config::PrintRequest;
use rustc_session::Session;
use rustc_span::symbol::Symbol;
use rustc_target::spec::{MergeFunctions, PanicStrategy};
use std::ffi::{CStr, CString};

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
pub fn to_llvm_feature<'a>(sess: &Session, s: &'a str) -> &'a str {
    let arch = if sess.target.arch == "x86_64" { "x86" } else { &*sess.target.arch };
    match (arch, s) {
        ("x86", "pclmulqdq") => "pclmul",
        ("x86", "rdrand") => "rdrnd",
        ("x86", "bmi1") => "bmi",
        ("x86", "cmpxchg16b") => "cx16",
        ("x86", "avx512vaes") => "vaes",
        ("x86", "avx512gfni") => "gfni",
        ("x86", "avx512vpclmulqdq") => "vpclmulqdq",
        ("aarch64", "fp") => "fp-armv8",
        ("aarch64", "fp16") => "fullfp16",
        (_, s) => s,
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
            let llvm_feature = to_llvm_feature(sess, feature);
            let cstr = CString::new(llvm_feature).unwrap();
            unsafe { llvm::LLVMRustHasFeature(target_machine, cstr.as_ptr()) }
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

pub(crate) fn print(req: PrintRequest, sess: &Session) {
    require_inited();
    let tm = create_informational_target_machine(sess);
    unsafe {
        match req {
            PrintRequest::TargetCPUs => llvm::LLVMRustPrintTargetCPUs(tm),
            PrintRequest::TargetFeatures => llvm::LLVMRustPrintTargetFeatures(tm),
            _ => bug!("rustc_codegen_llvm can't handle print request: {:?}", req),
        }
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

pub fn handle_native_features(sess: &Session) -> Vec<String> {
    match sess.opts.cg.target_cpu {
        Some(ref s) => {
            if s != "native" {
                return vec![];
            }

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

            features_string.split(",").map(|s| s.to_owned()).collect()
        }
        None => vec![],
    }
}

pub fn tune_cpu(sess: &Session) -> Option<&str> {
    let name = sess.opts.debugging_opts.tune_cpu.as_ref()?;
    Some(handle_native(name))
}
