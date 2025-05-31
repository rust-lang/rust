//! Shim which is passed to Cargo as "rustc" when running the bootstrap.
//!
//! This shim will take care of some various tasks that our build process
//! requires that Cargo can't quite do through normal configuration:
//!
//! 1. When compiling build scripts and build dependencies, we need a guaranteed
//!    full standard library available. The only compiler which actually has
//!    this is the snapshot, so we detect this situation and always compile with
//!    the snapshot compiler.
//! 2. We pass a bunch of `--cfg` and other flags based on what we're compiling
//!    (and this slightly differs based on a whether we're using a snapshot or
//!    not), so we do that all here.
//!
//! This may one day be replaced by RUSTFLAGS, but the dynamic nature of
//! switching compilers for the bootstrap and for build scripts will probably
//! never get replaced.

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::Instant;

use shared_helpers::{
    dylib_path, dylib_path_var, exe, maybe_dump, parse_rustc_stage, parse_rustc_verbose,
    parse_value_from_args,
};

#[path = "../utils/shared_helpers.rs"]
mod shared_helpers;

#[path = "../utils/proc_macro_deps.rs"]
mod proc_macro_deps;

fn main() {
    let orig_args = env::args_os().skip(1).collect::<Vec<_>>();
    let mut args = orig_args.clone();

    let stage = parse_rustc_stage();
    let verbose = parse_rustc_verbose();

    // Detect whether or not we're a build script depending on whether --target
    // is passed (a bit janky...)
    let target = parse_value_from_args(&orig_args, "--target");
    let version = args.iter().find(|w| &**w == "-vV");

    // Use a different compiler for build scripts, since there may not yet be a
    // libstd for the real compiler to use. However, if Cargo is attempting to
    // determine the version of the compiler, the real compiler needs to be
    // used. Currently, these two states are differentiated based on whether
    // --target and -vV is/isn't passed.
    let is_build_script = target.is_none() && version.is_none();
    let (rustc, libdir) = if is_build_script {
        ("RUSTC_SNAPSHOT", "RUSTC_SNAPSHOT_LIBDIR")
    } else {
        ("RUSTC_REAL", "RUSTC_LIBDIR")
    };

    let sysroot = env::var_os("RUSTC_SYSROOT").expect("RUSTC_SYSROOT was not set");
    let on_fail = env::var_os("RUSTC_ON_FAIL").map(Command::new);

    let rustc_real = env::var_os(rustc).unwrap_or_else(|| panic!("{rustc:?} was not set"));
    let libdir = env::var_os(libdir).unwrap_or_else(|| panic!("{libdir:?} was not set"));
    let mut dylib_path = dylib_path();
    dylib_path.insert(0, PathBuf::from(&libdir));

    // if we're running clippy, trust cargo-clippy to set clippy-driver appropriately (and don't override it with rustc).
    // otherwise, substitute whatever cargo thinks rustc should be with RUSTC_REAL.
    // NOTE: this means we ignore RUSTC in the environment.
    // FIXME: We might want to consider removing RUSTC_REAL and setting RUSTC directly?
    // NOTE: we intentionally pass the name of the host, not the target.
    let host = env::var("CFG_COMPILER_BUILD_TRIPLE").unwrap();
    let is_clippy = args[0].to_string_lossy().ends_with(&exe("clippy-driver", &host));
    let rustc_driver = if is_clippy {
        if is_build_script {
            // Don't run clippy on build scripts (for one thing, we may not have libstd built with
            // the appropriate version yet, e.g. for stage 1 std).
            // Also remove the `clippy-driver` param in addition to the RUSTC param.
            args.drain(..2);
            rustc_real
        } else {
            args.remove(0)
        }
    } else {
        // Cargo doesn't respect RUSTC_WRAPPER for version information >:(
        // don't remove the first arg if we're being run as RUSTC instead of RUSTC_WRAPPER.
        // Cargo also sometimes doesn't pass the `.exe` suffix on Windows - add it manually.
        let current_exe = env::current_exe().expect("couldn't get path to rustc shim");
        let arg0 = exe(args[0].to_str().expect("only utf8 paths are supported"), &host);
        if Path::new(&arg0) == current_exe {
            args.remove(0);
        }
        rustc_real
    };

    // Get the name of the crate we're compiling, if any.
    let crate_name = parse_value_from_args(&orig_args, "--crate-name");

    // When statically linking `std` into `rustc_driver`, remove `-C prefer-dynamic`
    if env::var("RUSTC_LINK_STD_INTO_RUSTC_DRIVER").unwrap() == "1"
        && crate_name == Some("rustc_driver")
    {
        if let Some(pos) = args.iter().enumerate().position(|(i, a)| {
            a == "-C" && args.get(i + 1).map(|a| a == "prefer-dynamic").unwrap_or(false)
        }) {
            args.remove(pos);
            args.remove(pos);
        }
        if let Some(pos) = args.iter().position(|a| a == "-Cprefer-dynamic") {
            args.remove(pos);
        }
    }

    let mut cmd = match env::var_os("RUSTC_WRAPPER_REAL") {
        Some(wrapper) if !wrapper.is_empty() => {
            let mut cmd = Command::new(wrapper);
            cmd.arg(rustc_driver);
            cmd
        }
        _ => Command::new(rustc_driver),
    };
    cmd.args(&args).env(dylib_path_var(), env::join_paths(&dylib_path).unwrap());

    if let Some(crate_name) = crate_name
        && let Some(target) = env::var_os("RUSTC_TIME")
        && (target == "all"
            || target.into_string().unwrap().split(',').any(|c| c.trim() == crate_name))
    {
        cmd.arg("-Ztime-passes");
    }

    // Print backtrace in case of ICE
    if env::var("RUSTC_BACKTRACE_ON_ICE").is_ok() && env::var("RUST_BACKTRACE").is_err() {
        cmd.env("RUST_BACKTRACE", "1");
    }

    if let Ok(lint_flags) = env::var("RUSTC_LINT_FLAGS") {
        cmd.args(lint_flags.split_whitespace());
    }

    // Conditionally pass `-Zon-broken-pipe=kill` to underlying rustc. Not all binaries want
    // `-Zon-broken-pipe=kill`, which includes cargo itself.
    if env::var_os("FORCE_ON_BROKEN_PIPE_KILL").is_some() {
        cmd.arg("-Z").arg("on-broken-pipe=kill");
    }

    if target.is_some() {
        // The stage0 compiler has a special sysroot distinct from what we
        // actually downloaded, so we just always pass the `--sysroot` option,
        // unless one is already set.
        if !args.iter().any(|arg| arg == "--sysroot") {
            cmd.arg("--sysroot").arg(&sysroot);
        }

        // If we're compiling specifically the `panic_abort` crate then we pass
        // the `-C panic=abort` option. Note that we do not do this for any
        // other crate intentionally as this is the only crate for now that we
        // ship with panic=abort.
        //
        // This... is a bit of a hack how we detect this. Ideally this
        // information should be encoded in the crate I guess? Would likely
        // require an RFC amendment to RFC 1513, however.
        if crate_name == Some("panic_abort") {
            cmd.arg("-C").arg("panic=abort");
        }

        let crate_type = parse_value_from_args(&orig_args, "--crate-type");
        // `-Ztls-model=initial-exec` must not be applied to proc-macros, see
        // issue https://github.com/rust-lang/rust/issues/100530
        if env::var("RUSTC_TLS_MODEL_INITIAL_EXEC").is_ok()
            && crate_type != Some("proc-macro")
            && proc_macro_deps::CRATES.binary_search(&crate_name.unwrap_or_default()).is_err()
        {
            cmd.arg("-Ztls-model=initial-exec");
        }
    } else {
        // Find any host flags that were passed by bootstrap.
        // The flags are stored in a RUSTC_HOST_FLAGS variable, separated by spaces.
        if let Ok(flags) = std::env::var("RUSTC_HOST_FLAGS") {
            cmd.args(flags.split(' '));
        }
    }

    if let Ok(map) = env::var("RUSTC_DEBUGINFO_MAP") {
        cmd.arg("--remap-path-prefix").arg(&map);
    }
    // The remap flags for Cargo registry sources need to be passed after the remapping for the
    // Rust source code directory, to handle cases when $CARGO_HOME is inside the source directory.
    if let Ok(maps) = env::var("RUSTC_CARGO_REGISTRY_SRC_TO_REMAP") {
        for map in maps.split('\t') {
            cmd.arg("--remap-path-prefix").arg(map);
        }
    }

    // Force all crates compiled by this compiler to (a) be unstable and (b)
    // allow the `rustc_private` feature to link to other unstable crates
    // also in the sysroot. We also do this for host crates, since those
    // may be proc macros, in which case we might ship them.
    if env::var_os("RUSTC_FORCE_UNSTABLE").is_some() {
        cmd.arg("-Z").arg("force-unstable-if-unmarked");
    }

    // allow-features is handled from within this rustc wrapper because of
    // issues with build scripts. Some packages use build scripts to
    // dynamically detect if certain nightly features are available.
    // There are different ways this causes problems:
    //
    // * rustix runs `rustc` on a small test program to see if the feature is
    //   available (and sets a `cfg` if it is). It does not honor
    //   CARGO_ENCODED_RUSTFLAGS.
    // * proc-macro2 detects if `rustc -vV` says "nighty" or "dev" and enables
    //   nightly features. It will scan CARGO_ENCODED_RUSTFLAGS for
    //   -Zallow-features. Unfortunately CARGO_ENCODED_RUSTFLAGS is not set
    //   for build-dependencies when --target is used.
    //
    // The issues above means we can't just use RUSTFLAGS, and we can't use
    // `cargo -Zallow-features=…`. Passing it through here ensures that it
    // always gets set. Unfortunately that also means we need to enable more
    // features than we really want (like those for proc-macro2), but there
    // isn't much of a way around it.
    //
    // I think it is unfortunate that build scripts are doing this at all,
    // since changes to nightly features can cause crates to break even if the
    // user didn't want or care about the use of the nightly features. I think
    // nightly features should be opt-in only. Unfortunately the dynamic
    // checks are now too wide spread that we just need to deal with it.
    //
    // If you want to try to remove this, I suggest working with the crate
    // authors to remove the dynamic checking. Another option is to pursue
    // https://github.com/rust-lang/cargo/issues/11244 and
    // https://github.com/rust-lang/cargo/issues/4423, which will likely be
    // very difficult, but could help expose -Zallow-features into build
    // scripts so they could try to honor them.
    if let Ok(allow_features) = env::var("RUSTC_ALLOW_FEATURES") {
        cmd.arg(format!("-Zallow-features={allow_features}"));
    }

    if let Ok(flags) = env::var("MAGIC_EXTRA_RUSTFLAGS") {
        for flag in flags.split(' ') {
            cmd.arg(flag);
        }
    }

    if env::var_os("RUSTC_BOLT_LINK_FLAGS").is_some()
        && let Some("rustc_driver") = crate_name
    {
        cmd.arg("-Clink-args=-Wl,-q");
    }

    let is_test = args.iter().any(|a| a == "--test");
    if verbose > 2 {
        let rust_env_vars =
            env::vars().filter(|(k, _)| k.starts_with("RUST") || k.starts_with("CARGO"));
        let prefix = if is_test { "[RUSTC-SHIM] rustc --test" } else { "[RUSTC-SHIM] rustc" };
        let prefix = match crate_name {
            Some(crate_name) => format!("{prefix} {crate_name}"),
            None => prefix.to_string(),
        };
        for (i, (k, v)) in rust_env_vars.enumerate() {
            eprintln!("{prefix} env[{i}]: {k:?}={v:?}");
        }
        eprintln!("{} working directory: {}", prefix, env::current_dir().unwrap().display());
        eprintln!(
            "{} command: {:?}={:?} {:?}",
            prefix,
            dylib_path_var(),
            env::join_paths(&dylib_path).unwrap(),
            cmd,
        );
        eprintln!("{prefix} sysroot: {sysroot:?}");
        eprintln!("{prefix} libdir: {libdir:?}");
    }

    maybe_dump(format!("stage{stage}-rustc"), &cmd);

    let start = Instant::now();
    let (child, status) = {
        let errmsg = format!("\nFailed to run:\n{cmd:?}\n-------------");
        let mut child = cmd.spawn().expect(&errmsg);
        let status = child.wait().expect(&errmsg);
        (child, status)
    };

    if (env::var_os("RUSTC_PRINT_STEP_TIMINGS").is_some()
        || env::var_os("RUSTC_PRINT_STEP_RUSAGE").is_some())
        && let Some(crate_name) = crate_name
    {
        let dur = start.elapsed();
        // If the user requested resource usage data, then
        // include that in addition to the timing output.
        let rusage_data =
            env::var_os("RUSTC_PRINT_STEP_RUSAGE").and_then(|_| format_rusage_data(child));
        eprintln!(
            "[RUSTC-TIMING] {} test:{} {}.{:03}{}{}",
            crate_name,
            is_test,
            dur.as_secs(),
            dur.subsec_millis(),
            if rusage_data.is_some() { " " } else { "" },
            rusage_data.unwrap_or_default(),
        );
    }

    if status.success() {
        std::process::exit(0);
        // NOTE: everything below here is unreachable. do not put code that
        // should run on success, after this block.
    }
    if verbose > 0 {
        println!("\nDid not run successfully: {status}\n{cmd:?}\n-------------");
    }

    if let Some(mut on_fail) = on_fail {
        on_fail.status().expect("Could not run the on_fail command");
    }

    // Preserve the exit code. In case of signal, exit with 0xfe since it's
    // awkward to preserve this status in a cross-platform way.
    match status.code() {
        Some(i) => std::process::exit(i),
        None => {
            eprintln!("rustc exited with {status}");
            std::process::exit(0xfe);
        }
    }
}

#[cfg(all(not(unix), not(windows)))]
// In the future we can add this for more platforms
fn format_rusage_data(_child: Child) -> Option<String> {
    None
}

#[cfg(windows)]
fn format_rusage_data(child: Child) -> Option<String> {
    use std::os::windows::io::AsRawHandle;

    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows::Win32::System::Threading::GetProcessTimes;
    use windows::Win32::System::Time::FileTimeToSystemTime;

    let handle = HANDLE(child.as_raw_handle());

    let mut user_filetime = Default::default();
    let mut user_time = Default::default();
    let mut kernel_filetime = Default::default();
    let mut kernel_time = Default::default();
    let mut memory_counters = PROCESS_MEMORY_COUNTERS::default();
    let memory_counters_size = size_of_val(&memory_counters);

    unsafe {
        GetProcessTimes(
            handle,
            &mut Default::default(),
            &mut Default::default(),
            &mut kernel_filetime,
            &mut user_filetime,
        )
    }
    .ok()?;
    unsafe { FileTimeToSystemTime(&user_filetime, &mut user_time) }.ok()?;
    unsafe { FileTimeToSystemTime(&kernel_filetime, &mut kernel_time) }.ok()?;

    // Unlike on Linux with RUSAGE_CHILDREN, this will only return memory information for the process
    // with the given handle and none of that process's children.
    unsafe { K32GetProcessMemoryInfo(handle, &mut memory_counters, memory_counters_size as u32) }
        .ok()
        .ok()?;

    // Guide on interpreting these numbers:
    // https://docs.microsoft.com/en-us/windows/win32/psapi/process-memory-usage-information
    let peak_working_set = memory_counters.PeakWorkingSetSize / 1024;
    let peak_page_file = memory_counters.PeakPagefileUsage / 1024;
    let peak_paged_pool = memory_counters.QuotaPeakPagedPoolUsage / 1024;
    let peak_nonpaged_pool = memory_counters.QuotaPeakNonPagedPoolUsage / 1024;
    Some(format!(
        "user: {USER_SEC}.{USER_USEC:03} \
         sys: {SYS_SEC}.{SYS_USEC:03} \
         peak working set (kb): {PEAK_WORKING_SET} \
         peak page file usage (kb): {PEAK_PAGE_FILE} \
         peak paged pool usage (kb): {PEAK_PAGED_POOL} \
         peak non-paged pool usage (kb): {PEAK_NONPAGED_POOL} \
         page faults: {PAGE_FAULTS}",
        USER_SEC = user_time.wSecond + (user_time.wMinute * 60),
        USER_USEC = user_time.wMilliseconds,
        SYS_SEC = kernel_time.wSecond + (kernel_time.wMinute * 60),
        SYS_USEC = kernel_time.wMilliseconds,
        PEAK_WORKING_SET = peak_working_set,
        PEAK_PAGE_FILE = peak_page_file,
        PEAK_PAGED_POOL = peak_paged_pool,
        PEAK_NONPAGED_POOL = peak_nonpaged_pool,
        PAGE_FAULTS = memory_counters.PageFaultCount,
    ))
}

#[cfg(unix)]
/// Tries to build a string with human readable data for several of the rusage
/// fields. Note that we are focusing mainly on data that we believe to be
/// supplied on Linux (the `rusage` struct has other fields in it but they are
/// currently unsupported by Linux).
fn format_rusage_data(_child: Child) -> Option<String> {
    let rusage: libc::rusage = unsafe {
        let mut recv = std::mem::zeroed();
        // -1 is RUSAGE_CHILDREN, which means to get the rusage for all children
        // (and grandchildren, etc) processes that have respectively terminated
        // and been waited for.
        let retval = libc::getrusage(-1, &mut recv);
        if retval != 0 {
            return None;
        }
        recv
    };
    // Mac OS X reports the maxrss in bytes, not kb.
    let divisor = if env::consts::OS == "macos" { 1024 } else { 1 };
    let maxrss = (rusage.ru_maxrss + (divisor - 1)) / divisor;

    let mut init_str = format!(
        "user: {USER_SEC}.{USER_USEC:03} \
         sys: {SYS_SEC}.{SYS_USEC:03} \
         max rss (kb): {MAXRSS}",
        USER_SEC = rusage.ru_utime.tv_sec,
        USER_USEC = rusage.ru_utime.tv_usec,
        SYS_SEC = rusage.ru_stime.tv_sec,
        SYS_USEC = rusage.ru_stime.tv_usec,
        MAXRSS = maxrss
    );

    // The remaining rusage stats vary in platform support. So we treat
    // uniformly zero values in each category as "not worth printing", since it
    // either means no events of that type occurred, or that the platform
    // does not support it.

    let minflt = rusage.ru_minflt;
    let majflt = rusage.ru_majflt;
    if minflt != 0 || majflt != 0 {
        init_str.push_str(&format!(" page reclaims: {minflt} page faults: {majflt}"));
    }

    let inblock = rusage.ru_inblock;
    let oublock = rusage.ru_oublock;
    if inblock != 0 || oublock != 0 {
        init_str.push_str(&format!(" fs block inputs: {inblock} fs block outputs: {oublock}"));
    }

    let nvcsw = rusage.ru_nvcsw;
    let nivcsw = rusage.ru_nivcsw;
    if nvcsw != 0 || nivcsw != 0 {
        init_str.push_str(&format!(
            " voluntary ctxt switches: {nvcsw} involuntary ctxt switches: {nivcsw}"
        ));
    }

    Some(init_str)
}
