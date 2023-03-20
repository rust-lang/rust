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

include!("../dylib_util.rs");

use std::env;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::str::FromStr;
use std::time::Instant;

fn main() {
    let args = env::args_os().skip(1).collect::<Vec<_>>();
    let arg = |name| args.windows(2).find(|args| args[0] == name).and_then(|args| args[1].to_str());

    // Detect whether or not we're a build script depending on whether --target
    // is passed (a bit janky...)
    let target = arg("--target");
    let version = args.iter().find(|w| &**w == "-vV");

    let verbose = match env::var("RUSTC_VERBOSE") {
        Ok(s) => usize::from_str(&s).expect("RUSTC_VERBOSE should be an integer"),
        Err(_) => 0,
    };

    // Use a different compiler for build scripts, since there may not yet be a
    // libstd for the real compiler to use. However, if Cargo is attempting to
    // determine the version of the compiler, the real compiler needs to be
    // used. Currently, these two states are differentiated based on whether
    // --target and -vV is/isn't passed.
    let (rustc, libdir) = if target.is_none() && version.is_none() {
        ("RUSTC_SNAPSHOT", "RUSTC_SNAPSHOT_LIBDIR")
    } else {
        ("RUSTC_REAL", "RUSTC_LIBDIR")
    };
    let stage = env::var("RUSTC_STAGE").expect("RUSTC_STAGE was not set");
    let sysroot = env::var_os("RUSTC_SYSROOT").expect("RUSTC_SYSROOT was not set");
    let on_fail = env::var_os("RUSTC_ON_FAIL").map(Command::new);

    let rustc = env::var_os(rustc).unwrap_or_else(|| panic!("{:?} was not set", rustc));
    let libdir = env::var_os(libdir).unwrap_or_else(|| panic!("{:?} was not set", libdir));
    let mut dylib_path = dylib_path();
    dylib_path.insert(0, PathBuf::from(&libdir));

    let mut cmd = Command::new(rustc);
    cmd.args(&args).env(dylib_path_var(), env::join_paths(&dylib_path).unwrap());

    // Get the name of the crate we're compiling, if any.
    let crate_name = arg("--crate-name");

    if let Some(crate_name) = crate_name {
        if let Some(target) = env::var_os("RUSTC_TIME") {
            if target == "all"
                || target.into_string().unwrap().split(',').any(|c| c.trim() == crate_name)
            {
                cmd.arg("-Ztime-passes");
            }
        }
    }

    // Print backtrace in case of ICE
    if env::var("RUSTC_BACKTRACE_ON_ICE").is_ok() && env::var("RUST_BACKTRACE").is_err() {
        cmd.env("RUST_BACKTRACE", "1");
    }

    if let Ok(lint_flags) = env::var("RUSTC_LINT_FLAGS") {
        cmd.args(lint_flags.split_whitespace());
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

        // `-Ztls-model=initial-exec` must not be applied to proc-macros, see
        // issue https://github.com/rust-lang/rust/issues/100530
        if env::var("RUSTC_TLS_MODEL_INITIAL_EXEC").is_ok()
            && arg("--crate-type") != Some("proc-macro")
            && !matches!(crate_name, Some("proc_macro2" | "quote" | "syn" | "synstructure"))
        {
            cmd.arg("-Ztls-model=initial-exec");
        }
    } else {
        // FIXME(rust-lang/cargo#5754) we shouldn't be using special env vars
        // here, but rather Cargo should know what flags to pass rustc itself.

        // Override linker if necessary.
        if let Ok(host_linker) = env::var("RUSTC_HOST_LINKER") {
            cmd.arg(format!("-Clinker={}", host_linker));
        }
        if env::var_os("RUSTC_HOST_FUSE_LD_LLD").is_some() {
            cmd.arg("-Clink-args=-fuse-ld=lld");
        }

        if let Ok(s) = env::var("RUSTC_HOST_CRT_STATIC") {
            if s == "true" {
                cmd.arg("-C").arg("target-feature=+crt-static");
            }
            if s == "false" {
                cmd.arg("-C").arg("target-feature=-crt-static");
            }
        }

        // Cargo doesn't pass RUSTFLAGS to proc_macros:
        // https://github.com/rust-lang/cargo/issues/4423
        // Thus, if we are on stage 0, we explicitly set `--cfg=bootstrap`.
        // We also declare that the flag is expected, which we need to do to not
        // get warnings about it being unexpected.
        if stage == "0" {
            cmd.arg("--cfg=bootstrap");
        }
        cmd.arg("-Zunstable-options");
        cmd.arg("--check-cfg=values(bootstrap)");
    }

    if let Ok(map) = env::var("RUSTC_DEBUGINFO_MAP") {
        cmd.arg("--remap-path-prefix").arg(&map);
    }

    // Force all crates compiled by this compiler to (a) be unstable and (b)
    // allow the `rustc_private` feature to link to other unstable crates
    // also in the sysroot. We also do this for host crates, since those
    // may be proc macros, in which case we might ship them.
    if env::var_os("RUSTC_FORCE_UNSTABLE").is_some() && (stage != "0" || target.is_some()) {
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

    let is_test = args.iter().any(|a| a == "--test");
    if verbose > 2 {
        let rust_env_vars =
            env::vars().filter(|(k, _)| k.starts_with("RUST") || k.starts_with("CARGO"));
        let prefix = if is_test { "[RUSTC-SHIM] rustc --test" } else { "[RUSTC-SHIM] rustc" };
        let prefix = match crate_name {
            Some(crate_name) => format!("{} {}", prefix, crate_name),
            None => prefix.to_string(),
        };
        for (i, (k, v)) in rust_env_vars.enumerate() {
            eprintln!("{} env[{}]: {:?}={:?}", prefix, i, k, v);
        }
        eprintln!("{} working directory: {}", prefix, env::current_dir().unwrap().display());
        eprintln!(
            "{} command: {:?}={:?} {:?}",
            prefix,
            dylib_path_var(),
            env::join_paths(&dylib_path).unwrap(),
            cmd,
        );
        eprintln!("{} sysroot: {:?}", prefix, sysroot);
        eprintln!("{} libdir: {:?}", prefix, libdir);
    }

    let start = Instant::now();
    let (child, status) = {
        let errmsg = format!("\nFailed to run:\n{:?}\n-------------", cmd);
        let mut child = cmd.spawn().expect(&errmsg);
        let status = child.wait().expect(&errmsg);
        (child, status)
    };

    if env::var_os("RUSTC_PRINT_STEP_TIMINGS").is_some()
        || env::var_os("RUSTC_PRINT_STEP_RUSAGE").is_some()
    {
        if let Some(crate_name) = crate_name {
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
                rusage_data.unwrap_or(String::new()),
            );
        }
    }

    if status.success() {
        std::process::exit(0);
        // note: everything below here is unreachable. do not put code that
        // should run on success, after this block.
    }
    if verbose > 0 {
        println!("\nDid not run successfully: {}\n{:?}\n-------------", status, cmd);
    }

    if let Some(mut on_fail) = on_fail {
        on_fail.status().expect("Could not run the on_fail command");
    }

    // Preserve the exit code. In case of signal, exit with 0xfe since it's
    // awkward to preserve this status in a cross-platform way.
    match status.code() {
        Some(i) => std::process::exit(i),
        None => {
            eprintln!("rustc exited with {}", status);
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

    use windows::{
        Win32::Foundation::HANDLE,
        Win32::System::ProcessStatus::{
            K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
        },
        Win32::System::Threading::GetProcessTimes,
        Win32::System::Time::FileTimeToSystemTime,
    };

    let handle = HANDLE(child.as_raw_handle() as isize);

    let mut user_filetime = Default::default();
    let mut user_time = Default::default();
    let mut kernel_filetime = Default::default();
    let mut kernel_time = Default::default();
    let mut memory_counters = PROCESS_MEMORY_COUNTERS::default();

    unsafe {
        GetProcessTimes(
            handle,
            &mut Default::default(),
            &mut Default::default(),
            &mut kernel_filetime,
            &mut user_filetime,
        )
    }
    .ok()
    .ok()?;
    unsafe { FileTimeToSystemTime(&user_filetime, &mut user_time) }.ok().ok()?;
    unsafe { FileTimeToSystemTime(&kernel_filetime, &mut kernel_time) }.ok().ok()?;

    // Unlike on Linux with RUSAGE_CHILDREN, this will only return memory information for the process
    // with the given handle and none of that process's children.
    unsafe {
        K32GetProcessMemoryInfo(
            handle,
            &mut memory_counters,
            std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        )
    }
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
        init_str.push_str(&format!(" page reclaims: {} page faults: {}", minflt, majflt));
    }

    let inblock = rusage.ru_inblock;
    let oublock = rusage.ru_oublock;
    if inblock != 0 || oublock != 0 {
        init_str.push_str(&format!(" fs block inputs: {} fs block outputs: {}", inblock, oublock));
    }

    let nvcsw = rusage.ru_nvcsw;
    let nivcsw = rusage.ru_nivcsw;
    if nvcsw != 0 || nivcsw != 0 {
        init_str.push_str(&format!(
            " voluntary ctxt switches: {} involuntary ctxt switches: {}",
            nvcsw, nivcsw
        ));
    }

    return Some(init_str);
}
