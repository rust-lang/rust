use std::env;
use std::process::Command;
use std::sync::Arc;

use camino::{Utf8Path, Utf8PathBuf};

use crate::common::{Config, Debugger};

pub(crate) fn configure_cdb(config: &Config) -> Option<Arc<Config>> {
    config.cdb.as_ref()?;

    Some(Arc::new(Config { debugger: Some(Debugger::Cdb), ..config.clone() }))
}

pub(crate) fn configure_gdb(config: &Config) -> Option<Arc<Config>> {
    config.gdb_version?;

    if config.matches_env("msvc") {
        return None;
    }

    if config.remote_test_client.is_some() && !config.target.contains("android") {
        println!(
            "WARNING: debuginfo tests are not available when \
             testing with remote"
        );
        return None;
    }

    if config.target.contains("android") {
        println!(
            "{} debug-info test uses tcp 5039 port.\
             please reserve it",
            config.target
        );

        // android debug-info test uses remote debugger so, we test 1 thread
        // at once as they're all sharing the same TCP port to communicate
        // over.
        //
        // we should figure out how to lift this restriction! (run them all
        // on different ports allocated dynamically).
        //
        // SAFETY: at this point we are still single-threaded.
        unsafe { env::set_var("RUST_TEST_THREADS", "1") };
    }

    Some(Arc::new(Config { debugger: Some(Debugger::Gdb), ..config.clone() }))
}

pub(crate) fn configure_lldb(config: &Config) -> Option<Arc<Config>> {
    config.lldb_python_dir.as_ref()?;

    Some(Arc::new(Config { debugger: Some(Debugger::Lldb), ..config.clone() }))
}

/// Returns `true` if the given target is an Android target for the
/// purposes of GDB testing.
pub(crate) fn is_android_gdb_target(target: &str) -> bool {
    matches!(
        &target[..],
        "arm-linux-androideabi" | "armv7-linux-androideabi" | "aarch64-linux-android"
    )
}

/// Returns `true` if the given target is a MSVC target for the purposes of CDB testing.
fn is_pc_windows_msvc_target(target: &str) -> bool {
    target.ends_with("-pc-windows-msvc")
}

/// FIXME: this is very questionable...
fn find_cdb(target: &str) -> Option<Utf8PathBuf> {
    if !(cfg!(windows) && is_pc_windows_msvc_target(target)) {
        return None;
    }

    let pf86 = Utf8PathBuf::from_path_buf(
        env::var_os("ProgramFiles(x86)").or_else(|| env::var_os("ProgramFiles"))?.into(),
    )
    .unwrap();
    let cdb_arch = if cfg!(target_arch = "x86") {
        "x86"
    } else if cfg!(target_arch = "x86_64") {
        "x64"
    } else if cfg!(target_arch = "aarch64") {
        "arm64"
    } else if cfg!(target_arch = "arm") {
        "arm"
    } else {
        return None; // No compatible CDB.exe in the Windows 10 SDK
    };

    let mut path = pf86;
    path.push(r"Windows Kits\10\Debuggers"); // We could check 8.1 etc. too?
    path.push(cdb_arch);
    path.push(r"cdb.exe");

    if !path.exists() {
        return None;
    }

    Some(path)
}

/// Returns Path to CDB
pub(crate) fn analyze_cdb(
    cdb: Option<String>,
    target: &str,
) -> (Option<Utf8PathBuf>, Option<[u16; 4]>) {
    let cdb = cdb.map(Utf8PathBuf::from).or_else(|| find_cdb(target));

    let mut version = None;
    if let Some(cdb) = cdb.as_ref() {
        if let Ok(output) = Command::new(cdb).arg("/version").output() {
            if let Some(first_line) = String::from_utf8_lossy(&output.stdout).lines().next() {
                version = extract_cdb_version(&first_line);
            }
        }
    }

    (cdb, version)
}

pub(crate) fn extract_cdb_version(full_version_line: &str) -> Option<[u16; 4]> {
    // Example full_version_line: "cdb version 10.0.18362.1"
    let version = full_version_line.rsplit(' ').next()?;
    let mut components = version.split('.');
    let major: u16 = components.next().unwrap().parse().unwrap();
    let minor: u16 = components.next().unwrap().parse().unwrap();
    let patch: u16 = components.next().unwrap_or("0").parse().unwrap();
    let build: u16 = components.next().unwrap_or("0").parse().unwrap();
    Some([major, minor, patch, build])
}

/// Returns (Path to GDB, GDB Version)
pub(crate) fn analyze_gdb(
    gdb: Option<String>,
    target: &str,
    android_cross_path: &Utf8Path,
) -> (Option<String>, Option<u32>) {
    #[cfg(not(windows))]
    const GDB_FALLBACK: &str = "gdb";
    #[cfg(windows)]
    const GDB_FALLBACK: &str = "gdb.exe";

    let fallback_gdb = || {
        if is_android_gdb_target(target) {
            let mut gdb_path = android_cross_path.to_string();
            gdb_path.push_str("/bin/gdb");
            gdb_path
        } else {
            GDB_FALLBACK.to_owned()
        }
    };

    let gdb = match gdb {
        None => fallback_gdb(),
        Some(ref s) if s.is_empty() => fallback_gdb(), // may be empty if configure found no gdb
        Some(ref s) => s.to_owned(),
    };

    let mut version_line = None;
    if let Ok(output) = Command::new(&gdb).arg("--version").output() {
        if let Some(first_line) = String::from_utf8_lossy(&output.stdout).lines().next() {
            version_line = Some(first_line.to_string());
        }
    }

    let version = match version_line {
        Some(line) => extract_gdb_version(&line),
        None => return (None, None),
    };

    (Some(gdb), version)
}

pub(crate) fn extract_gdb_version(full_version_line: &str) -> Option<u32> {
    let full_version_line = full_version_line.trim();

    // GDB versions look like this: "major.minor.patch?.yyyymmdd?", with both
    // of the ? sections being optional

    // We will parse up to 3 digits for each component, ignoring the date

    // We skip text in parentheses.  This avoids accidentally parsing
    // the openSUSE version, which looks like:
    //  GNU gdb (GDB; openSUSE Leap 15.0) 8.1
    // This particular form is documented in the GNU coding standards:
    // https://www.gnu.org/prep/standards/html_node/_002d_002dversion.html#g_t_002d_002dversion

    let unbracketed_part = full_version_line.split('[').next().unwrap();
    let mut splits = unbracketed_part.trim_end().rsplit(' ');
    let version_string = splits.next().unwrap();

    let mut splits = version_string.split('.');
    let major = splits.next().unwrap();
    let minor = splits.next().unwrap();
    let patch = splits.next();

    let major: u32 = major.parse().unwrap();
    let (minor, patch): (u32, u32) = match minor.find(not_a_digit) {
        None => {
            let minor = minor.parse().unwrap();
            let patch: u32 = match patch {
                Some(patch) => match patch.find(not_a_digit) {
                    None => patch.parse().unwrap(),
                    Some(idx) if idx > 3 => 0,
                    Some(idx) => patch[..idx].parse().unwrap(),
                },
                None => 0,
            };
            (minor, patch)
        }
        // There is no patch version after minor-date (e.g. "4-2012").
        Some(idx) => {
            let minor = minor[..idx].parse().unwrap();
            (minor, 0)
        }
    };

    Some(((major * 1000) + minor) * 1000 + patch)
}

/// Returns LLDB version
pub(crate) fn extract_lldb_version(full_version_line: &str) -> Option<u32> {
    // Extract the major LLDB version from the given version string.
    // LLDB version strings are different for Apple and non-Apple platforms.
    // The Apple variant looks like this:
    //
    // LLDB-179.5 (older versions)
    // lldb-300.2.51 (new versions)
    //
    // We are only interested in the major version number, so this function
    // will return `Some(179)` and `Some(300)` respectively.
    //
    // Upstream versions look like:
    // lldb version 6.0.1
    //
    // There doesn't seem to be a way to correlate the Apple version
    // with the upstream version, and since the tests were originally
    // written against Apple versions, we make a fake Apple version by
    // multiplying the first number by 100. This is a hack.

    let full_version_line = full_version_line.trim();

    if let Some(apple_ver) =
        full_version_line.strip_prefix("LLDB-").or_else(|| full_version_line.strip_prefix("lldb-"))
    {
        if let Some(idx) = apple_ver.find(not_a_digit) {
            let version: u32 = apple_ver[..idx].parse().unwrap();
            return Some(version);
        }
    } else if let Some(lldb_ver) = full_version_line.strip_prefix("lldb version ") {
        if let Some(idx) = lldb_ver.find(not_a_digit) {
            let version: u32 = lldb_ver[..idx].parse().ok()?;
            return Some(version * 100);
        }
    }
    None
}

fn not_a_digit(c: char) -> bool {
    !c.is_ascii_digit()
}
