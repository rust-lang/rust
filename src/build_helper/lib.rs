pub mod channel;

use std::borrow::Cow;
use std::error::Error;
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::str;
use std::time::{SystemTime, UNIX_EPOCH};
use std::{env, fs};

/// A helper macro to `unwrap` a result except also print out details like:
///
/// * The file/line of the panic
/// * The expression that failed
/// * The error itself
///
/// This is currently used judiciously throughout the build system rather than
/// using a `Result` with `try!`, but this may change one day...
#[macro_export]
macro_rules! t {
    ($e:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with {}", stringify!($e), e),
        }
    };
    // it can show extra info in the second parameter
    ($e:expr, $extra:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with {} ({:?})", stringify!($e), e, $extra),
        }
    };
}

// Because Cargo adds the compiler's dylib path to our library search path, llvm-config may
// break: the dylib path for the compiler, as of this writing, contains a copy of the LLVM
// shared library, which means that when our freshly built llvm-config goes to load it's
// associated LLVM, it actually loads the compiler's LLVM. In particular when building the first
// compiler (i.e., in stage 0) that's a problem, as the compiler's LLVM is likely different from
// the one we want to use. As such, we restore the environment to what bootstrap saw. This isn't
// perfect -- we might actually want to see something from Cargo's added library paths -- but
// for now it works.
pub fn restore_library_path() {
    println!("cargo:rerun-if-env-changed=REAL_LIBRARY_PATH_VAR");
    println!("cargo:rerun-if-env-changed=REAL_LIBRARY_PATH");
    if let Some(key) = env::var_os("REAL_LIBRARY_PATH_VAR") {
        if let Some(env) = env::var_os("REAL_LIBRARY_PATH") {
            env::set_var(&key, &env);
        } else {
            env::remove_var(&key);
        }
    }
}

/// Run the command, printing what we are running.
pub fn run_verbose(cmd: &mut Command) {
    println!("running: {:?}", cmd);
    run(cmd);
}

pub fn run(cmd: &mut Command) {
    if !try_run(cmd) {
        std::process::exit(1);
    }
}

pub fn try_run(cmd: &mut Command) -> bool {
    let status = match cmd.status() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {:?}\nerror: {}", cmd, e)),
    };
    if !status.success() {
        println!(
            "\n\ncommand did not execute successfully: {:?}\n\
             expected success, got: {}\n\n",
            cmd, status
        );
    }
    status.success()
}

pub fn run_suppressed(cmd: &mut Command) {
    if !try_run_suppressed(cmd) {
        std::process::exit(1);
    }
}

pub fn try_run_suppressed(cmd: &mut Command) -> bool {
    let output = match cmd.output() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {:?}\nerror: {}", cmd, e)),
    };
    if !output.status.success() {
        println!(
            "\n\ncommand did not execute successfully: {:?}\n\
             expected success, got: {}\n\n\
             stdout ----\n{}\n\
             stderr ----\n{}\n\n",
            cmd,
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    output.status.success()
}

pub fn gnu_target(target: &str) -> &str {
    match target {
        "i686-pc-windows-msvc" => "i686-pc-win32",
        "x86_64-pc-windows-msvc" => "x86_64-pc-win32",
        "i686-pc-windows-gnu" => "i686-w64-mingw32",
        "x86_64-pc-windows-gnu" => "x86_64-w64-mingw32",
        s => s,
    }
}

pub fn make(host: &str) -> PathBuf {
    if host.contains("dragonfly")
        || host.contains("freebsd")
        || host.contains("netbsd")
        || host.contains("openbsd")
    {
        PathBuf::from("gmake")
    } else {
        PathBuf::from("make")
    }
}

pub fn output(cmd: &mut Command) -> String {
    let output = match cmd.stderr(Stdio::inherit()).output() {
        Ok(status) => status,
        Err(e) => fail(&format!("failed to execute command: {:?}\nerror: {}", cmd, e)),
    };
    if !output.status.success() {
        panic!(
            "command did not execute successfully: {:?}\n\
             expected success, got: {}",
            cmd, output.status
        );
    }
    String::from_utf8(output.stdout).unwrap()
}

pub fn rerun_if_changed_anything_in_dir(dir: &Path) {
    let mut stack = dir
        .read_dir()
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| &*e.file_name() != ".git")
        .collect::<Vec<_>>();
    while let Some(entry) = stack.pop() {
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            stack.extend(path.read_dir().unwrap().map(|e| e.unwrap()));
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// Returns the last-modified time for `path`, or zero if it doesn't exist.
pub fn mtime(path: &Path) -> SystemTime {
    fs::metadata(path).and_then(|f| f.modified()).unwrap_or(UNIX_EPOCH)
}

/// Returns `true` if `dst` is up to date given that the file or files in `src`
/// are used to generate it.
///
/// Uses last-modified time checks to verify this.
pub fn up_to_date(src: &Path, dst: &Path) -> bool {
    if !dst.exists() {
        return false;
    }
    let threshold = mtime(dst);
    let meta = match fs::metadata(src) {
        Ok(meta) => meta,
        Err(e) => panic!("source {:?} failed to get metadata: {}", src, e),
    };
    if meta.is_dir() {
        dir_up_to_date(src, threshold)
    } else {
        meta.modified().unwrap_or(UNIX_EPOCH) <= threshold
    }
}

fn dir_up_to_date(src: &Path, threshold: SystemTime) -> bool {
    t!(fs::read_dir(src)).map(|e| t!(e)).all(|e| {
        let meta = t!(e.metadata());
        if meta.is_dir() {
            dir_up_to_date(&e.path(), threshold)
        } else {
            meta.modified().unwrap_or(UNIX_EPOCH) < threshold
        }
    })
}

fn fail(s: &str) -> ! {
    println!("\n\n{}\n\n", s);
    std::process::exit(1);
}

#[derive(Default)]
pub struct RustcVersion {
    pub version: Option<String>,
    pub release: Option<String>,
    pub commit_hash: Option<String>,
    pub commit_date: Option<String>,
    pub is_stable_or_beta: bool,
}

fn get_rustc_version() -> Result<RustcVersion, Box<dyn Error>> {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));

    // check whether we are stable/beta and version in a single command for nightly compilers
    let output = Command::new(&rustc)
        .arg("-Z")
        .arg("verbose")
        .arg("-Vv")
        .env_remove("RUSTC_BOOTSTRAP")
        .output()?;
    let (is_stable_or_beta, version_lines) = if output.status.success() {
        (false, String::from_utf8(output.stdout)?)
    } else {
        let output = Command::new(&rustc).arg("-Vv").output()?;
        (true, String::from_utf8(output.stdout)?)
    };
    let mut version_lines = version_lines.lines();
    let mut rustc_version = RustcVersion::default();
    rustc_version.is_stable_or_beta = is_stable_or_beta;
    rustc_version.version =
        version_lines.next().and_then(|x| x.splitn(2, " ").skip(1).next().map(|x| x.into()));
    for line in version_lines {
        let mut line = line.splitn(2, ": ");
        let key = line.next();
        let value = line.next();
        match key {
            Some("release") => rustc_version.release = value.map(|x| x.into()),
            Some("commit-hash") => rustc_version.commit_hash = value.map(|x| x.into()),
            Some("commit-date") => rustc_version.commit_date = value.map(|x| x.into()),
            _ => {}
        };
    }

    Ok(rustc_version)
}

fn get_upstream_commit_hash(commit_ref: &str) -> Result<String, Box<dyn Error>> {
    Ok(String::from_utf8(
        Command::new("git").arg("merge-base").arg(commit_ref).arg("origin/master").output()?.stdout,
    )?
    .trim()
    .into())
}

enum Stage0Rustc {
    Beta { date: String },
    Stable { release: String },
}

struct Stage0 {
    rustc: Result<Stage0Rustc, &'static str>,
    #[allow(dead_code)]
    cargo: Option<String>,
}

fn get_stage0() -> Result<Stage0, Box<dyn Error>> {
    let mut date = None;
    let mut rustc = None;
    let mut cargo = None;
    for line in BufReader::new(File::open("../stage0.txt")?).lines() {
        let line = line?;
        if !line.starts_with("#") {
            let mut line = line.splitn(2, ": ");
            if let Some(key) = line.next() {
                let value = line.next();
                match key {
                    "date" => date = value.map(|x| x.into()),
                    "rustc" => rustc = value.map(|x| x.into()),
                    "cargo" => cargo = value.map(|x| x.into()),
                    _ => {}
                }
            }
        }
    }

    Ok(Stage0 {
        rustc: match rustc {
            Some(rustc) if rustc == "beta" => match date {
                Some(date) => Ok(Stage0Rustc::Beta { date }),
                _ => Err("src/stage0.txt does not contain a 'date = ...' line".into()),
            },
            Some(rustc) => Ok(Stage0Rustc::Stable { release: rustc }),
            _ => Err("src/stage0.txt does not contain a 'rustc = ...' line".into()),
        },
        cargo,
    })
}

const BUILD_SUGGESTIONS_HEADER: &'static str = "\
    This can result in compilation failures and inability to use a built rustc with the installed libstd or viceversa\n\
    \n\
    Suggestions for solving this issue if you are working on the Rust codebase:";

const BUILD_SUGGESTIONS_FOOTER: &'static str = "If you are just trying to install Rust from source or build a distributable package, then use `./x.py` instead as described in README.md\n";

const BUILD_SUGGESTIONS_NIGHTLY: &'static str = "\
    1. Download the latest nightly with `rustup install nightly`\n\
    2. Get the git commit hash of the nightly with `rustc +nightly -vV|sed -nre 's/^commit-hash: (.*)/\\1/p'`\n\
    3. Rebase your work onto it with `git rebase --onto $(rustc +nightly -vV|sed -nre 's/^commit-hash: (.*)/\\1/p') origin/master`\n\
    4. Build again with `cargo +nightly build`";

const BUILD_SUGGESTIONS_BETA: &'static str = "\
    1. Download the latest beta with `rustup install beta`\n\
    2. Build again with `cargo +beta build`";

fn build_suggestions_stable(release: &str) -> String {
    format!(
        "\
        1. Download Rust {} with `rustup install {}`\n\
        2. Build again with `cargo +{} build`",
        release, release, release
    )
}

fn build_suggestions(body: impl AsRef<str>) -> String {
    format!("{}\n{}\n\n{}", BUILD_SUGGESTIONS_HEADER, body.as_ref(), BUILD_SUGGESTIONS_FOOTER)
}

#[derive(Default)]
pub struct RustcInfo {
    pub version: RustcVersion,
    pub bootstrap: bool,
}

fn check_rustc(warn: bool, beta_warning: Option<&str>) -> Option<RustcInfo> {
    if env::var_os("RUSTC_STAGE").is_some() {
        // being built by x.py, skip all checks and adjustments
        None
    } else {
        macro_rules! warn {
            ($($arg:tt)*) => ({
                if warn {
                    println!("cargo:warning={}", format!($($arg)*).replace("\n", "\ncargo:warning="));
                }
            })
        }

        let rustc = get_rustc_version();
        if rustc.as_ref().map(|x| x.is_stable_or_beta).unwrap_or_default()
            && std::env::var_os("RUSTC_BOOTSTRAP").is_none()
        {
            // currently we can't set this automatically without also modifying the dependencies from crates.io that don't set it
            eprintln!(
                "ERROR: you must set RUSTC_BOOTSTRAP=1 when using a stable or beta toolchain"
            );
            if warn {
                eprintln!(
                    "\
                    You may want to use the latest nightly unless you want to bootstrap libstd\n\
                    {}",
                    build_suggestions(BUILD_SUGGESTIONS_NIGHTLY)
                );
            };
            std::process::exit(1);
        }
        match rustc {
            Ok(rustc) => {
                let rustc_upstream_commit_hash = if let Some(rustc_commit_hash) = &rustc.commit_hash
                {
                    get_upstream_commit_hash(&rustc_commit_hash).map_err(|err|
                        warn!("ERROR: unable to get upstream commit hash that the rustc git commit is based on: {}", err)
                    ).ok()
                } else {
                    warn!("ERROR: rustc -vV output did not include a commit hash");
                    None
                };
                let source_upstream_commit_hash = get_upstream_commit_hash("HEAD").map_err(|err|
                     warn!("ERROR: unable to get upstream commit hash that the current git tree is based on: {}", err)
                ).ok();
                let dev_ok = match (&rustc_upstream_commit_hash, &source_upstream_commit_hash) {
                    (Some(a), Some(b)) if a == b => true,
                    _ => false,
                };
                let bootstrap = match beta_warning {
                    _ if dev_ok => false,
                    Some(beta_warning) if rustc.is_stable_or_beta => {
                        warn!(
                            "\
                            {}\n\
                            {}",
                            beta_warning,
                            build_suggestions(BUILD_SUGGESTIONS_NIGHTLY)
                        );
                        false
                    }
                    _ => {
                        let stage0 = get_stage0()
                            .map_err(|err| warn!("ERROR: unable to parse src/stage0.txt: {}", err))
                            .ok();

                        let stage0rustc = stage0.and_then(|stage0| stage0.rustc.map_err(|err|
                            warn!("ERROR: unable to get required rustc version from src/stage0.txt: {}", err)
                        ).ok());

                        let bootstrap_ok = match (&stage0rustc, &rustc.release, &rustc.commit_date)
                        {
                            (
                                Some(Stage0Rustc::Beta { date }),
                                Some(rustc_release),
                                Some(rustc_commit_date),
                            ) if rustc_release.contains("-beta") => date <= rustc_commit_date,
                            (Some(Stage0Rustc::Stable { release }), Some(rustc_release), _) => {
                                release == rustc_release
                            }
                            _ => false,
                        };

                        if bootstrap_ok {
                            true
                        } else if rustc.is_stable_or_beta {
                            if rustc
                                .release
                                .as_ref()
                                .map(|x| x.contains("-beta"))
                                .unwrap_or_default()
                            {
                                match &stage0rustc {
                                    Some(Stage0Rustc::Stable { release }) => {
                                        warn!(
                                            "\
                                            IMPORTANT: you are building with a beta toolchain, but you must use stable version `{}` as specified in src/stage0.txt\n\
                                            {}",
                                            release,
                                            build_suggestions(build_suggestions_stable(release))
                                        );
                                    }
                                    _ => {
                                        let date = match stage0rustc {
                                            Some(Stage0Rustc::Beta { date }) => Some(date),
                                            _ => None,
                                        };
                                        warn!(
                                            "\
                                            IMPORTANT: the beta toolchain you are using to build is older than the required build specified in stage0.txt\n\
                                            The toolchain date is `{}`, but stage0.txt requires a date of `{}` or later\n\
                                            {}",
                                            rustc
                                                .commit_date
                                                .as_ref()
                                                .map(|x| x.as_str())
                                                .unwrap_or("<unknown>"),
                                            date.as_ref()
                                                .map(|x| x.as_str())
                                                .unwrap_or("<unknown>"),
                                            build_suggestions(BUILD_SUGGESTIONS_BETA)
                                        )
                                    }
                                }
                            } else {
                                match &stage0rustc {
                                    Some(Stage0Rustc::Stable { release }) => warn!(
                                        "\
                                            IMPORTANT: the stable toolchain you are using to build is different than the required build specified in stage0.txt\n\
                                            The toolchain release is `{}`, but stage0.txt requires release `{}`\n\
                                            {}",
                                        rustc
                                            .release
                                            .as_ref()
                                            .map(|x| x.as_str())
                                            .unwrap_or("<unknown>"),
                                        release,
                                        build_suggestions(build_suggestions_stable(release))
                                    ),
                                    _ => warn!(
                                        "\
                                            IMPORTANT: you are building with a stable toolchain, but you must use the most recent nightly, or the most recent beta if you want to bootstrap libstd\n\
                                            {}",
                                        build_suggestions(BUILD_SUGGESTIONS_NIGHTLY)
                                    ),
                                }
                            }
                            true
                        } else {
                            warn!(
                                "\
                                IMPORTANT: the toolchain you are using to build does not match the upstream git commit hash of your repository!\n\
                                The toolchain is based upon upstream git commit `{}`, but the repository is based upon upstream git commit `{}`\n\
                                {}",
                                rustc_upstream_commit_hash
                                    .as_ref()
                                    .map(|x| x.as_str())
                                    .unwrap_or("<unknown>"),
                                source_upstream_commit_hash
                                    .as_ref()
                                    .map(|x| x.as_str())
                                    .unwrap_or("<unknown>"),
                                build_suggestions(BUILD_SUGGESTIONS_NIGHTLY)
                            );
                            false
                        }
                    }
                };
                Some(RustcInfo { version: rustc, bootstrap })
            }
            Err(err) => {
                warn!(
                    "ERROR: unable to get rustc version: {}\n\
                    Unable to check if you are using the correct toolchain version to build\n\
                    {}",
                    err,
                    build_suggestions(BUILD_SUGGESTIONS_NIGHTLY)
                );
                Some(Default::default())
            }
        }
    }
}

pub fn build_stdlib(warn: bool) -> Option<RustcInfo> {
    if let Some(rustc) = check_rustc(warn, None) {
        if rustc.bootstrap {
            println!("cargo:rustc-cfg=bootstrap");
        }
        Some(rustc)
    } else {
        None
    }
}

pub fn set_env<'a>(key: &str, value: impl FnOnce() -> Option<Cow<'a, str>>) {
    if std::env::var_os(key).is_none() {
        if let Some(value) = value() {
            println!("cargo:rustc-env={}={}", key, value);
        }
    }
}

const RUSTC_BETA_WARNING: &'static str = "\
    IMPORTANT: you are trying to build rustc with a stable or beta toolchain, which is not supported.\n\
    Rustc is only guaranteed to successfully build with a nightly build matching the git commit hash of the repository, since it depends on the libstd with the same git version";

pub fn build_rustc() -> Option<RustcInfo> {
    check_rustc(true, Some(RUSTC_BETA_WARNING))
}
