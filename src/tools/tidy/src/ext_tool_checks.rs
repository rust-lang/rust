//! Optional checks for file types other than Rust source
//!
//! Handles python tool version managment via a virtual environment in
//! `build/venv`.
//!
//! # Functional outline
//!
//! 1. Run tidy with an extra option: `--extra-checks=py,shell`,
//!    `--extra-checks=py:lint`, or similar. Optionally provide specific
//!    configuration after a double dash (`--extra-checks=py -- foo.py`)
//! 2. Build configuration based on args/environment:
//!    - Formatters by default are in check only mode
//!    - If in CI (TIDY_PRINT_DIFF=1 is set), check and print the diff
//!    - If `--bless` is provided, formatters may run
//!    - Pass any additional config after the `--`. If no files are specified,
//!      use a default.
//! 3. Print the output of the given command. If it fails and `TIDY_PRINT_DIFF`
//!    is set, rerun the tool to print a suggestion diff (for e.g. CI)

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{fmt, fs, io};

const MIN_PY_REV: (u32, u32) = (3, 9);
const MIN_PY_REV_STR: &str = "≥3.9";

/// Path to find the python executable within a virtual environment
#[cfg(target_os = "windows")]
const REL_PY_PATH: &[&str] = &["Scripts", "python3.exe"];
#[cfg(not(target_os = "windows"))]
const REL_PY_PATH: &[&str] = &["bin", "python3"];

const RUFF_CONFIG_PATH: &[&str] = &["src", "tools", "tidy", "config", "ruff.toml"];
const BLACK_CONFIG_PATH: &[&str] = &["src", "tools", "tidy", "config", "black.toml"];
/// Location within build directory
const RUFF_CACH_PATH: &[&str] = &["cache", "ruff_cache"];
const PIP_REQ_PATH: &[&str] = &["src", "tools", "tidy", "config", "requirements.txt"];

pub fn check(
    root_path: &Path,
    outdir: &Path,
    bless: bool,
    extra_checks: Option<&str>,
    pos_args: &[String],
    bad: &mut bool,
) {
    if let Err(e) = check_impl(root_path, outdir, bless, extra_checks, pos_args) {
        tidy_error!(bad, "{e}");
    }
}

fn check_impl(
    root_path: &Path,
    outdir: &Path,
    bless: bool,
    extra_checks: Option<&str>,
    pos_args: &[String],
) -> Result<(), Error> {
    let show_diff = std::env::var("TIDY_PRINT_DIFF")
        .map_or(false, |v| v.eq_ignore_ascii_case("true") || v == "1");

    // Split comma-separated args up
    let lint_args = match extra_checks {
        Some(s) => s.strip_prefix("--extra-checks=").unwrap().split(',').collect(),
        None => vec![],
    };

    let python_all = lint_args.contains(&"py");
    let python_lint = lint_args.contains(&"py:lint") || python_all;
    let python_fmt = lint_args.contains(&"py:fmt") || python_all;
    let shell_all = lint_args.contains(&"shell");
    let shell_lint = lint_args.contains(&"shell:lint") || shell_all;
    let cpp_all = lint_args.contains(&"cpp");
    let cpp_fmt = lint_args.contains(&"cpp:fmt") || cpp_all;

    let mut py_path = None;

    let (cfg_args, file_args): (Vec<_>, Vec<_>) = pos_args
        .iter()
        .map(OsStr::new)
        .partition(|arg| arg.to_str().is_some_and(|s| s.starts_with('-')));

    if python_lint || python_fmt || cpp_fmt {
        let venv_path = outdir.join("venv");
        let mut reqs_path = root_path.to_owned();
        reqs_path.extend(PIP_REQ_PATH);
        py_path = Some(get_or_create_venv(&venv_path, &reqs_path)?);
    }

    if python_lint {
        eprintln!("linting python files");
        let mut cfg_args_ruff = cfg_args.clone();
        let mut file_args_ruff = file_args.clone();

        let mut cfg_path = root_path.to_owned();
        cfg_path.extend(RUFF_CONFIG_PATH);
        let mut cache_dir = outdir.to_owned();
        cache_dir.extend(RUFF_CACH_PATH);

        cfg_args_ruff.extend([
            "--config".as_ref(),
            cfg_path.as_os_str(),
            "--cache-dir".as_ref(),
            cache_dir.as_os_str(),
        ]);

        if file_args_ruff.is_empty() {
            file_args_ruff.push(root_path.as_os_str());
        }

        let mut args = merge_args(&cfg_args_ruff, &file_args_ruff);
        args.insert(0, "check".as_ref());
        let res = py_runner(py_path.as_ref().unwrap(), true, None, "ruff", &args);

        if res.is_err() && show_diff {
            eprintln!("\npython linting failed! Printing diff suggestions:");

            args.insert(1, "--diff".as_ref());
            let _ = py_runner(py_path.as_ref().unwrap(), true, None, "ruff", &args);
        }
        // Rethrow error
        let _ = res?;
    }

    if python_fmt {
        let mut cfg_args_black = cfg_args.clone();
        let mut file_args_black = file_args.clone();

        if bless {
            eprintln!("formatting python files");
        } else {
            eprintln!("checking python file formatting");
            cfg_args_black.push("--check".as_ref());
        }

        let mut cfg_path = root_path.to_owned();
        cfg_path.extend(BLACK_CONFIG_PATH);

        cfg_args_black.extend(["--config".as_ref(), cfg_path.as_os_str()]);

        if file_args_black.is_empty() {
            file_args_black.push(root_path.as_os_str());
        }

        let mut args = merge_args(&cfg_args_black, &file_args_black);
        let res = py_runner(py_path.as_ref().unwrap(), true, None, "black", &args);

        if res.is_err() && show_diff {
            eprintln!("\npython formatting does not match! Printing diff:");

            args.insert(0, "--diff".as_ref());
            let _ = py_runner(py_path.as_ref().unwrap(), true, None, "black", &args);
        }
        // Rethrow error
        let _ = res?;
    }

    if cpp_fmt {
        let mut cfg_args_clang_format = cfg_args.clone();
        let mut file_args_clang_format = file_args.clone();
        let config_path = root_path.join(".clang-format");
        let config_file_arg = format!("file:{}", config_path.display());
        cfg_args_clang_format.extend(&["--style".as_ref(), config_file_arg.as_ref()]);
        if bless {
            eprintln!("formatting C++ files");
            cfg_args_clang_format.push("-i".as_ref());
        } else {
            eprintln!("checking C++ file formatting");
            cfg_args_clang_format.extend(&["--dry-run".as_ref(), "--Werror".as_ref()]);
        }
        let files;
        if file_args_clang_format.is_empty() {
            let llvm_wrapper = root_path.join("compiler/rustc_llvm/llvm-wrapper");
            files = find_with_extension(root_path, Some(llvm_wrapper.as_path()), &[
                OsStr::new("h"),
                OsStr::new("cpp"),
            ])?;
            file_args_clang_format.extend(files.iter().map(|p| p.as_os_str()));
        }
        let args = merge_args(&cfg_args_clang_format, &file_args_clang_format);
        let res = py_runner(py_path.as_ref().unwrap(), false, None, "clang-format", &args);

        if res.is_err() && show_diff {
            eprintln!("\nclang-format linting failed! Printing diff suggestions:");

            let mut cfg_args_clang_format_diff = cfg_args.clone();
            cfg_args_clang_format_diff.extend(&["--style".as_ref(), config_file_arg.as_ref()]);
            for file in file_args_clang_format {
                let mut formatted = String::new();
                let mut diff_args = cfg_args_clang_format_diff.clone();
                diff_args.push(file);
                let _ = py_runner(
                    py_path.as_ref().unwrap(),
                    false,
                    Some(&mut formatted),
                    "clang-format",
                    &diff_args,
                );
                if formatted.is_empty() {
                    eprintln!(
                        "failed to obtain the formatted content for '{}'",
                        file.to_string_lossy()
                    );
                    continue;
                }
                let actual = std::fs::read_to_string(file).unwrap_or_else(|e| {
                    panic!(
                        "failed to read the C++ file at '{}' due to '{e}'",
                        file.to_string_lossy()
                    )
                });
                if formatted != actual {
                    let diff = similar::TextDiff::from_lines(&actual, &formatted);
                    eprintln!(
                        "{}",
                        diff.unified_diff().context_radius(4).header(
                            &format!("{} (actual)", file.to_string_lossy()),
                            &format!("{} (formatted)", file.to_string_lossy())
                        )
                    );
                }
            }
        }
        // Rethrow error
        let _ = res?;
    }

    if shell_lint {
        eprintln!("linting shell files");

        let mut file_args_shc = file_args.clone();
        let files;
        if file_args_shc.is_empty() {
            files = find_with_extension(root_path, None, &[OsStr::new("sh")])?;
            file_args_shc.extend(files.iter().map(|p| p.as_os_str()));
        }

        shellcheck_runner(&merge_args(&cfg_args, &file_args_shc))?;
    }

    Ok(())
}

/// Helper to create `cfg1 cfg2 -- file1 file2` output
fn merge_args<'a>(cfg_args: &[&'a OsStr], file_args: &[&'a OsStr]) -> Vec<&'a OsStr> {
    let mut args = cfg_args.to_owned();
    args.push("--".as_ref());
    args.extend(file_args);
    args
}

/// Run a python command with given arguments. `py_path` should be a virtualenv.
///
/// Captures `stdout` to a string if provided, otherwise prints the output.
fn py_runner(
    py_path: &Path,
    as_module: bool,
    stdout: Option<&mut String>,
    bin: &'static str,
    args: &[&OsStr],
) -> Result<(), Error> {
    let mut cmd = Command::new(py_path);
    if as_module {
        cmd.arg("-m").arg(bin).args(args);
    } else {
        let bin_path = py_path.with_file_name(bin);
        cmd.arg(bin_path).args(args);
    }
    let status = if let Some(stdout) = stdout {
        let output = cmd.output()?;
        if let Ok(s) = std::str::from_utf8(&output.stdout) {
            stdout.push_str(s);
        }
        output.status
    } else {
        cmd.status()?
    };
    if status.success() { Ok(()) } else { Err(Error::FailedCheck(bin)) }
}

/// Create a virtuaenv at a given path if it doesn't already exist, or validate
/// the install if it does. Returns the path to that venv's python executable.
fn get_or_create_venv(venv_path: &Path, src_reqs_path: &Path) -> Result<PathBuf, Error> {
    let mut should_create = true;
    let dst_reqs_path = venv_path.join("requirements.txt");
    let mut py_path = venv_path.to_owned();
    py_path.extend(REL_PY_PATH);

    if let Ok(req) = fs::read_to_string(&dst_reqs_path) {
        if req == fs::read_to_string(src_reqs_path)? {
            // found existing environment
            should_create = false;
        } else {
            eprintln!("requirements.txt file mismatch, recreating environment");
        }
    }

    if should_create {
        eprintln!("removing old virtual environment");
        if venv_path.is_dir() {
            fs::remove_dir_all(venv_path).unwrap_or_else(|_| {
                panic!("failed to remove directory at {}", venv_path.display())
            });
        }
        create_venv_at_path(venv_path)?;
        install_requirements(&py_path, src_reqs_path, &dst_reqs_path)?;
    }

    verify_py_version(&py_path)?;
    Ok(py_path)
}

/// Attempt to create a virtualenv at this path. Cycles through all expected
/// valid python versions to find one that is installed.
fn create_venv_at_path(path: &Path) -> Result<(), Error> {
    /// Preferred python versions in order. Newest to oldest then current
    /// development versions
    const TRY_PY: &[&str] =
        &["python3.11", "python3.10", "python3.9", "python3", "python", "python3.12", "python3.13"];

    let mut sys_py = None;
    let mut found = Vec::new();

    for py in TRY_PY {
        match verify_py_version(Path::new(py)) {
            Ok(_) => {
                sys_py = Some(*py);
                break;
            }
            // Skip not found errors
            Err(Error::Io(e)) if e.kind() == io::ErrorKind::NotFound => (),
            // Skip insufficient version errors
            Err(Error::Version { installed, .. }) => found.push(installed),
            // just log and skip unrecognized errors
            Err(e) => eprintln!("note: error running '{py}': {e}"),
        }
    }

    let Some(sys_py) = sys_py else {
        let ret = if found.is_empty() {
            Error::MissingReq("python3", "python file checks", None)
        } else {
            found.sort();
            found.dedup();
            Error::Version {
                program: "python3",
                required: MIN_PY_REV_STR,
                installed: found.join(", "),
            }
        };
        return Err(ret);
    };

    eprintln!("creating virtual environment at '{}' using '{sys_py}'", path.display());
    let out = Command::new(sys_py).args(["-m", "virtualenv"]).arg(path).output().unwrap();

    if out.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&out.stderr);
    let err = if stderr.contains("No module named virtualenv") {
        Error::Generic(format!(
            "virtualenv not found: you may need to install it \
                               (`{sys_py} -m pip install virtualenv`)"
        ))
    } else {
        Error::Generic(format!(
            "failed to create venv at '{}' using {sys_py}: {stderr}",
            path.display()
        ))
    };
    Err(err)
}

/// Parse python's version output (`Python x.y.z`) and ensure we have a
/// suitable version.
fn verify_py_version(py_path: &Path) -> Result<(), Error> {
    let out = Command::new(py_path).arg("--version").output()?;
    let outstr = String::from_utf8_lossy(&out.stdout);
    let vers = outstr.trim().split_ascii_whitespace().nth(1).unwrap().trim();
    let mut vers_comps = vers.split('.');
    let major: u32 = vers_comps.next().unwrap().parse().unwrap();
    let minor: u32 = vers_comps.next().unwrap().parse().unwrap();

    if (major, minor) < MIN_PY_REV {
        Err(Error::Version {
            program: "python",
            required: MIN_PY_REV_STR,
            installed: vers.to_owned(),
        })
    } else {
        Ok(())
    }
}

fn install_requirements(
    py_path: &Path,
    src_reqs_path: &Path,
    dst_reqs_path: &Path,
) -> Result<(), Error> {
    let stat = Command::new(py_path)
        .args(["-m", "pip", "install", "--upgrade", "pip"])
        .status()
        .expect("failed to launch pip");
    if !stat.success() {
        return Err(Error::Generic(format!("pip install failed with status {stat}")));
    }

    let stat = Command::new(py_path)
        .args(["-m", "pip", "install", "--quiet", "--require-hashes", "-r"])
        .arg(src_reqs_path)
        .status()?;
    if !stat.success() {
        return Err(Error::Generic(format!(
            "failed to install requirements at {}",
            src_reqs_path.display()
        )));
    }
    fs::copy(src_reqs_path, dst_reqs_path)?;
    assert_eq!(
        fs::read_to_string(src_reqs_path).unwrap(),
        fs::read_to_string(dst_reqs_path).unwrap()
    );
    Ok(())
}

/// Check that shellcheck is installed then run it at the given path
fn shellcheck_runner(args: &[&OsStr]) -> Result<(), Error> {
    match Command::new("shellcheck").arg("--version").status() {
        Ok(_) => (),
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return Err(Error::MissingReq(
                "shellcheck",
                "shell file checks",
                Some(
                    "see <https://github.com/koalaman/shellcheck#installing> \
                    for installation instructions"
                        .to_owned(),
                ),
            ));
        }
        Err(e) => return Err(e.into()),
    }

    let status = Command::new("shellcheck").args(args).status()?;
    if status.success() { Ok(()) } else { Err(Error::FailedCheck("black")) }
}

/// Check git for tracked files matching an extension
fn find_with_extension(
    root_path: &Path,
    find_dir: Option<&Path>,
    extensions: &[&OsStr],
) -> Result<Vec<PathBuf>, Error> {
    // Untracked files show up for short status and are indicated with a leading `?`
    // -C changes git to be as if run from that directory
    let stat_output =
        Command::new("git").arg("-C").arg(root_path).args(["status", "--short"]).output()?.stdout;

    if String::from_utf8_lossy(&stat_output).lines().filter(|ln| ln.starts_with('?')).count() > 0 {
        eprintln!("found untracked files, ignoring");
    }

    let mut output = Vec::new();
    let binding = {
        let mut command = Command::new("git");
        command.arg("-C").arg(root_path).args(["ls-files"]);
        if let Some(find_dir) = find_dir {
            command.arg(find_dir);
        }
        command.output()?
    };
    let tracked = String::from_utf8_lossy(&binding.stdout);

    for line in tracked.lines() {
        let line = line.trim();
        let path = Path::new(line);

        let Some(ref extension) = path.extension() else {
            continue;
        };
        if extensions.contains(extension) {
            output.push(root_path.join(path));
        }
    }

    Ok(output)
}

#[derive(Debug)]
enum Error {
    Io(io::Error),
    /// a is required to run b. c is extra info
    MissingReq(&'static str, &'static str, Option<String>),
    /// Tool x failed the check
    FailedCheck(&'static str),
    /// Any message, just print it
    Generic(String),
    /// Installed but wrong version
    Version {
        program: &'static str,
        required: &'static str,
        installed: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingReq(a, b, ex) => {
                write!(
                    f,
                    "{a} is required to run {b} but it could not be located. Is it installed?"
                )?;
                if let Some(s) = ex {
                    write!(f, "\n{s}")?;
                };
                Ok(())
            }
            Self::Version { program, required, installed } => write!(
                f,
                "insufficient version of '{program}' to run external tools: \
                {required} required but found {installed}",
            ),
            Self::Generic(s) => f.write_str(s),
            Self::Io(e) => write!(f, "IO error: {e}"),
            Self::FailedCheck(s) => write!(f, "checks with external tool '{s}' failed"),
        }
    }
}

impl From<io::Error> for Error {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}
