//! Optional checks for file types other than Rust source
//!
//! Handles python tool version management via a virtual environment in
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
use std::str::FromStr;
use std::{fmt, fs, io};

use crate::CiInfo;

mod rustdoc_js;

const MIN_PY_REV: (u32, u32) = (3, 9);
const MIN_PY_REV_STR: &str = "≥3.9";

/// Path to find the python executable within a virtual environment
#[cfg(target_os = "windows")]
const REL_PY_PATH: &[&str] = &["Scripts", "python3.exe"];
#[cfg(not(target_os = "windows"))]
const REL_PY_PATH: &[&str] = &["bin", "python3"];

const RUFF_CONFIG_PATH: &[&str] = &["src", "tools", "tidy", "config", "ruff.toml"];
/// Location within build directory
const RUFF_CACHE_PATH: &[&str] = &["cache", "ruff_cache"];
const PIP_REQ_PATH: &[&str] = &["src", "tools", "tidy", "config", "requirements.txt"];

const SPELLCHECK_DIRS: &[&str] = &["compiler", "library", "src/bootstrap", "src/librustdoc"];

pub fn check(
    root_path: &Path,
    outdir: &Path,
    ci_info: &CiInfo,
    librustdoc_path: &Path,
    tools_path: &Path,
    npm: &Path,
    cargo: &Path,
    bless: bool,
    extra_checks: Option<&str>,
    pos_args: &[String],
    bad: &mut bool,
) {
    if let Err(e) = check_impl(
        root_path,
        outdir,
        ci_info,
        librustdoc_path,
        tools_path,
        npm,
        cargo,
        bless,
        extra_checks,
        pos_args,
    ) {
        tidy_error!(bad, "{e}");
    }
}

fn check_impl(
    root_path: &Path,
    outdir: &Path,
    ci_info: &CiInfo,
    librustdoc_path: &Path,
    tools_path: &Path,
    npm: &Path,
    cargo: &Path,
    bless: bool,
    extra_checks: Option<&str>,
    pos_args: &[String],
) -> Result<(), Error> {
    let show_diff =
        std::env::var("TIDY_PRINT_DIFF").is_ok_and(|v| v.eq_ignore_ascii_case("true") || v == "1");

    // Split comma-separated args up
    let mut lint_args = match extra_checks {
        Some(s) => s
            .strip_prefix("--extra-checks=")
            .unwrap()
            .split(',')
            .map(|s| {
                if s == "spellcheck:fix" {
                    eprintln!("warning: `spellcheck:fix` is no longer valid, use `--extra-checks=spellcheck --bless`");
                }
                (ExtraCheckArg::from_str(s), s)
            })
            .filter_map(|(res, src)| match res {
                Ok(arg) => {
                    Some(arg)
                }
                Err(err) => {
                    // only warn because before bad extra checks would be silently ignored.
                    eprintln!("warning: bad extra check argument {src:?}: {err:?}");
                    None
                }
            })
            .collect(),
        None => vec![],
    };
    if lint_args.iter().any(|ck| ck.auto) {
        crate::files_modified_batch_filter(ci_info, &mut lint_args, |ck, path| {
            ck.is_non_auto_or_matches(path)
        });
    }

    macro_rules! extra_check {
        ($lang:ident, $kind:ident) => {
            lint_args.iter().any(|arg| arg.matches(ExtraCheckLang::$lang, ExtraCheckKind::$kind))
        };
    }

    let rerun_with_bless = |mode: &str, action: &str| {
        if !bless {
            eprintln!("rerun tidy with `--extra-checks={mode} --bless` to {action}");
        }
    };

    let python_lint = extra_check!(Py, Lint);
    let python_fmt = extra_check!(Py, Fmt);
    let shell_lint = extra_check!(Shell, Lint);
    let cpp_fmt = extra_check!(Cpp, Fmt);
    let spellcheck = extra_check!(Spellcheck, None);
    let js_lint = extra_check!(Js, Lint);
    let js_typecheck = extra_check!(Js, Typecheck);

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
        let py_path = py_path.as_ref().unwrap();
        let args: &[&OsStr] = if bless {
            eprintln!("linting python files and applying suggestions");
            &["check".as_ref(), "--fix".as_ref()]
        } else {
            eprintln!("linting python files");
            &["check".as_ref()]
        };

        let res = run_ruff(root_path, outdir, py_path, &cfg_args, &file_args, args);

        if res.is_err() && show_diff && !bless {
            eprintln!("\npython linting failed! Printing diff suggestions:");

            let diff_res = run_ruff(
                root_path,
                outdir,
                py_path,
                &cfg_args,
                &file_args,
                &["check".as_ref(), "--diff".as_ref()],
            );
            // `ruff check --diff` will return status 0 if there are no suggestions.
            if diff_res.is_err() {
                rerun_with_bless("py:lint", "apply ruff suggestions");
            }
        }
        // Rethrow error
        res?;
    }

    if python_fmt {
        let mut args: Vec<&OsStr> = vec!["format".as_ref()];
        if bless {
            eprintln!("formatting python files");
        } else {
            eprintln!("checking python file formatting");
            args.push("--check".as_ref());
        }

        let py_path = py_path.as_ref().unwrap();
        let res = run_ruff(root_path, outdir, py_path, &cfg_args, &file_args, &args);

        if res.is_err() && !bless {
            if show_diff {
                eprintln!("\npython formatting does not match! Printing diff:");

                let _ = run_ruff(
                    root_path,
                    outdir,
                    py_path,
                    &cfg_args,
                    &file_args,
                    &["format".as_ref(), "--diff".as_ref()],
                );
            }
            rerun_with_bless("py:fmt", "reformat Python code");
        }

        // Rethrow error
        res?;
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
            files = find_with_extension(
                root_path,
                Some(llvm_wrapper.as_path()),
                &[OsStr::new("h"), OsStr::new("cpp")],
            )?;
            file_args_clang_format.extend(files.iter().map(|p| p.as_os_str()));
        }
        let args = merge_args(&cfg_args_clang_format, &file_args_clang_format);
        let res = py_runner(py_path.as_ref().unwrap(), false, None, "clang-format", &args);

        if res.is_err() && show_diff && !bless {
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
            rerun_with_bless("cpp:fmt", "reformat C++ code");
        }
        // Rethrow error
        res?;
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

    if spellcheck {
        let config_path = root_path.join("typos.toml");
        let mut args = vec!["-c", config_path.as_os_str().to_str().unwrap()];

        args.extend_from_slice(SPELLCHECK_DIRS);

        if bless {
            eprintln!("spellchecking files and fixing typos");
            args.push("--write-changes");
        } else {
            eprintln!("spellchecking files");
        }
        let res = spellcheck_runner(root_path, &outdir, &cargo, &args);
        if res.is_err() {
            rerun_with_bless("spellcheck", "fix typos");
        }
        res?;
    }

    if js_lint || js_typecheck {
        rustdoc_js::npm_install(root_path, outdir, npm)?;
    }

    if js_lint {
        if bless {
            eprintln!("linting javascript files");
        } else {
            eprintln!("linting javascript files and applying suggestions");
        }
        let res = rustdoc_js::lint(outdir, librustdoc_path, tools_path, bless);
        if res.is_err() {
            rerun_with_bless("js:lint", "apply eslint suggestions");
        }
        res?;
        rustdoc_js::es_check(outdir, librustdoc_path)?;
    }

    if js_typecheck {
        eprintln!("typechecking javascript files");
        rustdoc_js::typecheck(outdir, librustdoc_path)?;
    }

    Ok(())
}

fn run_ruff(
    root_path: &Path,
    outdir: &Path,
    py_path: &Path,
    cfg_args: &[&OsStr],
    file_args: &[&OsStr],
    ruff_args: &[&OsStr],
) -> Result<(), Error> {
    let mut cfg_args_ruff = cfg_args.to_vec();
    let mut file_args_ruff = file_args.to_vec();

    let mut cfg_path = root_path.to_owned();
    cfg_path.extend(RUFF_CONFIG_PATH);
    let mut cache_dir = outdir.to_owned();
    cache_dir.extend(RUFF_CACHE_PATH);

    cfg_args_ruff.extend([
        "--config".as_ref(),
        cfg_path.as_os_str(),
        "--cache-dir".as_ref(),
        cache_dir.as_os_str(),
    ]);

    if file_args_ruff.is_empty() {
        file_args_ruff.push(root_path.as_os_str());
    }

    let mut args: Vec<&OsStr> = ruff_args.to_vec();
    args.extend(merge_args(&cfg_args_ruff, &file_args_ruff));
    py_runner(py_path, true, None, "ruff", &args)
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
    const TRY_PY: &[&str] = &[
        "python3.13",
        "python3.12",
        "python3.11",
        "python3.10",
        "python3.9",
        "python3",
        "python",
        "python3.14",
    ];

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

    // First try venv, which should be packaged in the Python3 standard library.
    // If it is not available, try to create the virtual environment using the
    // virtualenv package.
    if try_create_venv(sys_py, path, "venv").is_ok() {
        return Ok(());
    }
    try_create_venv(sys_py, path, "virtualenv")
}

fn try_create_venv(python: &str, path: &Path, module: &str) -> Result<(), Error> {
    eprintln!(
        "creating virtual environment at '{}' using '{python}' and '{module}'",
        path.display()
    );
    let out = Command::new(python).args(["-m", module]).arg(path).output().unwrap();

    if out.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&out.stderr);
    let err = if stderr.contains(&format!("No module named {module}")) {
        Error::Generic(format!(
            r#"{module} not found: you may need to install it:
`{python} -m pip install {module}`
If you see an error about "externally managed environment" when running the above command,
either install `{module}` using your system package manager
(e.g. `sudo apt-get install {python}-{module}`) or create a virtual environment manually, install
`{module}` in it and then activate it before running tidy.
"#
        ))
    } else {
        Error::Generic(format!(
            "failed to create venv at '{}' using {python} -m {module}: {stderr}",
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
    if status.success() { Ok(()) } else { Err(Error::FailedCheck("shellcheck")) }
}

/// Ensure that spellchecker is installed then run it at the given path
fn spellcheck_runner(
    src_root: &Path,
    outdir: &Path,
    cargo: &Path,
    args: &[&str],
) -> Result<(), Error> {
    let bin_path =
        crate::ensure_version_or_cargo_install(outdir, cargo, "typos-cli", "typos", "1.34.0")?;
    match Command::new(bin_path).current_dir(src_root).args(args).status() {
        Ok(status) => {
            if status.success() {
                Ok(())
            } else {
                Err(Error::FailedCheck("typos"))
            }
        }
        Err(err) => Err(Error::Generic(format!("failed to run typos tool: {err:?}"))),
    }
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

#[derive(Debug)]
enum ExtraCheckParseError {
    #[allow(dead_code, reason = "shown through Debug")]
    UnknownKind(String),
    #[allow(dead_code)]
    UnknownLang(String),
    UnsupportedKindForLang,
    /// Too many `:`
    TooManyParts,
    /// Tried to parse the empty string
    Empty,
    /// `auto` specified without lang part.
    AutoRequiresLang,
}

struct ExtraCheckArg {
    auto: bool,
    lang: ExtraCheckLang,
    /// None = run all extra checks for the given lang
    kind: Option<ExtraCheckKind>,
}

impl ExtraCheckArg {
    fn matches(&self, lang: ExtraCheckLang, kind: ExtraCheckKind) -> bool {
        self.lang == lang && self.kind.map(|k| k == kind).unwrap_or(true)
    }

    /// Returns `false` if this is an auto arg and the passed filename does not trigger the auto rule
    fn is_non_auto_or_matches(&self, filepath: &str) -> bool {
        if !self.auto {
            return true;
        }
        let exts: &[&str] = match self.lang {
            ExtraCheckLang::Py => &[".py"],
            ExtraCheckLang::Cpp => &[".cpp"],
            ExtraCheckLang::Shell => &[".sh"],
            ExtraCheckLang::Js => &[".js", ".ts"],
            ExtraCheckLang::Spellcheck => {
                if SPELLCHECK_DIRS.iter().any(|dir| Path::new(filepath).starts_with(dir)) {
                    return true;
                }
                &[]
            }
        };
        exts.iter().any(|ext| filepath.ends_with(ext))
    }

    fn has_supported_kind(&self) -> bool {
        let Some(kind) = self.kind else {
            // "run all extra checks" mode is supported for all languages.
            return true;
        };
        use ExtraCheckKind::*;
        let supported_kinds: &[_] = match self.lang {
            ExtraCheckLang::Py => &[Fmt, Lint],
            ExtraCheckLang::Cpp => &[Fmt],
            ExtraCheckLang::Shell => &[Lint],
            ExtraCheckLang::Spellcheck => &[],
            ExtraCheckLang::Js => &[Lint, Typecheck],
        };
        supported_kinds.contains(&kind)
    }
}

impl FromStr for ExtraCheckArg {
    type Err = ExtraCheckParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut auto = false;
        let mut parts = s.split(':');
        let Some(mut first) = parts.next() else {
            return Err(ExtraCheckParseError::Empty);
        };
        if first == "auto" {
            let Some(part) = parts.next() else {
                return Err(ExtraCheckParseError::AutoRequiresLang);
            };
            auto = true;
            first = part;
        }
        let second = parts.next();
        if parts.next().is_some() {
            return Err(ExtraCheckParseError::TooManyParts);
        }
        let arg = Self { auto, lang: first.parse()?, kind: second.map(|s| s.parse()).transpose()? };
        if !arg.has_supported_kind() {
            return Err(ExtraCheckParseError::UnsupportedKindForLang);
        }

        Ok(arg)
    }
}

#[derive(PartialEq, Copy, Clone)]
enum ExtraCheckLang {
    Py,
    Shell,
    Cpp,
    Spellcheck,
    Js,
}

impl FromStr for ExtraCheckLang {
    type Err = ExtraCheckParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "py" => Self::Py,
            "shell" => Self::Shell,
            "cpp" => Self::Cpp,
            "spellcheck" => Self::Spellcheck,
            "js" => Self::Js,
            _ => return Err(ExtraCheckParseError::UnknownLang(s.to_string())),
        })
    }
}

#[derive(PartialEq, Copy, Clone)]
enum ExtraCheckKind {
    Lint,
    Fmt,
    Typecheck,
    /// Never parsed, but used as a placeholder for
    /// langs that never have a specific kind.
    None,
}

impl FromStr for ExtraCheckKind {
    type Err = ExtraCheckParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "lint" => Self::Lint,
            "fmt" => Self::Fmt,
            "typecheck" => Self::Typecheck,
            _ => return Err(ExtraCheckParseError::UnknownKind(s.to_string())),
        })
    }
}
