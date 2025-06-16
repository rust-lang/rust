use std::collections::HashMap;
use std::ffi::OsStr;
use std::fmt::Debug;
use std::fs;
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output};

fn exec_command(
    input: &[&dyn AsRef<OsStr>],
    cwd: Option<&Path>,
    env: Option<&HashMap<String, String>>,
) -> Result<ExitStatus, String> {
    let status = get_command_inner(input, cwd, env)
        .spawn()
        .map_err(|e| command_error(input, &cwd, e))?
        .wait()
        .map_err(|e| command_error(input, &cwd, e))?;
    #[cfg(unix)]
    {
        if let Some(signal) = status.signal() {
            // In case the signal didn't kill the current process.
            return Err(command_error(input, &cwd, format!("Process received signal {signal}")));
        }
    }
    Ok(status)
}

pub(crate) fn get_command_inner(
    input: &[&dyn AsRef<OsStr>],
    cwd: Option<&Path>,
    env: Option<&HashMap<String, String>>,
) -> Command {
    let (cmd, args) = match input {
        [] => panic!("empty command"),
        [cmd, args @ ..] => (cmd, args),
    };
    let mut command = Command::new(cmd);
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    if let Some(env) = env {
        command.envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    }
    command
}

fn check_exit_status(
    input: &[&dyn AsRef<OsStr>],
    cwd: Option<&Path>,
    exit_status: ExitStatus,
    output: Option<&Output>,
    show_err: bool,
) -> Result<(), String> {
    if exit_status.success() {
        return Ok(());
    }
    let mut error = format!(
        "Command `{}`{} exited with status {:?}",
        input.iter().map(|s| s.as_ref().to_str().unwrap()).collect::<Vec<_>>().join(" "),
        cwd.map(|cwd| format!(" (running in folder `{}`)", cwd.display())).unwrap_or_default(),
        exit_status.code()
    );
    let input = input.iter().map(|i| i.as_ref()).collect::<Vec<&OsStr>>();
    if show_err {
        eprintln!("Command `{input:?}` failed");
    }
    if let Some(output) = output {
        let stdout = String::from_utf8_lossy(&output.stdout);
        if !stdout.is_empty() {
            error.push_str("\n==== STDOUT ====\n");
            error.push_str(&stdout);
        }
        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.is_empty() {
            error.push_str("\n==== STDERR ====\n");
            error.push_str(&stderr);
        }
    }
    Err(error)
}

fn command_error<D: Debug>(input: &[&dyn AsRef<OsStr>], cwd: &Option<&Path>, error: D) -> String {
    format!(
        "Command `{}`{} failed to run: {error:?}",
        input.iter().map(|s| s.as_ref().to_str().unwrap()).collect::<Vec<_>>().join(" "),
        cwd.as_ref()
            .map(|cwd| format!(" (running in folder `{}`)", cwd.display(),))
            .unwrap_or_default(),
    )
}

pub fn run_command(input: &[&dyn AsRef<OsStr>], cwd: Option<&Path>) -> Result<Output, String> {
    run_command_with_env(input, cwd, None)
}

pub fn run_command_with_env(
    input: &[&dyn AsRef<OsStr>],
    cwd: Option<&Path>,
    env: Option<&HashMap<String, String>>,
) -> Result<Output, String> {
    let output =
        get_command_inner(input, cwd, env).output().map_err(|e| command_error(input, &cwd, e))?;
    check_exit_status(input, cwd, output.status, Some(&output), true)?;
    Ok(output)
}

pub fn run_command_with_output(
    input: &[&dyn AsRef<OsStr>],
    cwd: Option<&Path>,
) -> Result<(), String> {
    let exit_status = exec_command(input, cwd, None)?;
    check_exit_status(input, cwd, exit_status, None, true)?;
    Ok(())
}

pub fn run_command_with_output_and_env(
    input: &[&dyn AsRef<OsStr>],
    cwd: Option<&Path>,
    env: Option<&HashMap<String, String>>,
) -> Result<(), String> {
    let exit_status = exec_command(input, cwd, env)?;
    check_exit_status(input, cwd, exit_status, None, true)?;
    Ok(())
}

#[cfg(not(unix))]
pub fn run_command_with_output_and_env_no_err(
    input: &[&dyn AsRef<OsStr>],
    cwd: Option<&Path>,
    env: Option<&HashMap<String, String>>,
) -> Result<(), String> {
    let exit_status = exec_command(input, cwd, env)?;
    check_exit_status(input, cwd, exit_status, None, false)?;
    Ok(())
}

pub fn cargo_install(to_install: &str) -> Result<(), String> {
    let output = run_command(&[&"cargo", &"install", &"--list"], None)?;

    let to_install_needle = format!("{to_install} ");
    // cargo install --list returns something like this:
    //
    // mdbook-toc v0.8.0:
    //     mdbook-toc
    // rust-reduce v0.1.0:
    //     rust-reduce
    //
    // We are only interested into the command name so we only look for lines ending with `:`.
    if String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .any(|line| line.ends_with(':') && line.starts_with(&to_install_needle))
    {
        return Ok(());
    }
    // We voluntarily ignore this error.
    if run_command_with_output(&[&"cargo", &"install", &to_install], None).is_err() {
        println!("Skipping installation of `{to_install}`");
    }
    Ok(())
}

pub fn get_os_name() -> Result<String, String> {
    let output = run_command(&[&"uname"], None)?;
    let name = std::str::from_utf8(&output.stdout).unwrap_or("").trim().to_string();
    if !name.is_empty() { Ok(name) } else { Err("Failed to retrieve the OS name".to_string()) }
}

#[derive(Default, PartialEq)]
pub struct RustcVersionInfo {
    pub short: String,
    pub version: String,
    pub host: Option<String>,
    pub commit_hash: Option<String>,
    pub commit_date: Option<String>,
}

pub fn rustc_toolchain_version_info(toolchain: &str) -> Result<RustcVersionInfo, String> {
    rustc_version_info_inner(None, Some(toolchain))
}

pub fn rustc_version_info(rustc: Option<&str>) -> Result<RustcVersionInfo, String> {
    rustc_version_info_inner(rustc, None)
}

fn rustc_version_info_inner(
    rustc: Option<&str>,
    toolchain: Option<&str>,
) -> Result<RustcVersionInfo, String> {
    let output = if let Some(toolchain) = toolchain {
        run_command(&[&rustc.unwrap_or("rustc"), &toolchain, &"-vV"], None)
    } else {
        run_command(&[&rustc.unwrap_or("rustc"), &"-vV"], None)
    }?;
    let content = std::str::from_utf8(&output.stdout).unwrap_or("");

    let mut info = RustcVersionInfo::default();
    let mut lines = content.split('\n');
    info.short = match lines.next() {
        Some(s) => s.to_string(),
        None => return Err("failed to retrieve rustc version".to_string()),
    };

    for line in lines.map(|line| line.trim()) {
        match line.split_once(':') {
            Some(("host", data)) => info.host = Some(data.trim().to_string()),
            Some(("release", data)) => info.version = data.trim().to_string(),
            Some(("commit-hash", data)) => info.commit_hash = Some(data.trim().to_string()),
            Some(("commit-date", data)) => info.commit_date = Some(data.trim().to_string()),
            _ => {}
        }
    }
    if info.version.is_empty() {
        Err("failed to retrieve rustc version".to_string())
    } else {
        Ok(info)
    }
}

pub fn get_toolchain() -> Result<String, String> {
    let content = match fs::read_to_string("rust-toolchain") {
        Ok(content) => content,
        Err(_) => return Err("No `rust-toolchain` file found".to_string()),
    };
    match content
        .split('\n')
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .filter_map(|line| {
            if !line.starts_with("channel") {
                return None;
            }
            line.split('"').nth(1)
        })
        .next()
    {
        Some(toolchain) => Ok(toolchain.to_string()),
        None => Err("Couldn't find `channel` in `rust-toolchain` file".to_string()),
    }
}

pub struct CloneResult {
    pub ran_clone: bool,
    pub repo_name: String,
    pub repo_dir: String,
}

fn git_clone_inner(
    to_clone: &str,
    dest: &Path,
    shallow_clone: bool,
    repo_name: String,
) -> Result<CloneResult, String> {
    if dest.is_dir() {
        return Ok(CloneResult {
            ran_clone: false,
            repo_name,
            repo_dir: dest.display().to_string(),
        });
    }

    let mut command: Vec<&dyn AsRef<OsStr>> = vec![&"git", &"clone", &to_clone, &dest];
    if shallow_clone {
        command.push(&"--depth");
        command.push(&"1");
    }
    run_command_with_output(&command, None)?;
    Ok(CloneResult { ran_clone: true, repo_name, repo_dir: dest.display().to_string() })
}

fn get_repo_name(url: &str) -> String {
    let repo_name = url.split('/').next_back().unwrap();
    match repo_name.strip_suffix(".git") {
        Some(n) => n.to_string(),
        None => repo_name.to_string(),
    }
}

pub fn git_clone(
    to_clone: &str,
    dest: Option<&Path>,
    shallow_clone: bool,
) -> Result<CloneResult, String> {
    let repo_name = get_repo_name(to_clone);
    let tmp: PathBuf;

    let dest = match dest {
        Some(dest) => dest,
        None => {
            tmp = repo_name.clone().into();
            &tmp
        }
    };
    git_clone_inner(to_clone, dest, shallow_clone, repo_name)
}

pub fn create_dir<P: AsRef<Path>>(path: P) -> Result<(), String> {
    fs::create_dir_all(&path).map_err(|error| {
        format!("Failed to create directory `{}`: {:?}", path.as_ref().display(), error)
    })
}

pub fn copy_file<F: AsRef<Path>, T: AsRef<Path>>(from: F, to: T) -> Result<(), String> {
    fs::copy(&from, &to)
        .map_err(|error| {
            format!(
                "Failed to copy file `{}` into `{}`: {:?}",
                from.as_ref().display(),
                to.as_ref().display(),
                error
            )
        })
        .map(|_| ())
}

/// This function differs from `git_clone` in how it handles *where* the repository will be cloned.
/// In `git_clone`, it is cloned in the provided path. In this function, the path you provide is
/// the parent folder. So if you pass "a" as folder and try to clone "b.git", it will be cloned into
/// `a/b`.
pub fn git_clone_root_dir(
    to_clone: &str,
    dest_parent_dir: &Path,
    shallow_clone: bool,
) -> Result<CloneResult, String> {
    let repo_name = get_repo_name(to_clone);

    git_clone_inner(to_clone, &dest_parent_dir.join(&repo_name), shallow_clone, repo_name)
}

pub fn walk_dir<P, D, F>(
    dir: P,
    dir_cb: &mut D,
    file_cb: &mut F,
    recursive: bool,
) -> Result<(), String>
where
    P: AsRef<Path>,
    D: FnMut(&Path) -> Result<(), String>,
    F: FnMut(&Path) -> Result<(), String>,
{
    let dir = dir.as_ref();
    for entry in fs::read_dir(dir)
        .map_err(|error| format!("Failed to read dir `{}`: {:?}", dir.display(), error))?
    {
        let entry = entry
            .map_err(|error| format!("Failed to read entry in `{}`: {:?}", dir.display(), error))?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            dir_cb(&entry_path)?;
            if recursive {
                walk_dir(entry_path, dir_cb, file_cb, recursive)?; // Recursive call
            }
        } else {
            file_cb(&entry_path)?;
        }
    }
    Ok(())
}

pub fn split_args(args: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut start = 0;
    let args = args.trim();
    let mut iter = args.char_indices().peekable();

    while let Some((pos, c)) = iter.next() {
        if c == ' ' {
            out.push(args[start..pos].to_string());
            let mut found_start = false;
            while let Some((pos, c)) = iter.peek() {
                if *c != ' ' {
                    start = *pos;
                    found_start = true;
                    break;
                } else {
                    iter.next();
                }
            }
            if !found_start {
                return Ok(out);
            }
        } else if c == '"' || c == '\'' {
            let end = c;
            let mut found_end = false;
            while let Some((_, c)) = iter.next() {
                if c == end {
                    found_end = true;
                    break;
                } else if c == '\\' {
                    // We skip the escaped character.
                    iter.next();
                }
            }
            if !found_end {
                return Err(format!("Didn't find `{}` at the end of `{}`", end, &args[start..]));
            }
        } else if c == '\\' {
            // We skip the escaped character.
            iter.next();
        }
    }
    let s = args[start..].trim();
    if !s.is_empty() {
        out.push(s.to_string());
    }
    Ok(out)
}

pub fn remove_file<P: AsRef<Path> + ?Sized>(file_path: &P) -> Result<(), String> {
    std::fs::remove_file(file_path).map_err(|error| {
        format!("Failed to remove `{}`: {:?}", file_path.as_ref().display(), error)
    })
}

pub fn create_symlink<P: AsRef<Path>, Q: AsRef<Path>>(original: P, link: Q) -> Result<(), String> {
    #[cfg(windows)]
    let symlink = std::os::windows::fs::symlink_file;
    #[cfg(not(windows))]
    let symlink = std::os::unix::fs::symlink;

    symlink(&original, &link).map_err(|err| {
        format!(
            "failed to create a symlink `{}` to `{}`: {:?}",
            original.as_ref().display(),
            link.as_ref().display(),
            err,
        )
    })
}

pub fn get_sysroot_dir() -> PathBuf {
    Path::new(crate::BUILD_DIR).join("build_sysroot")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_args() {
        // Missing `"` at the end.
        assert!(split_args("\"tada").is_err());
        // Missing `'` at the end.
        assert!(split_args("\'tada").is_err());

        assert_eq!(
            split_args("a \"b\" c"),
            Ok(vec!["a".to_string(), "\"b\"".to_string(), "c".to_string()])
        );
        // Trailing whitespace characters.
        assert_eq!(
            split_args("    a    \"b\" c    "),
            Ok(vec!["a".to_string(), "\"b\"".to_string(), "c".to_string()])
        );
    }
}
