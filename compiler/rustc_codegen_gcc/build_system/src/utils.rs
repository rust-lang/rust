use std::collections::HashMap;
use std::ffi::OsStr;
use std::fmt::Debug;
use std::fs;
use std::path::Path;
use std::process::{Command, ExitStatus, Output};

fn get_command_inner(
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
) -> Result<(), String> {
    if exit_status.success() {
        Ok(())
    } else {
        Err(format!(
            "Command `{}`{} exited with status {:?}",
            input
                .iter()
                .map(|s| s.as_ref().to_str().unwrap())
                .collect::<Vec<_>>()
                .join(" "),
            cwd.map(|cwd| format!(" (running in folder `{}`)", cwd.display()))
                .unwrap_or_default(),
            exit_status.code(),
        ))
    }
}

fn command_error<D: Debug>(input: &[&dyn AsRef<OsStr>], cwd: &Option<&Path>, error: D) -> String {
    format!(
        "Command `{}`{} failed to run: {error:?}",
        input
            .iter()
            .map(|s| s.as_ref().to_str().unwrap())
            .collect::<Vec<_>>()
            .join(" "),
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
    let output = get_command_inner(input, cwd, env)
        .output()
        .map_err(|e| command_error(input, &cwd, e))?;
    check_exit_status(input, cwd, output.status)?;
    Ok(output)
}

pub fn run_command_with_output(
    input: &[&dyn AsRef<OsStr>],
    cwd: Option<&Path>,
) -> Result<(), String> {
    let exit_status = get_command_inner(input, cwd, None)
        .spawn()
        .map_err(|e| command_error(input, &cwd, e))?
        .wait()
        .map_err(|e| command_error(input, &cwd, e))?;
    check_exit_status(input, cwd, exit_status)?;
    Ok(())
}

pub fn run_command_with_output_and_env(
    input: &[&dyn AsRef<OsStr>],
    cwd: Option<&Path>,
    env: Option<&HashMap<String, String>>,
) -> Result<(), String> {
    let exit_status = get_command_inner(input, cwd, env)
        .spawn()
        .map_err(|e| command_error(input, &cwd, e))?
        .wait()
        .map_err(|e| command_error(input, &cwd, e))?;
    check_exit_status(input, cwd, exit_status)?;
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
    let name = std::str::from_utf8(&output.stdout)
        .unwrap_or("")
        .trim()
        .to_string();
    if !name.is_empty() {
        Ok(name)
    } else {
        Err("Failed to retrieve the OS name".to_string())
    }
}

pub fn get_rustc_host_triple() -> Result<String, String> {
    let output = run_command(&[&"rustc", &"-vV"], None)?;
    let content = std::str::from_utf8(&output.stdout).unwrap_or("");

    for line in content.split('\n').map(|line| line.trim()) {
        if !line.starts_with("host:") {
            continue;
        }
        return Ok(line.split(':').nth(1).unwrap().trim().to_string());
    }
    Err("Cannot find host triple".to_string())
}

pub fn get_gcc_path() -> Result<String, String> {
    let content = match fs::read_to_string("gcc_path") {
        Ok(content) => content,
        Err(_) => {
            return Err(
                "Please put the path to your custom build of libgccjit in the file \
                   `gcc_path`, see Readme.md for details"
                    .into(),
            )
        }
    };
    match content
        .split('\n')
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .next()
    {
        Some(gcc_path) => {
            let path = Path::new(gcc_path);
            if !path.exists() {
                Err(format!(
                    "Path `{}` contained in the `gcc_path` file doesn't exist",
                    gcc_path,
                ))
            } else {
                Ok(gcc_path.into())
            }
        }
        None => Err("No path found in `gcc_path` file".into()),
    }
}

pub struct CloneResult {
    pub ran_clone: bool,
    pub repo_name: String,
}

pub fn git_clone(to_clone: &str, dest: Option<&Path>) -> Result<CloneResult, String> {
    let repo_name = to_clone.split('/').last().unwrap();
    let repo_name = match repo_name.strip_suffix(".git") {
        Some(n) => n.to_string(),
        None => repo_name.to_string(),
    };

    let dest = dest
        .map(|dest| dest.join(&repo_name))
        .unwrap_or_else(|| Path::new(&repo_name).into());
    if dest.is_dir() {
        return Ok(CloneResult {
            ran_clone: false,
            repo_name,
        });
    }

    run_command_with_output(&[&"git", &"clone", &to_clone, &dest], None)?;
    Ok(CloneResult {
        ran_clone: true,
        repo_name,
    })
}

pub fn walk_dir<P, D, F>(dir: P, mut dir_cb: D, mut file_cb: F) -> Result<(), String>
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
        } else {
            file_cb(&entry_path)?;
        }
    }
    Ok(())
}
