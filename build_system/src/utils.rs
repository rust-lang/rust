use std::ffi::OsStr;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};

fn run_command_inner(input: &[&dyn AsRef<OsStr>], cwd: Option<&Path>) -> Command {
    let (cmd, args) = match input {
        [] => panic!("empty command"),
        [cmd, args @ ..] => (cmd, args),
    };
    let mut command = Command::new(cmd);
    command.args(args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    command
}

pub fn run_command(input: &[&dyn AsRef<OsStr>], cwd: Option<&Path>) -> Result<Output, String> {
    run_command_inner(input, cwd).output()
        .map_err(|e| format!(
            "Command `{}` failed to run: {e:?}",
            input.iter()
                .map(|s| s.as_ref().to_str().unwrap())
                .collect::<Vec<_>>()
                .join(" "),
        ))
}

pub fn run_command_with_output(
    input: &[&dyn AsRef<OsStr>],
    cwd: Option<&Path>,
) -> Result<(), String> {
    run_command_inner(input, cwd).spawn()
        .map_err(|e| format!(
            "Command `{}` failed to run: {e:?}",
            input.iter()
                .map(|s| s.as_ref().to_str().unwrap())
                .collect::<Vec<_>>()
                .join(" "),
        ))?
        .wait()
        .map_err(|e| format!(
            "Failed to wait for command `{}` to run: {e:?}",
            input.iter()
                .map(|s| s.as_ref().to_str().unwrap())
                .collect::<Vec<_>>()
                .join(" "),
        ))?;
    Ok(())
}

pub fn cargo_install(to_install: &str) -> Result<(), String> {
    let output = run_command(&[&"cargo", &"install", &"--list"], None)?;

    let to_install = format!("{to_install} ");
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
        .any(|line| line.ends_with(':') && line.starts_with(&to_install))
    {
        return Ok(());
    }
    run_command(&[&"cargo", &"install", &to_install], None)?;
    Ok(())
}

pub struct CloneResult {
    pub ran_clone: bool,
    pub repo_name: String,
}

pub fn git_clone(to_clone: &str, dest: Option<&Path>) -> Result<CloneResult, String> {
    let repo_name = to_clone.split('/').last().unwrap();
    let repo_name = match repo_name.strip_suffix(".git") {
        Some(n) => n.to_owned(),
        None => repo_name.to_owned(),
    };

    let dest = dest.unwrap_or_else(|| Path::new(&repo_name));
    if dest.is_dir() {
        return Ok(CloneResult { ran_clone: false, repo_name });
    }

    run_command(&[&"git", &"clone", &to_clone, &dest], None)?;
    Ok(CloneResult { ran_clone: true, repo_name })
}

pub fn walk_dir<P, D, F>(dir: P, dir_cb: D, file_cb: F) -> Result<(), String>
where
    P: AsRef<Path>,
    D: Fn(&Path) -> Result<(), String>,
    F: Fn(&Path) -> Result<(), String>,
{
    let dir = dir.as_ref();
    for entry in fs::read_dir(dir).map_err(|e| format!("Failed to read dir `{}`: {e:?}", dir.display()))? {
        let entry = entry.map_err(|e| format!("Failed to read entry in `{}`: {e:?}", dir.display()))?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            dir_cb(&entry_path)?;
        } else {
            file_cb(&entry_path)?;
        }
    }
    Ok(())
}
