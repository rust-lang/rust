use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::Path;
use std::process::Command;

pub fn write_file(file_path: impl AsRef<Path>, content: &str) -> Result<(), String> {
    std::fs::write(&file_path, content).map_err(|error| {
        format!(
            "Failed to create empty `{}` file: {error:?}",
            file_path.as_ref().display(),
        )
    })
}

pub fn run_command_with_env<I, S>(
    bin: &str,
    args: I,
    current_dir: &str,
    env: &HashMap<&str, &str>,
) -> Result<(), String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let exit_status = Command::new(bin)
        .args(args)
        .envs(env)
        .current_dir(current_dir)
        .spawn()
        .map_err(|error| format!("Failed to spawn command `{bin}`: {error:?}"))?
        .wait()
        .map_err(|error| format!("Failed to wait command `{bin}`: {error:?}"))?;
    if exit_status.success() {
        Ok(())
    } else {
        Err(format!("Command `{bin}` failed"))
    }
}

pub fn run_command<I, S>(bin: &str, args: I, current_dir: &str) -> Result<(), String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    run_command_with_env(bin, args, current_dir, &HashMap::new())
}

pub struct CommandOutput {
    pub output: String,
    pub exited_successfully: bool,
}

pub fn run_command_with_output_and_env<I, S>(
    bin: &str,
    args: I,
    current_dir: &str,
    env: &HashMap<&str, &str>,
) -> Result<CommandOutput, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let cmd_output = Command::new(bin)
        .args(args)
        .envs(env)
        .current_dir(current_dir)
        .output()
        .map_err(|error| format!("Failed to spawn command `{bin}`: {error:?}"))?;
    let mut output = String::from_utf8_lossy(&cmd_output.stdout).into_owned();
    output.push_str(&String::from_utf8_lossy(&cmd_output.stderr));
    Ok(CommandOutput {
        output,
        exited_successfully: cmd_output.status.success(),
    })
}

pub fn run_command_with_output<I, S>(
    bin: &str,
    args: I,
    current_dir: &str,
) -> Result<CommandOutput, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    run_command_with_output_and_env(bin, args, current_dir, &HashMap::new())
}
