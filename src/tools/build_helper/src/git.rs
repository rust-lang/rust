use std::process::Command;

/// Finds the remote for rust-lang/rust.
/// For example for these remotes it will return `upstream`.
/// ```text
/// origin  https://github.com/Nilstrieb/rust.git (fetch)
/// origin  https://github.com/Nilstrieb/rust.git (push)
/// upstream        https://github.com/rust-lang/rust (fetch)
/// upstream        https://github.com/rust-lang/rust (push)
/// ```
pub fn get_rust_lang_rust_remote() -> Result<String, String> {
    let mut git = Command::new("git");
    git.args(["config", "--local", "--get-regex", "remote\\..*\\.url"]);

    let output = git.output().map_err(|err| format!("{err:?}"))?;
    if !output.status.success() {
        return Err("failed to execute git config command".to_owned());
    }

    let stdout = String::from_utf8(output.stdout).map_err(|err| format!("{err:?}"))?;

    let rust_lang_remote = stdout
        .lines()
        .find(|remote| remote.contains("rust-lang"))
        .ok_or_else(|| "rust-lang/rust remote not found".to_owned())?;

    let remote_name =
        rust_lang_remote.split('.').nth(1).ok_or_else(|| "remote name not found".to_owned())?;
    Ok(remote_name.into())
}
