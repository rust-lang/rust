use std::path::PathBuf;

use crate::core::builder::Builder;
use crate::utils::exec::command;

pub(crate) struct Lldb {
    pub(crate) lldb_version: String,
    pub(crate) lldb_python_dir: String,
}

pub(crate) fn discover_lldb(builder: &Builder<'_>) -> Option<Lldb> {
    // FIXME(#148361): We probably should not be picking up whatever arbitrary
    // lldb happens to be in the user's path, and instead require some kind of
    // explicit opt-in or configuration.
    let lldb_exe = builder.config.lldb.clone().unwrap_or_else(|| PathBuf::from("lldb"));

    let lldb_version = command(&lldb_exe)
        .allow_failure()
        .arg("--version")
        .run_capture(builder)
        .stdout_if_ok()
        .and_then(|v| if v.trim().is_empty() { None } else { Some(v) })?;

    let lldb_python_dir = command(&lldb_exe)
        .allow_failure()
        .arg("-P")
        .run_capture_stdout(builder)
        .stdout_if_ok()
        .map(|p| p.lines().next().expect("lldb Python dir not found").to_string())?;

    Some(Lldb { lldb_version, lldb_python_dir })
}
