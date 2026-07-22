use std::path::PathBuf;

use crate::core::builder::Builder;
use crate::utils::exec::command;

pub(crate) struct Lldb {
    pub(crate) lldb_exe: PathBuf,
    pub(crate) lldb_version: String,
}

pub(crate) fn discover_lldb(builder: &Builder<'_>) -> Option<Lldb> {
    // FIXME(#148361): We probably should not be picking up whatever arbitrary
    // lldb happens to be in the user's path, and instead require some kind of
    // explicit opt-in or configuration.
    let lldb_exe = builder.config.lldb.clone().unwrap_or_else(|| PathBuf::from("lldb"));

    let mut cmd = command(&lldb_exe);
    cmd.arg("--version");

    // If a path to a LLDB binary was provided, it has to exist and return some version, to avoid
    // silent failures.
    let explicitly_set_lldb = builder.config.lldb.is_some();
    if !explicitly_set_lldb {
        cmd = cmd.allow_failure();
    }
    let lldb_version = cmd.run_capture(builder).stdout_if_ok().filter(|v| !v.trim().is_empty())?;

    Some(Lldb { lldb_exe, lldb_version })
}
