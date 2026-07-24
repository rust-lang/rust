use std::path::PathBuf;

use crate::core::android;
use crate::core::builder::Builder;
use crate::core::config::DebuggerPath;
use crate::utils::exec::BootstrapCommand;

pub(crate) struct Gdb {
    pub(crate) gdb: PathBuf,
}

pub(crate) fn discover_gdb(
    builder: &Builder<'_>,
    android: Option<&android::Android>,
) -> Option<Gdb> {
    // If there's an explicitly-configured gdb, use that.
    match &builder.config.gdb {
        Some(DebuggerPath::Path(path)) => {
            return Some(Gdb { gdb: path.clone() });
        }
        Some(DebuggerPath::Discover) => {}
        None => return None,
    }

    // Otherwise, fall back to whatever gdb is sitting around in PATH.
    let gdb = match android {
        Some(android::Android { android_cross_path, .. }) => android_cross_path.join("bin/gdb"),
        None => PathBuf::from("gdb"),
    };

    // Check whether an ambient gdb exists, by running `gdb --version`.
    let output = {
        let mut gdb_command = BootstrapCommand::new(&gdb).allow_failure();
        gdb_command.arg("--version");
        gdb_command.run_capture(builder)
    };

    if output.is_success() { Some(Gdb { gdb }) } else { None }
}
