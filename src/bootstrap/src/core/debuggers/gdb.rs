use std::borrow::Cow;
use std::path::Path;

use crate::core::android;
use crate::core::builder::Builder;
use crate::utils::exec::BootstrapCommand;

pub(crate) struct Gdb<'a> {
    pub(crate) gdb: Cow<'a, Path>,
}

pub(crate) fn discover_gdb<'a>(
    builder: &'a Builder<'_>,
    android: Option<&android::Android>,
) -> Option<Gdb<'a>> {
    // If there's an explicitly-configured gdb, use that.
    if let Some(gdb) = builder.config.gdb.as_deref() {
        // FIXME(Zalathar): Consider returning None if gdb is an empty string,
        // as a way to explicitly disable ambient gdb discovery.
        let gdb = Cow::Borrowed(gdb);
        return Some(Gdb { gdb });
    }

    // Otherwise, fall back to whatever gdb is sitting around in PATH.
    // (That's the historical behavior, but maybe we should require opt-in?)

    let gdb: Cow<'_, Path> = match android {
        Some(android::Android { android_cross_path, .. }) => {
            android_cross_path.join("bin/gdb").into()
        }
        None => Path::new("gdb").into(),
    };

    // Check whether an ambient gdb exists, by running `gdb --version`.
    let output = {
        let mut gdb_command = BootstrapCommand::new(gdb.as_ref()).allow_failure();
        gdb_command.arg("--version");
        gdb_command.run_capture(builder)
    };

    if output.is_success() { Some(Gdb { gdb }) } else { None }
}
