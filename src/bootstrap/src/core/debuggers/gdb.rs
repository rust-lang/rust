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
        if gdb.as_os_str().is_empty() {
            return None;
        }

        let gdb = Cow::Borrowed(gdb);
        if verify_gdb(builder, &gdb) {
            return Some(Gdb { gdb });
        } else {
            return None;
        }
    }

    // Otherwise, fall back to whatever gdb is sitting around in PATH.
    // (That's the historical behavior, but maybe we should require opt-in?)

    let gdb: Cow<'_, Path> = match android {
        Some(android::Android { android_cross_path, .. }) => {
            android_cross_path.join("bin/gdb").into()
        }
        None => Path::new("gdb").into(),
    };

    if verify_gdb(builder, &gdb) { Some(Gdb { gdb }) } else { None }
}

// Check whether an ambient gdb exists, by running `gdb --version`.
fn verify_gdb(builder: &Builder<'_>, gdb: &Path) -> bool {
    let mut cmd = BootstrapCommand::new(gdb).allow_failure();
    cmd.arg("--version");
    cmd.run_capture(builder).is_success()
}
