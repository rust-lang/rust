use std::path::PathBuf;

use crate::core::builder::Builder;
use crate::core::config::TargetSelection;

pub(crate) struct Android {
    pub(crate) adb_path: &'static str,
    pub(crate) adb_test_dir: &'static str,
    pub(crate) android_cross_path: PathBuf,
}

pub(crate) fn discover_android(builder: &Builder<'_>, target: TargetSelection) -> Option<Android> {
    let adb_path = "adb";
    // See <https://github.com/rust-lang/rust/pull/102755>.
    let adb_test_dir = "/data/local/tmp/work";

    let android_cross_path = if target.contains("android") && !builder.config.dry_run() {
        builder.cc(target).parent().unwrap().parent().unwrap().to_owned()
    } else {
        PathBuf::new()
    };

    Some(Android { adb_path, adb_test_dir, android_cross_path })
}
