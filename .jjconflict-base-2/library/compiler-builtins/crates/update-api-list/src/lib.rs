use std::path::{Path, PathBuf};
use std::sync::LazyLock;

pub static WORKSPACE_ROOT: LazyLock<PathBuf> = LazyLock::new(|| {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_owned()
});
