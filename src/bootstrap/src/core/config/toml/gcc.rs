use serde::{Deserialize, Deserializer};

use crate::core::config::toml::{Merge, ReplaceOpt};
use crate::{HashSet, PathBuf, define_config, exit};

define_config! {
    /// TOML representation of how the GCC build is configured.
    struct Gcc {
        download_ci_gcc: Option<bool> = "download-ci-gcc",
    }
}
