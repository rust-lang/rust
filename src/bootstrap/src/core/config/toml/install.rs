use serde::{Deserialize, Deserializer};

use crate::core::config::toml::{Merge, ReplaceOpt};
use crate::{HashSet, PathBuf, define_config, exit};

define_config! {
    /// TOML representation of various global install decisions.
    struct Install {
        prefix: Option<String> = "prefix",
        sysconfdir: Option<String> = "sysconfdir",
        docdir: Option<String> = "docdir",
        bindir: Option<String> = "bindir",
        libdir: Option<String> = "libdir",
        mandir: Option<String> = "mandir",
        datadir: Option<String> = "datadir",
    }
}
