use std::collections::BTreeMap;
use std::env;

pub fn capture_env(capture_all: bool) -> BTreeMap<String, String> {
    env::vars().filter(|(k, _)| capture_all || allowlisted_env(k)).collect::<BTreeMap<_, _>>()
}

pub fn allowlisted_env(key: &str) -> bool {
    key.starts_with("CARGO_")
        || key.starts_with("RUST")
        || key == "PATH"
        || key == "HOME"
        || key == "USERPROFILE"
        || key == "PWD"
        || key == "TMPDIR"
        || key == "TEMP"
        || key == "TMP"
        || key == "SOURCE_DATE_EPOCH"
        || key == "LANG"
        || key.starts_with("LC_")
        || key == "TZ"
}
