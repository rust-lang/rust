use crate::walk::{filter_dirs, walk};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Config {}

pub fn check(path: &Path, bad: &mut bool) {
    walk(path, &mut |path| filter_dirs(path), &mut |entry, contents| {
        let file = entry.path();
        let filename = file.file_name().unwrap();
        if filename != "triagebot.toml" {
            return;
        }
        let conf: Result<Config, toml::de::Error> = toml::from_str(contents);
        match conf {
            Ok(_) => {}
            Err(_err) => {
                tidy_error!(bad, "{} is an invalid toml file", file.display())
            }
        }
    });
}
