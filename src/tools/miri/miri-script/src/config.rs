use std::{fs, io};

use anyhow::{Context, Result, bail};
use path_macro::path;
use serde_derive::Deserialize;

use crate::util::miri_dir;

#[derive(Deserialize, Default)]
pub struct Config {
    #[serde(default)]
    pub toolchain: Toolchain,
    #[serde(default)]
    pub auto: Auto,
}

#[derive(Deserialize, Default)]
pub struct Toolchain {
    pub name: Option<String>,
}

#[derive(Deserialize, Default)]
pub struct Auto {
    #[serde(default)]
    pub toolchain: bool,
    #[serde(default)]
    pub fmt: bool,
    #[serde(default)]
    pub clippy: bool,
}

impl Config {
    pub fn load() -> Result<Self> {
        let miri_dir = miri_dir()?;

        Ok(match fs::read(path!(miri_dir / "miri.toml")) {
            Ok(config) => toml::from_slice(&config).context("failed to parse `miri.toml`")?,
            Err(err) if err.kind() == io::ErrorKind::NotFound => {
                // Just ignore error if the file does not exist. Fall back to parsing the `.auto-*`
                // files.
                let mut config = Config::default();
                let everything = path!(miri_dir / ".auto-everything").exists();
                config.auto.toolchain = everything || path!(miri_dir / ".auto-toolchain").exists();
                config.auto.fmt = everything || path!(miri_dir / ".auto-fmt").exists();
                config.auto.clippy = everything || path!(miri_dir / ".auto-clippy").exists();
                config
            }
            Err(err) => bail!("unable to open `miri.toml`: {err}"),
        })
    }
}
