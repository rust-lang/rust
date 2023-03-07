use anyhow::{Context, Error};
use curl::easy::Easy;
use indexmap::IndexMap;
use std::collections::HashMap;
use std::convert::TryInto;

const PATH: &str = "src/stage0.json";
const COMPILER_COMPONENTS: &[&str] = &["rustc", "rust-std", "cargo"];
const RUSTFMT_COMPONENTS: &[&str] = &["rustfmt-preview", "rustc"];

struct Tool {
    config: Config,
    comments: Vec<String>,

    channel: Channel,
    date: Option<String>,
    version: [u16; 3],
    checksums: IndexMap<String, String>,
}

impl Tool {
    fn new(date: Option<String>) -> Result<Self, Error> {
        let channel = match std::fs::read_to_string("src/ci/channel")?.trim() {
            "stable" => Channel::Stable,
            "beta" => Channel::Beta,
            "nightly" => Channel::Nightly,
            other => anyhow::bail!("unsupported channel: {}", other),
        };

        // Split "1.42.0" into [1, 42, 0]
        let version = std::fs::read_to_string("src/version")?
            .trim()
            .split('.')
            .map(|val| val.parse())
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("failed to parse version"))?;

        let existing: Stage0 = serde_json::from_slice(&std::fs::read(PATH)?)?;

        Ok(Self {
            channel,
            version,
            date,
            config: existing.config,
            comments: existing.comments,
            checksums: IndexMap::new(),
        })
    }

    fn update_json(mut self) -> Result<(), Error> {
        std::fs::write(
            PATH,
            format!(
                "{}\n",
                serde_json::to_string_pretty(&Stage0 {
                    compiler: self.detect_compiler()?,
                    rustfmt: self.detect_rustfmt()?,
                    checksums_sha256: {
                        // Keys are sorted here instead of beforehand because values in this map
                        // are added while filling the other struct fields just above this block.
                        self.checksums.sort_keys();
                        self.checksums
                    },
                    config: self.config,
                    comments: self.comments,
                })?
            ),
        )?;
        Ok(())
    }

    // Currently Rust always bootstraps from the previous stable release, and in our train model
    // this means that the master branch bootstraps from beta, beta bootstraps from current stable,
    // and stable bootstraps from the previous stable release.
    //
    // On the master branch the compiler version is configured to `beta` whereas if you're looking
    // at the beta or stable channel you'll likely see `1.x.0` as the version, with the previous
    // release's version number.
    fn detect_compiler(&mut self) -> Result<Stage0Toolchain, Error> {
        let channel = match self.channel {
            Channel::Stable | Channel::Beta => {
                // The 1.XX manifest points to the latest point release of that minor release.
                format!("{}.{}", self.version[0], self.version[1] - 1)
            }
            Channel::Nightly => "beta".to_string(),
        };

        let manifest = fetch_manifest(&self.config, &channel, self.date.as_deref())?;
        self.collect_checksums(&manifest, COMPILER_COMPONENTS)?;
        Ok(Stage0Toolchain {
            date: manifest.date,
            version: if self.channel == Channel::Nightly {
                "beta".to_string()
            } else {
                // The version field is like "1.42.0 (abcdef1234 1970-01-01)"
                manifest.pkg["rust"]
                    .version
                    .split_once(' ')
                    .expect("invalid version field")
                    .0
                    .to_string()
            },
        })
    }

    /// We use a nightly rustfmt to format the source because it solves some bootstrapping issues
    /// with use of new syntax in this repo. For the beta/stable channels rustfmt is not provided,
    /// as we don't want to depend on rustfmt from nightly there.
    fn detect_rustfmt(&mut self) -> Result<Option<Stage0Toolchain>, Error> {
        if self.channel != Channel::Nightly {
            return Ok(None);
        }

        let manifest = fetch_manifest(&self.config, "nightly", self.date.as_deref())?;
        self.collect_checksums(&manifest, RUSTFMT_COMPONENTS)?;
        Ok(Some(Stage0Toolchain { date: manifest.date, version: "nightly".into() }))
    }

    fn collect_checksums(&mut self, manifest: &Manifest, components: &[&str]) -> Result<(), Error> {
        let prefix = format!("{}/", self.config.dist_server);
        for component in components {
            let pkg = manifest
                .pkg
                .get(*component)
                .ok_or_else(|| anyhow::anyhow!("missing component from manifest: {}", component))?;
            for target in pkg.target.values() {
                for pair in &[(&target.url, &target.hash), (&target.xz_url, &target.xz_hash)] {
                    if let (Some(url), Some(sha256)) = pair {
                        let url = url
                            .strip_prefix(&prefix)
                            .ok_or_else(|| {
                                anyhow::anyhow!("url doesn't start with dist server base: {}", url)
                            })?
                            .to_string();
                        self.checksums.insert(url, sha256.clone());
                    }
                }
            }
        }
        Ok(())
    }
}

fn main() -> Result<(), Error> {
    let tool = Tool::new(std::env::args().nth(1))?;
    tool.update_json()?;
    Ok(())
}

fn fetch_manifest(config: &Config, channel: &str, date: Option<&str>) -> Result<Manifest, Error> {
    let url = if let Some(date) = date {
        format!("{}/dist/{}/channel-rust-{}.toml", config.dist_server, date, channel)
    } else {
        format!("{}/dist/channel-rust-{}.toml", config.dist_server, channel)
    };

    Ok(toml::from_slice(&http_get(&url)?)?)
}

fn http_get(url: &str) -> Result<Vec<u8>, Error> {
    let mut data = Vec::new();
    let mut handle = Easy::new();
    handle.fail_on_error(true)?;
    handle.url(url)?;
    {
        let mut transfer = handle.transfer();
        transfer.write_function(|new_data| {
            data.extend_from_slice(new_data);
            Ok(new_data.len())
        })?;
        transfer.perform().context(format!("failed to fetch {url}"))?;
    }
    Ok(data)
}

#[derive(Debug, PartialEq, Eq)]
enum Channel {
    Stable,
    Beta,
    Nightly,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct Stage0 {
    config: Config,
    // Comments are explicitly below the config, do not move them above.
    //
    // Downstream forks of the compiler codebase can change the configuration values defined above,
    // but doing so would risk merge conflicts whenever they import new changes that include a
    // bootstrap compiler bump.
    //
    // To lessen the pain, a big block of comments is placed between the configuration and the
    // auto-generated parts of the file, preventing git diffs of the config to include parts of the
    // auto-generated content and vice versa. This should prevent merge conflicts.
    #[serde(rename = "__comments")]
    comments: Vec<String>,
    compiler: Stage0Toolchain,
    rustfmt: Option<Stage0Toolchain>,
    checksums_sha256: IndexMap<String, String>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct Config {
    dist_server: String,
    // There are other fields in the configuration, which will be read by src/bootstrap or other
    // tools consuming stage0.json. To avoid the need to update bump-stage0 every time a new field
    // is added, we collect all the fields in an untyped Value and serialize them back with the
    // same order and structure they were deserialized in.
    #[serde(flatten)]
    other: serde_json::Value,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct Stage0Toolchain {
    date: String,
    version: String,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct Manifest {
    date: String,
    pkg: HashMap<String, ManifestPackage>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ManifestPackage {
    version: String,
    target: HashMap<String, ManifestTargetPackage>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ManifestTargetPackage {
    url: Option<String>,
    hash: Option<String>,
    xz_url: Option<String>,
    xz_hash: Option<String>,
}
