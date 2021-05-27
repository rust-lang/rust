use std::{
    env,
    fs::File,
    io,
    path::{Path, PathBuf},
};

use anyhow::Result;
use flate2::{write::GzEncoder, Compression};
use xshell::{cmd, mkdir_p, pushd, pushenv, read_file, rm_rf, write_file};

use crate::{date_iso, flags, project_root};

impl flags::Dist {
    pub(crate) fn run(self) -> Result<()> {
        let stable =
            std::env::var("GITHUB_REF").unwrap_or_default().as_str() == "refs/heads/release";

        let dist = project_root().join("dist");
        rm_rf(&dist)?;
        mkdir_p(&dist)?;

        if let Some(patch_version) = self.client_patch_version {
            let version = if stable {
                format!("0.2.{}", patch_version)
            } else {
                // A hack to make VS Code prefer nightly over stable.
                format!("0.3.{}", patch_version)
            };
            let release_tag = if stable { date_iso()? } else { "nightly".to_string() };
            dist_client(&version, &release_tag)?;
        }
        let release_channel = if stable { "stable" } else { "nightly" };
        dist_server(release_channel)?;
        Ok(())
    }
}

fn dist_client(version: &str, release_tag: &str) -> Result<()> {
    let _d = pushd("./editors/code")?;
    let nightly = release_tag == "nightly";

    let mut patch = Patch::new("./package.json")?;

    patch
        .replace(r#""version": "0.4.0-dev""#, &format!(r#""version": "{}""#, version))
        .replace(r#""releaseTag": null"#, &format!(r#""releaseTag": "{}""#, release_tag))
        .replace(r#""$generated-start": false,"#, "")
        .replace(",\n                \"$generated-end\": false", "");

    if nightly {
        patch.replace(
            r#""displayName": "rust-analyzer""#,
            r#""displayName": "rust-analyzer (nightly)""#,
        );
    }
    if !nightly {
        patch.replace(r#""enableProposedApi": true,"#, r#""#);
    }
    patch.commit()?;

    cmd!("npm ci").run()?;
    cmd!("npx vsce package -o ../../dist/rust-analyzer.vsix").run()?;
    Ok(())
}

fn dist_server(release_channel: &str) -> Result<()> {
    let _e = pushenv("RUST_ANALYZER_CHANNEL", release_channel);
    let _e = pushenv("CARGO_PROFILE_RELEASE_LTO", "thin");

    // Uncomment to enable debug info for releases. Note that:
    //   * debug info is split on windows and macs, so it does nothing for those platforms,
    //   * on Linux, this blows up the binary size from 8MB to 43MB, which is unreasonable.
    // let _e = pushenv("CARGO_PROFILE_RELEASE_DEBUG", "1");

    let target = get_target();
    if target.contains("-linux-gnu") || target.contains("-linux-musl") {
        env::set_var("CC", "clang");
    }

    cmd!("cargo build --manifest-path ./crates/rust-analyzer/Cargo.toml --bin rust-analyzer --target {target} --release").run()?;

    let suffix = exe_suffix(&target);
    let src =
        Path::new("target").join(&target).join("release").join(format!("rust-analyzer{}", suffix));
    let dst = Path::new("dist").join(format!("rust-analyzer-{}{}", target, suffix));
    gzip(&src, &dst.with_extension("gz"))?;

    Ok(())
}

fn get_target() -> String {
    match env::var("RA_TARGET") {
        Ok(target) => target,
        _ => {
            if cfg!(target_os = "linux") {
                "x86_64-unknown-linux-gnu".to_string()
            } else if cfg!(target_os = "windows") {
                "x86_64-pc-windows-msvc".to_string()
            } else if cfg!(target_os = "macos") {
                "x86_64-apple-darwin".to_string()
            } else {
                panic!("Unsupported OS, maybe try setting RA_TARGET")
            }
        }
    }
}

fn exe_suffix(target: &str) -> String {
    if target.contains("-windows-") {
        ".exe".into()
    } else {
        "".into()
    }
}

fn gzip(src_path: &Path, dest_path: &Path) -> Result<()> {
    let mut encoder = GzEncoder::new(File::create(dest_path)?, Compression::best());
    let mut input = io::BufReader::new(File::open(src_path)?);
    io::copy(&mut input, &mut encoder)?;
    encoder.finish()?;
    Ok(())
}

struct Patch {
    path: PathBuf,
    original_contents: String,
    contents: String,
}

impl Patch {
    fn new(path: impl Into<PathBuf>) -> Result<Patch> {
        let path = path.into();
        let contents = read_file(&path)?;
        Ok(Patch { path, original_contents: contents.clone(), contents })
    }

    fn replace(&mut self, from: &str, to: &str) -> &mut Patch {
        assert!(self.contents.contains(from));
        self.contents = self.contents.replace(from, to);
        self
    }

    fn commit(&self) -> Result<()> {
        write_file(&self.path, &self.contents)?;
        Ok(())
    }
}

impl Drop for Patch {
    fn drop(&mut self) {
        write_file(&self.path, &self.original_contents).unwrap();
    }
}
