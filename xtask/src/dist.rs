use std::{
    env,
    fs::File,
    io,
    path::{Path, PathBuf},
};

use anyhow::Result;
use flate2::{write::GzEncoder, Compression};
use xshell::{cmd, cp, mkdir_p, pushd, pushenv, read_file, rm_rf, write_file};

use crate::{date_iso, project_root};

pub(crate) struct DistCmd {
    pub(crate) nightly: bool,
    pub(crate) client_version: Option<String>,
}

impl DistCmd {
    pub(crate) fn run(self) -> Result<()> {
        let dist = project_root().join("dist");
        rm_rf(&dist)?;
        mkdir_p(&dist)?;

        if let Some(version) = self.client_version {
            let release_tag = if self.nightly { "nightly".to_string() } else { date_iso()? };
            dist_client(&version, &release_tag)?;
        }
        let release_channel = if self.nightly { "nightly" } else { "stable" };
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

    // FIXME: the old names are temporarily kept for client compatibility, but they should be removed
    // Remove this block after a couple of releases
    match target.as_ref() {
        "x86_64-unknown-linux-gnu" => {
            cp(&src, "dist/rust-analyzer-linux")?;
            gzip(&src, Path::new("dist/rust-analyzer-linux.gz"))?;
        }
        "x86_64-pc-windows-msvc" => {
            cp(&src, "dist/rust-analyzer-windows.exe")?;
            gzip(&src, Path::new("dist/rust-analyzer-windows.gz"))?;
        }
        "x86_64-apple-darwin" => {
            cp(&src, "dist/rust-analyzer-mac")?;
            gzip(&src, Path::new("dist/rust-analyzer-mac.gz"))?;
        }
        _ => {}
    }

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
