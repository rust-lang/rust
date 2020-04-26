use std::path::PathBuf;

use anyhow::Result;

use crate::{
    not_bash::{date_iso, fs2, pushd, rm_rf, run},
    project_root,
};

pub fn run_dist(nightly: bool, client_version: Option<String>) -> Result<()> {
    let dist = project_root().join("dist");
    rm_rf(&dist)?;
    fs2::create_dir_all(&dist)?;

    if let Some(version) = client_version {
        let release_tag = if nightly { "nightly".to_string() } else { date_iso()? };
        dist_client(&version, &release_tag)?;
    }
    dist_server(nightly)?;
    Ok(())
}

fn dist_client(version: &str, release_tag: &str) -> Result<()> {
    let _d = pushd("./editors/code");
    let nightly = release_tag == "nightly";

    let mut patch = Patch::new("./package.json")?;

    patch
        .replace(r#""version": "0.4.0-dev""#, &format!(r#""version": "{}""#, version))
        .replace(r#""releaseTag": null"#, &format!(r#""releaseTag": "{}""#, release_tag));

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

    run!("npm ci")?;
    run!("npx vsce package -o ../../dist/rust-analyzer.vsix")?;
    Ok(())
}

fn dist_server(nightly: bool) -> Result<()> {
    if cfg!(target_os = "linux") {
        std::env::set_var("CC", "clang");
        run!(
            "cargo build --manifest-path ./crates/rust-analyzer/Cargo.toml --bin rust-analyzer --release"
            // We'd want to add, but that requires setting the right linker somehow
            // --features=jemalloc
        )?;
        if !nightly {
            run!("strip ./target/release/rust-analyzer")?;
        }
    } else {
        run!("cargo build --manifest-path ./crates/rust-analyzer/Cargo.toml --bin rust-analyzer --release")?;
    }

    let (src, dst) = if cfg!(target_os = "linux") {
        ("./target/release/rust-analyzer", "./dist/rust-analyzer-linux")
    } else if cfg!(target_os = "windows") {
        ("./target/release/rust-analyzer.exe", "./dist/rust-analyzer-windows.exe")
    } else if cfg!(target_os = "macos") {
        ("./target/release/rust-analyzer", "./dist/rust-analyzer-mac")
    } else {
        panic!("Unsupported OS")
    };

    fs2::copy(src, dst)?;

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
        let contents = fs2::read_to_string(&path)?;
        Ok(Patch { path, original_contents: contents.clone(), contents })
    }

    fn replace(&mut self, from: &str, to: &str) -> &mut Patch {
        assert!(self.contents.contains(from));
        self.contents = self.contents.replace(from, to);
        self
    }

    fn commit(&self) -> Result<()> {
        fs2::write(&self.path, &self.contents)
    }
}

impl Drop for Patch {
    fn drop(&mut self) {
        fs2::write(&self.path, &self.original_contents).unwrap();
    }
}
