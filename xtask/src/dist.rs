use std::path::PathBuf;

use anyhow::Result;

use crate::{
    not_bash::{fs2, pushd, rm_rf, run},
    project_root,
};

pub fn run_dist(nightly: bool) -> Result<()> {
    let dist = project_root().join("dist");
    rm_rf(&dist)?;
    fs2::create_dir_all(&dist)?;

    if cfg!(target_os = "linux") {
        dist_client(nightly)?;
    }
    dist_server()?;
    Ok(())
}

fn dist_client(nightly: bool) -> Result<()> {
    let _d = pushd("./editors/code");

    let package_json_path = PathBuf::from("./package.json");
    let mut patch = Patch::new(package_json_path.clone())?;

    let date = run!("date --utc +%Y%m%d")?;
    let version_suffix = if nightly { "-nightly" } else { "" };

    patch.replace(
        r#""version": "0.2.20200309-nightly""#,
        &format!(r#""version": "0.1.{}{}""#, date, version_suffix),
    );

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

fn dist_server() -> Result<()> {
    if cfg!(target_os = "linux") {
        std::env::set_var("CC", "clang");
        run!(
            "cargo build --manifest-path ./crates/rust-analyzer/Cargo.toml --bin rust-analyzer --release
             --target x86_64-unknown-linux-musl
            "
            // We'd want to add, but that requires setting the right linker somehow
            // --features=jemalloc
        )?;
        run!("strip ./target/x86_64-unknown-linux-musl/release/rust-analyzer")?;
    } else {
        run!("cargo build --manifest-path ./crates/rust-analyzer/Cargo.toml --bin rust-analyzer --release")?;
    }

    let (src, dst) = if cfg!(target_os = "linux") {
        ("./target/x86_64-unknown-linux-musl/release/rust-analyzer", "./dist/rust-analyzer-linux")
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
    fn new(path: PathBuf) -> Result<Patch> {
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
