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
    let original_package_json = fs2::read_to_string(&package_json_path)?;
    let _restore =
        Restore { path: package_json_path.clone(), contents: original_package_json.clone() };

    let mut package_json = original_package_json.replace(r#""enableProposedApi": true,"#, r#""#);

    if nightly {
        package_json = package_json.replace(
            r#""displayName": "rust-analyzer""#,
            r#""displayName": "rust-analyzer nightly""#,
        );
    } else {
        package_json = original_package_json.replace(r#""enableProposedApi": true,"#, r#""#);
    }
    fs2::write(package_json_path, package_json)?;

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
             --features=jemalloc"
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

struct Restore {
    path: PathBuf,
    contents: String,
}

impl Drop for Restore {
    fn drop(&mut self) {
        fs2::write(&self.path, &self.contents).unwrap();
    }
}
