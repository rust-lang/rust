use anyhow::Context;
use flate2::{Compression, write::GzEncoder};
use std::{
    fs::File,
    io::{self, BufWriter},
    path::{Path, PathBuf},
};
use time::OffsetDateTime;
use xshell::{Cmd, Shell, cmd};
use zip::{DateTime, ZipWriter, write::SimpleFileOptions};

use crate::{
    date_iso,
    flags::{self, Malloc, PgoTrainingCrate},
    project_root,
    util::detect_target,
};

const VERSION_STABLE: &str = "0.3";
const VERSION_NIGHTLY: &str = "0.4";
const VERSION_DEV: &str = "0.5"; // keep this one in sync with `package.json`

impl flags::Dist {
    pub(crate) fn run(self, sh: &Shell) -> anyhow::Result<()> {
        let stable = sh.var("GITHUB_REF").unwrap_or_default().as_str() == "refs/heads/release";

        let project_root = project_root();
        let target = Target::get(&project_root, sh);
        let allocator = self.allocator();
        let dist = project_root.join("dist");
        sh.remove_path(&dist)?;
        sh.create_dir(&dist)?;

        if let Some(patch_version) = self.client_patch_version {
            let version = if stable {
                format!("{VERSION_STABLE}.{patch_version}")
            } else {
                // A hack to make VS Code prefer nightly over stable.
                format!("{VERSION_NIGHTLY}.{patch_version}")
            };
            dist_server(
                sh,
                &format!("{version}-standalone"),
                &target,
                allocator,
                self.zig,
                self.pgo,
            )?;
            let release_tag = if stable { date_iso(sh)? } else { "nightly".to_owned() };
            dist_client(sh, &version, &release_tag, &target)?;
        } else {
            dist_server(sh, "0.0.0-standalone", &target, allocator, self.zig, self.pgo)?;
        }
        Ok(())
    }
}

fn dist_client(
    sh: &Shell,
    version: &str,
    release_tag: &str,
    target: &Target,
) -> anyhow::Result<()> {
    let bundle_path = Path::new("editors").join("code").join("server");
    sh.create_dir(&bundle_path)?;
    sh.copy_file(&target.server_path, &bundle_path)?;
    if let Some(symbols_path) = &target.symbols_path {
        sh.copy_file(symbols_path, &bundle_path)?;
    }

    let _d = sh.push_dir("./editors/code");

    let mut patch = Patch::new(sh, "./package.json")?;
    patch
        .replace(
            &format!(r#""version": "{VERSION_DEV}.0-dev""#),
            &format!(r#""version": "{version}""#),
        )
        .replace(r#""releaseTag": null"#, &format!(r#""releaseTag": "{release_tag}""#))
        .replace(r#""title": "$generated-start""#, "")
        .replace(r#""title": "$generated-end""#, "")
        .replace(r#""enabledApiProposals": [],"#, r#""#);
    patch.commit(sh)?;

    Ok(())
}

fn dist_server(
    sh: &Shell,
    release: &str,
    target: &Target,
    allocator: Malloc,
    zig: bool,
    pgo: Option<PgoTrainingCrate>,
) -> anyhow::Result<()> {
    let _e = sh.push_env("CFG_RELEASE", release);
    let _e = sh.push_env("CARGO_PROFILE_RELEASE_LTO", "thin");

    // Uncomment to enable debug info for releases. Note that:
    //   * debug info is split on windows and macs, so it does nothing for those platforms,
    //   * on Linux, this blows up the binary size from 8MB to 43MB, which is unreasonable.
    // let _e = sh.push_env("CARGO_PROFILE_RELEASE_DEBUG", "1");

    let linux_target = target.is_linux();
    let target_name = match &target.libc_suffix {
        Some(libc_suffix) if zig => format!("{}.{libc_suffix}", target.name),
        _ => target.name.to_owned(),
    };
    let features = allocator.to_features();
    let command = if linux_target && zig { "zigbuild" } else { "build" };

    let pgo_profile = if let Some(train_crate) = pgo {
        Some(crate::pgo::gather_pgo_profile(
            sh,
            crate::pgo::build_command(sh, command, &target_name, features),
            &target_name,
            train_crate,
        )?)
    } else {
        None
    };

    let mut cmd = build_command(sh, command, &target_name, features);
    if let Some(profile) = pgo_profile {
        cmd = cmd.env("RUSTFLAGS", format!("-Cprofile-use={}", profile.to_str().unwrap()));
    }
    cmd.run().context("cannot build Rust Analyzer")?;

    let dst = Path::new("dist").join(&target.artifact_name);
    if target_name.contains("-windows-") {
        zip(&target.server_path, target.symbols_path.as_ref(), &dst.with_extension("zip"))?;
    } else {
        gzip(&target.server_path, &dst.with_extension("gz"))?;
    }

    Ok(())
}

fn build_command<'a>(
    sh: &'a Shell,
    command: &str,
    target_name: &str,
    features: &[&str],
) -> Cmd<'a> {
    cmd!(
        sh,
        "cargo {command} --manifest-path ./crates/rust-analyzer/Cargo.toml --bin rust-analyzer --target {target_name} {features...} --release"
    )
}

fn gzip(src_path: &Path, dest_path: &Path) -> anyhow::Result<()> {
    let mut encoder = GzEncoder::new(File::create(dest_path)?, Compression::best());
    let mut input = io::BufReader::new(File::open(src_path)?);
    io::copy(&mut input, &mut encoder)?;
    encoder.finish()?;
    Ok(())
}

fn zip(src_path: &Path, symbols_path: Option<&PathBuf>, dest_path: &Path) -> anyhow::Result<()> {
    let file = File::create(dest_path)?;
    let mut writer = ZipWriter::new(BufWriter::new(file));
    writer.start_file(
        src_path.file_name().unwrap().to_str().unwrap(),
        SimpleFileOptions::default()
            .last_modified_time(
                DateTime::try_from(OffsetDateTime::from(std::fs::metadata(src_path)?.modified()?))
                    .unwrap(),
            )
            .unix_permissions(0o755)
            .compression_method(zip::CompressionMethod::Deflated)
            .compression_level(Some(9)),
    )?;
    let mut input = io::BufReader::new(File::open(src_path)?);
    io::copy(&mut input, &mut writer)?;
    if let Some(symbols_path) = symbols_path {
        writer.start_file(
            symbols_path.file_name().unwrap().to_str().unwrap(),
            SimpleFileOptions::default()
                .last_modified_time(
                    DateTime::try_from(OffsetDateTime::from(
                        std::fs::metadata(src_path)?.modified()?,
                    ))
                    .unwrap(),
                )
                .compression_method(zip::CompressionMethod::Deflated)
                .compression_level(Some(9)),
        )?;
        let mut input = io::BufReader::new(File::open(symbols_path)?);
        io::copy(&mut input, &mut writer)?;
    }
    writer.finish()?;
    Ok(())
}

struct Target {
    name: String,
    libc_suffix: Option<String>,
    server_path: PathBuf,
    symbols_path: Option<PathBuf>,
    artifact_name: String,
}

impl Target {
    fn get(project_root: &Path, sh: &Shell) -> Self {
        let name = detect_target(sh);
        let (name, libc_suffix) = match name.split_once('.') {
            Some((l, r)) => (l.to_owned(), Some(r.to_owned())),
            None => (name, None),
        };
        let out_path = project_root.join("target").join(&name).join("release");
        let (exe_suffix, symbols_path) = if name.contains("-windows-") {
            (".exe".into(), Some(out_path.join("rust_analyzer.pdb")))
        } else {
            (String::new(), None)
        };
        let server_path = out_path.join(format!("rust-analyzer{exe_suffix}"));
        let artifact_name = format!("rust-analyzer-{name}{exe_suffix}");
        Self { name, libc_suffix, server_path, symbols_path, artifact_name }
    }

    fn is_linux(&self) -> bool {
        self.name.contains("-linux-")
    }
}

struct Patch {
    path: PathBuf,
    original_contents: String,
    contents: String,
}

impl Patch {
    fn new(sh: &Shell, path: impl Into<PathBuf>) -> anyhow::Result<Patch> {
        let path = path.into();
        let contents = sh.read_file(&path)?;
        Ok(Patch { path, original_contents: contents.clone(), contents })
    }

    fn replace(&mut self, from: &str, to: &str) -> &mut Patch {
        assert!(self.contents.contains(from));
        self.contents = self.contents.replace(from, to);
        self
    }

    fn commit(&self, sh: &Shell) -> anyhow::Result<()> {
        sh.write_file(&self.path, &self.contents)?;
        Ok(())
    }
}

impl Drop for Patch {
    fn drop(&mut self) {
        // FIXME: find a way to bring this back
        let _ = &self.original_contents;
        // write_file(&self.path, &self.original_contents).unwrap();
    }
}
