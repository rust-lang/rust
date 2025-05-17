use std::env;
use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use xz2::bufread::XzDecoder;

use crate::core::config::BUILDER_CONFIG_FILENAME;
use crate::utils::build_stamp::BuildStamp;
use crate::utils::exec::{BootstrapCommand, command};
use crate::utils::helpers::{check_run, exe, hex_encode, move_file};
use crate::{Config, t};

static SHOULD_FIX_BINS_AND_DYLIBS: OnceLock<bool> = OnceLock::new();

/// `Config::try_run` wrapper for this module to avoid warnings on `try_run`, since we don't have access to a `builder` yet.
fn try_run(config: &Config, cmd: &mut Command) -> Result<(), ()> {
    #[expect(deprecated)]
    config.try_run(cmd)
}

fn extract_curl_version(out: &[u8]) -> semver::Version {
    let out = String::from_utf8_lossy(out);
    // The output should look like this: "curl <major>.<minor>.<patch> ..."
    out.lines()
        .next()
        .and_then(|line| line.split(" ").nth(1))
        .and_then(|version| semver::Version::parse(version).ok())
        .unwrap_or(semver::Version::new(1, 0, 0))
}

fn curl_version() -> semver::Version {
    let mut curl = Command::new("curl");
    curl.arg("-V");
    let Ok(out) = curl.output() else { return semver::Version::new(1, 0, 0) };
    let out = out.stdout;
    extract_curl_version(&out)
}

/// Generic helpers that are useful anywhere in bootstrap.
impl Config {
    pub fn is_verbose(&self) -> bool {
        self.verbose > 0
    }

    pub(crate) fn create<P: AsRef<Path>>(&self, path: P, s: &str) {
        if self.dry_run() {
            return;
        }
        t!(fs::write(path, s));
    }

    pub(crate) fn remove(&self, f: &Path) {
        if self.dry_run() {
            return;
        }
        fs::remove_file(f).unwrap_or_else(|_| panic!("failed to remove {f:?}"));
    }

    /// Create a temporary directory in `out` and return its path.
    ///
    /// NOTE: this temporary directory is shared between all steps;
    /// if you need an empty directory, create a new subdirectory inside it.
    pub(crate) fn tempdir(&self) -> PathBuf {
        let tmp = self.out.join("tmp");
        t!(fs::create_dir_all(&tmp));
        tmp
    }

    /// Runs a command, printing out nice contextual information if it fails.
    /// Returns false if do not execute at all, otherwise returns its
    /// `status.success()`.
    pub(crate) fn check_run(&self, cmd: &mut BootstrapCommand) -> bool {
        if self.dry_run() && !cmd.run_always {
            return true;
        }
        self.verbose(|| println!("running: {cmd:?}"));
        check_run(cmd, self.is_verbose())
    }

    /// Whether or not `fix_bin_or_dylib` needs to be run; can only be true
    /// on NixOS
    fn should_fix_bins_and_dylibs(&self) -> bool {
        let val = *SHOULD_FIX_BINS_AND_DYLIBS.get_or_init(|| {
            match Command::new("uname").arg("-s").stderr(Stdio::inherit()).output() {
                Err(_) => return false,
                Ok(output) if !output.status.success() => return false,
                Ok(output) => {
                    let mut os_name = output.stdout;
                    if os_name.last() == Some(&b'\n') {
                        os_name.pop();
                    }
                    if os_name != b"Linux" {
                        return false;
                    }
                }
            }

            // If the user has asked binaries to be patched for Nix, then
            // don't check for NixOS or `/lib`.
            // NOTE: this intentionally comes after the Linux check:
            // - patchelf only works with ELF files, so no need to run it on Mac or Windows
            // - On other Unix systems, there is no stable syscall interface, so Nix doesn't manage the global libc.
            if let Some(explicit_value) = self.patch_binaries_for_nix {
                return explicit_value;
            }

            // Use `/etc/os-release` instead of `/etc/NIXOS`.
            // The latter one does not exist on NixOS when using tmpfs as root.
            let is_nixos = match File::open("/etc/os-release") {
                Err(e) if e.kind() == ErrorKind::NotFound => false,
                Err(e) => panic!("failed to access /etc/os-release: {e}"),
                Ok(os_release) => BufReader::new(os_release).lines().any(|l| {
                    let l = l.expect("reading /etc/os-release");
                    matches!(l.trim(), "ID=nixos" | "ID='nixos'" | "ID=\"nixos\"")
                }),
            };
            if !is_nixos {
                let in_nix_shell = env::var("IN_NIX_SHELL");
                if let Ok(in_nix_shell) = in_nix_shell {
                    eprintln!(
                        "The IN_NIX_SHELL environment variable is `{in_nix_shell}`; \
                         you may need to set `patch-binaries-for-nix=true` in bootstrap.toml"
                    );
                }
            }
            is_nixos
        });
        if val {
            eprintln!("INFO: You seem to be using Nix.");
        }
        val
    }

    /// Modifies the interpreter section of 'fname' to fix the dynamic linker,
    /// or the RPATH section, to fix the dynamic library search path
    ///
    /// This is only required on NixOS and uses the PatchELF utility to
    /// change the interpreter/RPATH of ELF executables.
    ///
    /// Please see <https://nixos.org/patchelf.html> for more information
    fn fix_bin_or_dylib(&self, fname: &Path) {
        assert_eq!(SHOULD_FIX_BINS_AND_DYLIBS.get(), Some(&true));
        println!("attempting to patch {}", fname.display());

        // Only build `.nix-deps` once.
        static NIX_DEPS_DIR: OnceLock<PathBuf> = OnceLock::new();
        let mut nix_build_succeeded = true;
        let nix_deps_dir = NIX_DEPS_DIR.get_or_init(|| {
            // Run `nix-build` to "build" each dependency (which will likely reuse
            // the existing `/nix/store` copy, or at most download a pre-built copy).
            //
            // Importantly, we create a gc-root called `.nix-deps` in the `build/`
            // directory, but still reference the actual `/nix/store` path in the rpath
            // as it makes it significantly more robust against changes to the location of
            // the `.nix-deps` location.
            //
            // bintools: Needed for the path of `ld-linux.so` (via `nix-support/dynamic-linker`).
            // zlib: Needed as a system dependency of `libLLVM-*.so`.
            // patchelf: Needed for patching ELF binaries (see doc comment above).
            let nix_deps_dir = self.out.join(".nix-deps");
            const NIX_EXPR: &str = "
            with (import <nixpkgs> {});
            symlinkJoin {
                name = \"rust-stage0-dependencies\";
                paths = [
                    zlib
                    patchelf
                    stdenv.cc.bintools
                ];
            }
            ";
            nix_build_succeeded = try_run(
                self,
                Command::new("nix-build").args([
                    Path::new("-E"),
                    Path::new(NIX_EXPR),
                    Path::new("-o"),
                    &nix_deps_dir,
                ]),
            )
            .is_ok();
            nix_deps_dir
        });
        if !nix_build_succeeded {
            return;
        }

        let mut patchelf = Command::new(nix_deps_dir.join("bin/patchelf"));
        patchelf.args(&[
            OsString::from("--add-rpath"),
            OsString::from(t!(fs::canonicalize(nix_deps_dir)).join("lib")),
        ]);
        if !path_is_dylib(fname) {
            // Finally, set the correct .interp for binaries
            let dynamic_linker_path = nix_deps_dir.join("nix-support/dynamic-linker");
            let dynamic_linker = t!(fs::read_to_string(dynamic_linker_path));
            patchelf.args(["--set-interpreter", dynamic_linker.trim_end()]);
        }

        let _ = try_run(self, patchelf.arg(fname));
    }

    fn download_file(&self, url: &str, dest_path: &Path, help_on_error: &str) {
        self.verbose(|| println!("download {url}"));
        // Use a temporary file in case we crash while downloading, to avoid a corrupt download in cache/.
        let tempfile = self.tempdir().join(dest_path.file_name().unwrap());
        // While bootstrap itself only supports http and https downloads, downstream forks might
        // need to download components from other protocols. The match allows them adding more
        // protocols without worrying about merge conflicts if we change the HTTP implementation.
        match url.split_once("://").map(|(proto, _)| proto) {
            Some("http") | Some("https") => {
                self.download_http_with_retries(&tempfile, url, help_on_error)
            }
            Some(other) => panic!("unsupported protocol {other} in {url}"),
            None => panic!("no protocol in {url}"),
        }
        t!(
            move_file(&tempfile, dest_path),
            format!("failed to rename {tempfile:?} to {dest_path:?}")
        );
    }

    fn download_http_with_retries(&self, tempfile: &Path, url: &str, help_on_error: &str) {
        println!("downloading {url}");
        // Try curl. If that fails and we are on windows, fallback to PowerShell.
        // options should be kept in sync with
        // src/bootstrap/src/core/download.rs
        // for consistency
        let mut curl = command("curl");
        curl.args([
            // follow redirect
            "--location",
            // timeout if speed is < 10 bytes/sec for > 30 seconds
            "--speed-time",
            "30",
            "--speed-limit",
            "10",
            // timeout if cannot connect within 30 seconds
            "--connect-timeout",
            "30",
            // output file
            "--output",
            tempfile.to_str().unwrap(),
            // if there is an error, don't restart the download,
            // instead continue where it left off.
            "--continue-at",
            "-",
            // retry up to 3 times.  note that this means a maximum of 4
            // attempts will be made, since the first attempt isn't a *re*try.
            "--retry",
            "3",
            // show errors, even if --silent is specified
            "--show-error",
            // set timestamp of downloaded file to that of the server
            "--remote-time",
            // fail on non-ok http status
            "--fail",
        ]);
        // Don't print progress in CI; the \r wrapping looks bad and downloads don't take long enough for progress to be useful.
        if self.is_running_on_ci {
            curl.arg("--silent");
        } else {
            curl.arg("--progress-bar");
        }
        // --retry-all-errors was added in 7.71.0, don't use it if curl is old.
        if curl_version() >= semver::Version::new(7, 71, 0) {
            curl.arg("--retry-all-errors");
        }
        curl.arg(url);
        if !self.check_run(&mut curl) {
            if self.build.contains("windows-msvc") {
                eprintln!("Fallback to PowerShell");
                for _ in 0..3 {
                    if try_run(self, Command::new("PowerShell.exe").args([
                        "/nologo",
                        "-Command",
                        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;",
                        &format!(
                            "(New-Object System.Net.WebClient).DownloadFile('{}', '{}')",
                            url, tempfile.to_str().expect("invalid UTF-8 not supported with powershell downloads"),
                        ),
                    ])).is_err() {
                        return;
                    }
                    eprintln!("\nspurious failure, trying again");
                }
            }
            if !help_on_error.is_empty() {
                eprintln!("{help_on_error}");
            }
            crate::exit!(1);
        }
    }

    fn unpack(&self, tarball: &Path, dst: &Path, pattern: &str) {
        eprintln!("extracting {} to {}", tarball.display(), dst.display());
        if !dst.exists() {
            t!(fs::create_dir_all(dst));
        }

        // `tarball` ends with `.tar.xz`; strip that suffix
        // example: `rust-dev-nightly-x86_64-unknown-linux-gnu`
        let uncompressed_filename =
            Path::new(tarball.file_name().expect("missing tarball filename")).file_stem().unwrap();
        let directory_prefix = Path::new(Path::new(uncompressed_filename).file_stem().unwrap());

        // decompress the file
        let data = t!(File::open(tarball), format!("file {} not found", tarball.display()));
        let decompressor = XzDecoder::new(BufReader::new(data));

        let mut tar = tar::Archive::new(decompressor);

        let is_ci_rustc = dst.ends_with("ci-rustc");
        let is_ci_llvm = dst.ends_with("ci-llvm");

        // `compile::Sysroot` needs to know the contents of the `rustc-dev` tarball to avoid adding
        // it to the sysroot unless it was explicitly requested. But parsing the 100 MB tarball is slow.
        // Cache the entries when we extract it so we only have to read it once.
        let mut recorded_entries = if is_ci_rustc { recorded_entries(dst, pattern) } else { None };

        for member in t!(tar.entries()) {
            let mut member = t!(member);
            let original_path = t!(member.path()).into_owned();
            // skip the top-level directory
            if original_path == directory_prefix {
                continue;
            }
            let mut short_path = t!(original_path.strip_prefix(directory_prefix));
            let is_builder_config = short_path.to_str() == Some(BUILDER_CONFIG_FILENAME);

            if !(short_path.starts_with(pattern)
                || ((is_ci_rustc || is_ci_llvm) && is_builder_config))
            {
                continue;
            }
            short_path = short_path.strip_prefix(pattern).unwrap_or(short_path);
            let dst_path = dst.join(short_path);
            self.verbose(|| {
                println!("extracting {} to {}", original_path.display(), dst.display())
            });
            if !t!(member.unpack_in(dst)) {
                panic!("path traversal attack ??");
            }
            if let Some(record) = &mut recorded_entries {
                t!(writeln!(record, "{}", short_path.to_str().unwrap()));
            }
            let src_path = dst.join(original_path);
            if src_path.is_dir() && dst_path.exists() {
                continue;
            }
            t!(move_file(src_path, dst_path));
        }
        let dst_dir = dst.join(directory_prefix);
        if dst_dir.exists() {
            t!(fs::remove_dir_all(&dst_dir), format!("failed to remove {}", dst_dir.display()));
        }
    }

    /// Returns whether the SHA256 checksum of `path` matches `expected`.
    pub(crate) fn verify(&self, path: &Path, expected: &str) -> bool {
        use sha2::Digest;

        self.verbose(|| println!("verifying {}", path.display()));

        if self.dry_run() {
            return false;
        }

        let mut hasher = sha2::Sha256::new();

        let file = t!(File::open(path));
        let mut reader = BufReader::new(file);

        loop {
            let buffer = t!(reader.fill_buf());
            let l = buffer.len();
            // break if EOF
            if l == 0 {
                break;
            }
            hasher.update(buffer);
            reader.consume(l);
        }

        let checksum = hex_encode(hasher.finalize().as_slice());
        let verified = checksum == expected;

        if !verified {
            println!(
                "invalid checksum: \n\
                found:    {checksum}\n\
                expected: {expected}",
            );
        }

        verified
    }
}

fn recorded_entries(dst: &Path, pattern: &str) -> Option<BufWriter<File>> {
    let name = if pattern == "rustc-dev" {
        ".rustc-dev-contents"
    } else if pattern.starts_with("rust-std") {
        ".rust-std-contents"
    } else {
        return None;
    };
    Some(BufWriter::new(t!(File::create(dst.join(name)))))
}

enum DownloadSource {
    CI,
    Dist,
}

/// Functions that are only ever called once, but named for clarity and to avoid thousand-line functions.
impl Config {
    pub(crate) fn download_clippy(&self) -> PathBuf {
        self.verbose(|| println!("downloading stage0 clippy artifacts"));

        let date = &self.stage0_metadata.compiler.date;
        let version = &self.stage0_metadata.compiler.version;
        let host = self.build;

        let clippy_stamp =
            BuildStamp::new(&self.initial_sysroot).with_prefix("clippy").add_stamp(date);
        let cargo_clippy = self.initial_sysroot.join("bin").join(exe("cargo-clippy", host));
        if cargo_clippy.exists() && clippy_stamp.is_up_to_date() {
            return cargo_clippy;
        }

        let filename = format!("clippy-{version}-{host}.tar.xz");
        self.download_component(DownloadSource::Dist, filename, "clippy-preview", date, "stage0");
        if self.should_fix_bins_and_dylibs() {
            self.fix_bin_or_dylib(&cargo_clippy);
            self.fix_bin_or_dylib(&cargo_clippy.with_file_name(exe("clippy-driver", host)));
        }

        t!(clippy_stamp.write());
        cargo_clippy
    }

    #[cfg(test)]
    pub(crate) fn maybe_download_rustfmt(&self) -> Option<PathBuf> {
        Some(PathBuf::new())
    }

    /// NOTE: rustfmt is a completely different toolchain than the bootstrap compiler, so it can't
    /// reuse target directories or artifacts
    #[cfg(not(test))]
    pub(crate) fn maybe_download_rustfmt(&self) -> Option<PathBuf> {
        use build_helper::stage0_parser::VersionMetadata;

        if self.dry_run() {
            return Some(PathBuf::new());
        }

        let VersionMetadata { date, version } = self.stage0_metadata.rustfmt.as_ref()?;
        let channel = format!("{version}-{date}");

        let host = self.build;
        let bin_root = self.out.join(host).join("rustfmt");
        let rustfmt_path = bin_root.join("bin").join(exe("rustfmt", host));
        let rustfmt_stamp = BuildStamp::new(&bin_root).with_prefix("rustfmt").add_stamp(channel);
        if rustfmt_path.exists() && rustfmt_stamp.is_up_to_date() {
            return Some(rustfmt_path);
        }

        self.download_component(
            DownloadSource::Dist,
            format!("rustfmt-{version}-{build}.tar.xz", build = host.triple),
            "rustfmt-preview",
            date,
            "rustfmt",
        );
        self.download_component(
            DownloadSource::Dist,
            format!("rustc-{version}-{build}.tar.xz", build = host.triple),
            "rustc",
            date,
            "rustfmt",
        );

        if self.should_fix_bins_and_dylibs() {
            self.fix_bin_or_dylib(&bin_root.join("bin").join("rustfmt"));
            self.fix_bin_or_dylib(&bin_root.join("bin").join("cargo-fmt"));
            let lib_dir = bin_root.join("lib");
            for lib in t!(fs::read_dir(&lib_dir), lib_dir.display().to_string()) {
                let lib = t!(lib);
                if path_is_dylib(&lib.path()) {
                    self.fix_bin_or_dylib(&lib.path());
                }
            }
        }

        t!(rustfmt_stamp.write());
        Some(rustfmt_path)
    }

    pub(crate) fn ci_rust_std_contents(&self) -> Vec<String> {
        self.ci_component_contents(".rust-std-contents")
    }

    pub(crate) fn ci_rustc_dev_contents(&self) -> Vec<String> {
        self.ci_component_contents(".rustc-dev-contents")
    }

    fn ci_component_contents(&self, stamp_file: &str) -> Vec<String> {
        assert!(self.download_rustc());
        if self.dry_run() {
            return vec![];
        }

        let ci_rustc_dir = self.ci_rustc_dir();
        let stamp_file = ci_rustc_dir.join(stamp_file);
        let contents_file = t!(File::open(&stamp_file), stamp_file.display().to_string());
        t!(BufReader::new(contents_file).lines().collect())
    }

    pub(crate) fn download_ci_rustc(&self, commit: &str) {
        self.verbose(|| println!("using downloaded stage2 artifacts from CI (commit {commit})"));

        let version = self.artifact_version_part(commit);
        // download-rustc doesn't need its own cargo, it can just use beta's. But it does need the
        // `rustc_private` crates for tools.
        let extra_components = ["rustc-dev"];

        self.download_toolchain(
            &version,
            "ci-rustc",
            &format!("{commit}-{}", self.llvm_assertions),
            &extra_components,
            Self::download_ci_component,
        );
    }

    #[cfg(test)]
    pub(crate) fn download_beta_toolchain(&self) {}

    #[cfg(not(test))]
    pub(crate) fn download_beta_toolchain(&self) {
        self.verbose(|| println!("downloading stage0 beta artifacts"));

        let date = &self.stage0_metadata.compiler.date;
        let version = &self.stage0_metadata.compiler.version;
        let extra_components = ["cargo"];

        let download_beta_component = |config: &Config, filename, prefix: &_, date: &_| {
            config.download_component(DownloadSource::Dist, filename, prefix, date, "stage0")
        };

        self.download_toolchain(
            version,
            "stage0",
            date,
            &extra_components,
            download_beta_component,
        );
    }

    fn download_toolchain(
        &self,
        version: &str,
        sysroot: &str,
        stamp_key: &str,
        extra_components: &[&str],
        download_component: fn(&Config, String, &str, &str),
    ) {
        let host = self.build.triple;
        let bin_root = self.out.join(host).join(sysroot);
        let rustc_stamp = BuildStamp::new(&bin_root).with_prefix("rustc").add_stamp(stamp_key);

        if !bin_root.join("bin").join(exe("rustc", self.build)).exists()
            || !rustc_stamp.is_up_to_date()
        {
            if bin_root.exists() {
                t!(fs::remove_dir_all(&bin_root));
            }
            let filename = format!("rust-std-{version}-{host}.tar.xz");
            let pattern = format!("rust-std-{host}");
            download_component(self, filename, &pattern, stamp_key);
            let filename = format!("rustc-{version}-{host}.tar.xz");
            download_component(self, filename, "rustc", stamp_key);

            for component in extra_components {
                let filename = format!("{component}-{version}-{host}.tar.xz");
                download_component(self, filename, component, stamp_key);
            }

            if self.should_fix_bins_and_dylibs() {
                self.fix_bin_or_dylib(&bin_root.join("bin").join("rustc"));
                self.fix_bin_or_dylib(&bin_root.join("bin").join("rustdoc"));
                self.fix_bin_or_dylib(
                    &bin_root.join("libexec").join("rust-analyzer-proc-macro-srv"),
                );
                let lib_dir = bin_root.join("lib");
                for lib in t!(fs::read_dir(&lib_dir), lib_dir.display().to_string()) {
                    let lib = t!(lib);
                    if path_is_dylib(&lib.path()) {
                        self.fix_bin_or_dylib(&lib.path());
                    }
                }
            }

            t!(rustc_stamp.write());
        }
    }

    /// Download a single component of a CI-built toolchain (not necessarily a published nightly).
    // NOTE: intentionally takes an owned string to avoid downloading multiple times by accident
    fn download_ci_component(&self, filename: String, prefix: &str, commit_with_assertions: &str) {
        Self::download_component(
            self,
            DownloadSource::CI,
            filename,
            prefix,
            commit_with_assertions,
            "ci-rustc",
        )
    }

    fn download_component(
        &self,
        mode: DownloadSource,
        filename: String,
        prefix: &str,
        key: &str,
        destination: &str,
    ) {
        if self.dry_run() {
            return;
        }

        let cache_dst =
            self.bootstrap_cache_path.as_ref().cloned().unwrap_or_else(|| self.out.join("cache"));

        let cache_dir = cache_dst.join(key);
        if !cache_dir.exists() {
            t!(fs::create_dir_all(&cache_dir));
        }

        let bin_root = self.out.join(self.build).join(destination);
        let tarball = cache_dir.join(&filename);
        let (base_url, url, should_verify) = match mode {
            DownloadSource::CI => {
                let dist_server = if self.llvm_assertions {
                    self.stage0_metadata.config.artifacts_with_llvm_assertions_server.clone()
                } else {
                    self.stage0_metadata.config.artifacts_server.clone()
                };
                let url = format!(
                    "{}/{filename}",
                    key.strip_suffix(&format!("-{}", self.llvm_assertions)).unwrap()
                );
                (dist_server, url, false)
            }
            DownloadSource::Dist => {
                let dist_server = env::var("RUSTUP_DIST_SERVER")
                    .unwrap_or(self.stage0_metadata.config.dist_server.to_string());
                // NOTE: make `dist` part of the URL because that's how it's stored in src/stage0
                (dist_server, format!("dist/{key}/{filename}"), true)
            }
        };

        // For the beta compiler, put special effort into ensuring the checksums are valid.
        let checksum = if should_verify {
            let error = format!(
                "src/stage0 doesn't contain a checksum for {url}. \
                Pre-built artifacts might not be available for this \
                target at this time, see https://doc.rust-lang.org/nightly\
                /rustc/platform-support.html for more information."
            );
            let sha256 = self.stage0_metadata.checksums_sha256.get(&url).expect(&error);
            if tarball.exists() {
                if self.verify(&tarball, sha256) {
                    self.unpack(&tarball, &bin_root, prefix);
                    return;
                } else {
                    self.verbose(|| {
                        println!(
                            "ignoring cached file {} due to failed verification",
                            tarball.display()
                        )
                    });
                    self.remove(&tarball);
                }
            }
            Some(sha256)
        } else if tarball.exists() {
            self.unpack(&tarball, &bin_root, prefix);
            return;
        } else {
            None
        };

        let mut help_on_error = "";
        if destination == "ci-rustc" {
            help_on_error = "ERROR: failed to download pre-built rustc from CI

NOTE: old builds get deleted after a certain time
HELP: if trying to compile an old commit of rustc, disable `download-rustc` in bootstrap.toml:

[rust]
download-rustc = false
";
        }
        self.download_file(&format!("{base_url}/{url}"), &tarball, help_on_error);
        if let Some(sha256) = checksum {
            if !self.verify(&tarball, sha256) {
                panic!("failed to verify {}", tarball.display());
            }
        }

        self.unpack(&tarball, &bin_root, prefix);
    }

    #[cfg(test)]
    pub(crate) fn maybe_download_ci_llvm(&self) {}

    #[cfg(not(test))]
    pub(crate) fn maybe_download_ci_llvm(&self) {
        use build_helper::exit;
        use build_helper::git::PathFreshness;

        use crate::core::build_steps::llvm::detect_llvm_freshness;
        use crate::core::config::check_incompatible_options_for_ci_llvm;

        if !self.llvm_from_ci {
            return;
        }

        let llvm_root = self.ci_llvm_root();
        let llvm_freshness =
            detect_llvm_freshness(self, self.rust_info.is_managed_git_subrepository());
        self.verbose(|| {
            eprintln!("LLVM freshness: {llvm_freshness:?}");
        });
        let llvm_sha = match llvm_freshness {
            PathFreshness::LastModifiedUpstream { upstream } => upstream,
            PathFreshness::HasLocalModifications { upstream } => upstream,
            PathFreshness::MissingUpstream => {
                eprintln!("error: could not find commit hash for downloading LLVM");
                eprintln!("HELP: maybe your repository history is too shallow?");
                eprintln!("HELP: consider disabling `download-ci-llvm`");
                eprintln!("HELP: or fetch enough history to include one upstream commit");
                crate::exit!(1);
            }
        };
        let stamp_key = format!("{}{}", llvm_sha, self.llvm_assertions);
        let llvm_stamp = BuildStamp::new(&llvm_root).with_prefix("llvm").add_stamp(stamp_key);
        if !llvm_stamp.is_up_to_date() && !self.dry_run() {
            self.download_ci_llvm(&llvm_sha);

            if self.should_fix_bins_and_dylibs() {
                for entry in t!(fs::read_dir(llvm_root.join("bin"))) {
                    self.fix_bin_or_dylib(&t!(entry).path());
                }
            }

            // Update the timestamp of llvm-config to force rustc_llvm to be
            // rebuilt. This is a hacky workaround for a deficiency in Cargo where
            // the rerun-if-changed directive doesn't handle changes very well.
            // https://github.com/rust-lang/cargo/issues/10791
            // Cargo only compares the timestamp of the file relative to the last
            // time `rustc_llvm` build script ran. However, the timestamps of the
            // files in the tarball are in the past, so it doesn't trigger a
            // rebuild.
            let now = std::time::SystemTime::now();
            let file_times = fs::FileTimes::new().set_accessed(now).set_modified(now);

            let llvm_config = llvm_root.join("bin").join(exe("llvm-config", self.build));
            t!(crate::utils::helpers::set_file_times(llvm_config, file_times));

            if self.should_fix_bins_and_dylibs() {
                let llvm_lib = llvm_root.join("lib");
                for entry in t!(fs::read_dir(llvm_lib)) {
                    let lib = t!(entry).path();
                    if path_is_dylib(&lib) {
                        self.fix_bin_or_dylib(&lib);
                    }
                }
            }

            t!(llvm_stamp.write());
        }

        if let Some(config_path) = &self.config {
            let current_config_toml = Self::get_toml(config_path).unwrap();

            match self.get_builder_toml("ci-llvm") {
                Ok(ci_config_toml) => {
                    t!(check_incompatible_options_for_ci_llvm(current_config_toml, ci_config_toml));
                }
                Err(e) if e.to_string().contains("unknown field") => {
                    println!(
                        "WARNING: CI LLVM has some fields that are no longer supported in bootstrap; download-ci-llvm will be disabled."
                    );
                    println!("HELP: Consider rebasing to a newer commit if available.");
                }
                Err(e) => {
                    eprintln!("ERROR: Failed to parse CI LLVM bootstrap.toml: {e}");
                    exit!(2);
                }
            };
        };
    }

    #[cfg(not(test))]
    fn download_ci_llvm(&self, llvm_sha: &str) {
        let llvm_assertions = self.llvm_assertions;

        let cache_prefix = format!("llvm-{llvm_sha}-{llvm_assertions}");
        let cache_dst =
            self.bootstrap_cache_path.as_ref().cloned().unwrap_or_else(|| self.out.join("cache"));

        let rustc_cache = cache_dst.join(cache_prefix);
        if !rustc_cache.exists() {
            t!(fs::create_dir_all(&rustc_cache));
        }
        let base = if llvm_assertions {
            &self.stage0_metadata.config.artifacts_with_llvm_assertions_server
        } else {
            &self.stage0_metadata.config.artifacts_server
        };
        let version = self.artifact_version_part(llvm_sha);
        let filename = format!("rust-dev-{}-{}.tar.xz", version, self.build.triple);
        let tarball = rustc_cache.join(&filename);
        if !tarball.exists() {
            let help_on_error = "ERROR: failed to download llvm from ci

    HELP: There could be two reasons behind this:
        1) The host triple is not supported for `download-ci-llvm`.
        2) Old builds get deleted after a certain time.
    HELP: In either case, disable `download-ci-llvm` in your bootstrap.toml:

    [llvm]
    download-ci-llvm = false
    ";
            self.download_file(&format!("{base}/{llvm_sha}/{filename}"), &tarball, help_on_error);
        }
        let llvm_root = self.ci_llvm_root();
        self.unpack(&tarball, &llvm_root, "rust-dev");
    }

    pub fn download_ci_gcc(&self, gcc_sha: &str, root_dir: &Path) {
        let cache_prefix = format!("gcc-{gcc_sha}");
        let cache_dst =
            self.bootstrap_cache_path.as_ref().cloned().unwrap_or_else(|| self.out.join("cache"));

        let gcc_cache = cache_dst.join(cache_prefix);
        if !gcc_cache.exists() {
            t!(fs::create_dir_all(&gcc_cache));
        }
        let base = &self.stage0_metadata.config.artifacts_server;
        let version = self.artifact_version_part(gcc_sha);
        let filename = format!("gcc-{version}-{}.tar.xz", self.build.triple);
        let tarball = gcc_cache.join(&filename);
        if !tarball.exists() {
            let help_on_error = "ERROR: failed to download gcc from ci

    HELP: There could be two reasons behind this:
        1) The host triple is not supported for `download-ci-gcc`.
        2) Old builds get deleted after a certain time.
    HELP: In either case, disable `download-ci-gcc` in your bootstrap.toml:

    [gcc]
    download-ci-gcc = false
    ";
            self.download_file(&format!("{base}/{gcc_sha}/{filename}"), &tarball, help_on_error);
        }
        self.unpack(&tarball, root_dir, "gcc");
    }
}

fn path_is_dylib(path: &Path) -> bool {
    // The .so is not necessarily the extension, it might be libLLVM.so.18.1
    path.to_str().is_some_and(|path| path.contains(".so"))
}

/// Checks whether the CI rustc is available for the given target triple.
pub(crate) fn is_download_ci_available(target_triple: &str, llvm_assertions: bool) -> bool {
    // All tier 1 targets and tier 2 targets with host tools.
    const SUPPORTED_PLATFORMS: &[&str] = &[
        "aarch64-apple-darwin",
        "aarch64-pc-windows-msvc",
        "aarch64-unknown-linux-gnu",
        "aarch64-unknown-linux-musl",
        "arm-unknown-linux-gnueabi",
        "arm-unknown-linux-gnueabihf",
        "armv7-unknown-linux-gnueabihf",
        "i686-pc-windows-gnu",
        "i686-pc-windows-msvc",
        "i686-unknown-linux-gnu",
        "loongarch64-unknown-linux-gnu",
        "powerpc-unknown-linux-gnu",
        "powerpc64-unknown-linux-gnu",
        "powerpc64le-unknown-linux-gnu",
        "riscv64gc-unknown-linux-gnu",
        "s390x-unknown-linux-gnu",
        "x86_64-apple-darwin",
        "x86_64-pc-windows-gnu",
        "x86_64-pc-windows-msvc",
        "x86_64-unknown-freebsd",
        "x86_64-unknown-illumos",
        "x86_64-unknown-linux-gnu",
        "x86_64-unknown-linux-musl",
        "x86_64-unknown-netbsd",
    ];

    const SUPPORTED_PLATFORMS_WITH_ASSERTIONS: &[&str] =
        &["x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc"];

    if llvm_assertions {
        SUPPORTED_PLATFORMS_WITH_ASSERTIONS.contains(&target_triple)
    } else {
        SUPPORTED_PLATFORMS.contains(&target_triple)
    }
}
