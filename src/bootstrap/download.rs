use std::{
    env,
    ffi::{OsStr, OsString},
    fs::{self, File},
    io::{BufRead, BufReader, BufWriter, ErrorKind, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use build_helper::util::try_run;
use once_cell::sync::OnceCell;
use xz2::bufread::XzDecoder;

use crate::{
    config::RustfmtMetadata,
    llvm::detect_llvm_sha,
    t,
    util::{check_run, exe, program_out_of_date},
    Config,
};

static SHOULD_FIX_BINS_AND_DYLIBS: OnceCell<bool> = OnceCell::new();

/// Generic helpers that are useful anywhere in bootstrap.
impl Config {
    pub fn is_verbose(&self) -> bool {
        self.verbose > 0
    }

    pub(crate) fn create(&self, path: &Path, s: &str) {
        if self.dry_run() {
            return;
        }
        t!(fs::write(path, s));
    }

    pub(crate) fn remove(&self, f: &Path) {
        if self.dry_run() {
            return;
        }
        fs::remove_file(f).unwrap_or_else(|_| panic!("failed to remove {:?}", f));
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
    /// Exits if the command failed to execute at all, otherwise returns its
    /// `status.success()`.
    pub(crate) fn try_run(&self, cmd: &mut Command) -> bool {
        if self.dry_run() {
            return true;
        }
        self.verbose(&format!("running: {:?}", cmd));
        try_run(cmd, self.is_verbose())
    }

    /// Runs a command, printing out nice contextual information if it fails.
    /// Returns false if do not execute at all, otherwise returns its
    /// `status.success()`.
    pub(crate) fn check_run(&self, cmd: &mut Command) -> bool {
        if self.dry_run() {
            return true;
        }
        self.verbose(&format!("running: {:?}", cmd));
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
            if self.patch_binaries_for_nix {
                return true;
            }

            // Use `/etc/os-release` instead of `/etc/NIXOS`.
            // The latter one does not exist on NixOS when using tmpfs as root.
            let is_nixos = match File::open("/etc/os-release") {
                Err(e) if e.kind() == ErrorKind::NotFound => false,
                Err(e) => panic!("failed to access /etc/os-release: {}", e),
                Ok(os_release) => BufReader::new(os_release).lines().any(|l| {
                    let l = l.expect("reading /etc/os-release");
                    matches!(l.trim(), "ID=nixos" | "ID='nixos'" | "ID=\"nixos\"")
                }),
            };
            is_nixos && !Path::new("/lib").exists()
        });
        if val {
            eprintln!("info: You seem to be using Nix.");
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
        static NIX_DEPS_DIR: OnceCell<PathBuf> = OnceCell::new();
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
            nix_build_succeeded = self.try_run(Command::new("nix-build").args(&[
                Path::new("-E"),
                Path::new(NIX_EXPR),
                Path::new("-o"),
                &nix_deps_dir,
            ]));
            nix_deps_dir
        });
        if !nix_build_succeeded {
            return;
        }

        let mut patchelf = Command::new(nix_deps_dir.join("bin/patchelf"));
        let rpath_entries = {
            // ORIGIN is a relative default, all binary and dynamic libraries we ship
            // appear to have this (even when `../lib` is redundant).
            // NOTE: there are only two paths here, delimited by a `:`
            let mut entries = OsString::from("$ORIGIN/../lib:");
            entries.push(t!(fs::canonicalize(nix_deps_dir)).join("lib"));
            entries
        };
        patchelf.args(&[OsString::from("--set-rpath"), rpath_entries]);
        if !fname.extension().map_or(false, |ext| ext == "so") {
            // Finally, set the correct .interp for binaries
            let dynamic_linker_path = nix_deps_dir.join("nix-support/dynamic-linker");
            // FIXME: can we support utf8 here? `args` doesn't accept Vec<u8>, only OsString ...
            let dynamic_linker = t!(String::from_utf8(t!(fs::read(dynamic_linker_path))));
            patchelf.args(&["--set-interpreter", dynamic_linker.trim_end()]);
        }

        self.try_run(patchelf.arg(fname));
    }

    fn download_file(&self, url: &str, dest_path: &Path, help_on_error: &str) {
        self.verbose(&format!("download {url}"));
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
        t!(std::fs::rename(&tempfile, dest_path));
    }

    fn download_http_with_retries(&self, tempfile: &Path, url: &str, help_on_error: &str) {
        println!("downloading {}", url);
        // Try curl. If that fails and we are on windows, fallback to PowerShell.
        let mut curl = Command::new("curl");
        curl.args(&[
            "-#",
            "-y",
            "30",
            "-Y",
            "10", // timeout if speed is < 10 bytes/sec for > 30 seconds
            "--connect-timeout",
            "30", // timeout if cannot connect within 30 seconds
            "--retry",
            "3",
            "-SRf",
        ]);
        curl.arg(url);
        let f = File::create(tempfile).unwrap();
        curl.stdout(Stdio::from(f));
        if !self.check_run(&mut curl) {
            if self.build.contains("windows-msvc") {
                eprintln!("Fallback to PowerShell");
                for _ in 0..3 {
                    if self.try_run(Command::new("PowerShell.exe").args(&[
                        "/nologo",
                        "-Command",
                        "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12;",
                        &format!(
                            "(New-Object System.Net.WebClient).DownloadFile('{}', '{}')",
                            url, tempfile.to_str().expect("invalid UTF-8 not supported with powershell downloads"),
                        ),
                    ])) {
                        return;
                    }
                    eprintln!("\nspurious failure, trying again");
                }
            }
            if !help_on_error.is_empty() {
                eprintln!("{}", help_on_error);
            }
            crate::detail_exit_macro!(1);
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

        // `compile::Sysroot` needs to know the contents of the `rustc-dev` tarball to avoid adding
        // it to the sysroot unless it was explicitly requested. But parsing the 100 MB tarball is slow.
        // Cache the entries when we extract it so we only have to read it once.
        let mut recorded_entries =
            if dst.ends_with("ci-rustc") { recorded_entries(dst, pattern) } else { None };

        for member in t!(tar.entries()) {
            let mut member = t!(member);
            let original_path = t!(member.path()).into_owned();
            // skip the top-level directory
            if original_path == directory_prefix {
                continue;
            }
            let mut short_path = t!(original_path.strip_prefix(directory_prefix));
            if !short_path.starts_with(pattern) {
                continue;
            }
            short_path = t!(short_path.strip_prefix(pattern));
            let dst_path = dst.join(short_path);
            self.verbose(&format!("extracting {} to {}", original_path.display(), dst.display()));
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
            t!(fs::rename(src_path, dst_path));
        }
        let dst_dir = dst.join(directory_prefix);
        if dst_dir.exists() {
            t!(fs::remove_dir_all(&dst_dir), format!("failed to remove {}", dst_dir.display()));
        }
    }

    /// Returns whether the SHA256 checksum of `path` matches `expected`.
    fn verify(&self, path: &Path, expected: &str) -> bool {
        use sha2::Digest;

        self.verbose(&format!("verifying {}", path.display()));
        let mut hasher = sha2::Sha256::new();
        // FIXME: this is ok for rustfmt (4.1 MB large at time of writing), but it seems memory-intensive for rustc and larger components.
        // Consider using streaming IO instead?
        let contents = if self.dry_run() { vec![] } else { t!(fs::read(path)) };
        hasher.update(&contents);
        let found = hex::encode(hasher.finalize().as_slice());
        let verified = found == expected;
        if !verified && !self.dry_run() {
            println!(
                "invalid checksum: \n\
                found:    {found}\n\
                expected: {expected}",
            );
        }
        return verified;
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

/// Functions that are only ever called once, but named for clarify and to avoid thousand-line functions.
impl Config {
    pub(crate) fn maybe_download_rustfmt(&self) -> Option<PathBuf> {
        let RustfmtMetadata { date, version } = self.stage0_metadata.rustfmt.as_ref()?;
        let channel = format!("{version}-{date}");

        let host = self.build;
        let bin_root = self.out.join(host.triple).join("rustfmt");
        let rustfmt_path = bin_root.join("bin").join(exe("rustfmt", host));
        let rustfmt_stamp = bin_root.join(".rustfmt-stamp");
        if rustfmt_path.exists() && !program_out_of_date(&rustfmt_stamp, &channel) {
            return Some(rustfmt_path);
        }

        self.download_component(
            DownloadSource::Dist,
            format!("rustfmt-{version}-{build}.tar.xz", build = host.triple),
            "rustfmt-preview",
            &date,
            "rustfmt",
        );
        self.download_component(
            DownloadSource::Dist,
            format!("rustc-{version}-{build}.tar.xz", build = host.triple),
            "rustc",
            &date,
            "rustfmt",
        );

        if self.should_fix_bins_and_dylibs() {
            self.fix_bin_or_dylib(&bin_root.join("bin").join("rustfmt"));
            self.fix_bin_or_dylib(&bin_root.join("bin").join("cargo-fmt"));
            let lib_dir = bin_root.join("lib");
            for lib in t!(fs::read_dir(&lib_dir), lib_dir.display().to_string()) {
                let lib = t!(lib);
                if lib.path().extension() == Some(OsStr::new("so")) {
                    self.fix_bin_or_dylib(&lib.path());
                }
            }
        }

        self.create(&rustfmt_stamp, &channel);
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
        let ci_rustc_dir = self.out.join(&*self.build.triple).join("ci-rustc");
        let stamp_file = ci_rustc_dir.join(stamp_file);
        let contents_file = t!(File::open(&stamp_file), stamp_file.display().to_string());
        t!(BufReader::new(contents_file).lines().collect())
    }

    pub(crate) fn download_ci_rustc(&self, commit: &str) {
        self.verbose(&format!("using downloaded stage2 artifacts from CI (commit {commit})"));

        let version = self.artifact_version_part(commit);
        // download-rustc doesn't need its own cargo, it can just use beta's. But it does need the
        // `rustc_private` crates for tools.
        let extra_components = ["rustc-dev"];

        self.download_toolchain(
            &version,
            "ci-rustc",
            commit,
            &extra_components,
            Self::download_ci_component,
        );
    }

    pub(crate) fn download_beta_toolchain(&self) {
        self.verbose(&format!("downloading stage0 beta artifacts"));

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
        let rustc_stamp = bin_root.join(".rustc-stamp");

        if !bin_root.join("bin").join(exe("rustc", self.build)).exists()
            || program_out_of_date(&rustc_stamp, stamp_key)
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
                    if lib.path().extension() == Some(OsStr::new("so")) {
                        self.fix_bin_or_dylib(&lib.path());
                    }
                }
            }

            t!(fs::write(rustc_stamp, stamp_key));
        }
    }

    /// Download a single component of a CI-built toolchain (not necessarily a published nightly).
    // NOTE: intentionally takes an owned string to avoid downloading multiple times by accident
    fn download_ci_component(&self, filename: String, prefix: &str, commit: &str) {
        Self::download_component(self, DownloadSource::CI, filename, prefix, commit, "ci-rustc")
    }

    fn download_component(
        &self,
        mode: DownloadSource,
        filename: String,
        prefix: &str,
        key: &str,
        destination: &str,
    ) {
        let cache_dst = self.out.join("cache");
        let cache_dir = cache_dst.join(key);
        if !cache_dir.exists() {
            t!(fs::create_dir_all(&cache_dir));
        }

        let bin_root = self.out.join(self.build.triple).join(destination);
        let tarball = cache_dir.join(&filename);
        let (base_url, url, should_verify) = match mode {
            DownloadSource::CI => (
                self.stage0_metadata.config.artifacts_server.clone(),
                format!("{key}/{filename}"),
                false,
            ),
            DownloadSource::Dist => {
                let dist_server = env::var("RUSTUP_DIST_SERVER")
                    .unwrap_or(self.stage0_metadata.config.dist_server.to_string());
                // NOTE: make `dist` part of the URL because that's how it's stored in src/stage0.json
                (dist_server, format!("dist/{key}/{filename}"), true)
            }
        };

        // For the beta compiler, put special effort into ensuring the checksums are valid.
        // FIXME: maybe we should do this for download-rustc as well? but it would be a pain to update
        // this on each and every nightly ...
        let checksum = if should_verify {
            let error = format!(
                "src/stage0.json doesn't contain a checksum for {url}. \
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
                    self.verbose(&format!(
                        "ignoring cached file {} due to failed verification",
                        tarball.display()
                    ));
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
            help_on_error = "error: failed to download pre-built rustc from CI

note: old builds get deleted after a certain time
help: if trying to compile an old commit of rustc, disable `download-rustc` in config.toml:

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

    pub(crate) fn maybe_download_ci_llvm(&self) {
        if !self.llvm_from_ci {
            return;
        }
        let llvm_root = self.ci_llvm_root();
        let llvm_stamp = llvm_root.join(".llvm-stamp");
        let llvm_sha = detect_llvm_sha(&self, self.rust_info.is_managed_git_subrepository());
        let key = format!("{}{}", llvm_sha, self.llvm_assertions);
        if program_out_of_date(&llvm_stamp, &key) && !self.dry_run() {
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
            let now = filetime::FileTime::from_system_time(std::time::SystemTime::now());
            let llvm_config = llvm_root.join("bin").join(exe("llvm-config", self.build));
            t!(filetime::set_file_times(&llvm_config, now, now));

            if self.should_fix_bins_and_dylibs() {
                let llvm_lib = llvm_root.join("lib");
                for entry in t!(fs::read_dir(&llvm_lib)) {
                    let lib = t!(entry).path();
                    if lib.extension().map_or(false, |ext| ext == "so") {
                        self.fix_bin_or_dylib(&lib);
                    }
                }
            }

            t!(fs::write(llvm_stamp, key));
        }
    }

    fn download_ci_llvm(&self, llvm_sha: &str) {
        let llvm_assertions = self.llvm_assertions;

        let cache_prefix = format!("llvm-{}-{}", llvm_sha, llvm_assertions);
        let cache_dst = self.out.join("cache");
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
            let help_on_error = "error: failed to download llvm from ci

    help: old builds get deleted after a certain time
    help: if trying to compile an old commit of rustc, disable `download-ci-llvm` in config.toml:

    [llvm]
    download-ci-llvm = false
    ";
            self.download_file(&format!("{base}/{llvm_sha}/{filename}"), &tarball, help_on_error);
        }
        let llvm_root = self.ci_llvm_root();
        self.unpack(&tarball, &llvm_root, "rust-dev");
    }
}
