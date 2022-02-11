//! Build a dist manifest, hash and sign everything.
//! This gets called by `promote-release`
//! (https://github.com/rust-lang/rust-central-station/tree/master/promote-release)
//! via `x.py dist hash-and-sign`; the cmdline arguments are set up
//! by rustbuild (in `src/bootstrap/dist.rs`).

mod checksum;
mod manifest;
mod versions;

use crate::checksum::Checksums;
use crate::manifest::{Component, Manifest, Package, Rename, Target};
use crate::versions::{PkgType, Versions};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

static HOSTS: &[&str] = &[
    "aarch64-apple-darwin",
    "aarch64-pc-windows-msvc",
    "aarch64-unknown-linux-gnu",
    "aarch64-unknown-linux-musl",
    "arm-unknown-linux-gnueabi",
    "arm-unknown-linux-gnueabihf",
    "armv7-unknown-linux-gnueabihf",
    "i686-apple-darwin",
    "i686-pc-windows-gnu",
    "i686-pc-windows-msvc",
    "i686-unknown-linux-gnu",
    "mips-unknown-linux-gnu",
    "mips64-unknown-linux-gnuabi64",
    "mips64el-unknown-linux-gnuabi64",
    "mipsel-unknown-linux-gnu",
    "mipsisa32r6-unknown-linux-gnu",
    "mipsisa32r6el-unknown-linux-gnu",
    "mipsisa64r6-unknown-linux-gnuabi64",
    "mipsisa64r6el-unknown-linux-gnuabi64",
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

static TARGETS: &[&str] = &[
    "aarch64-apple-darwin",
    "aarch64-apple-ios",
    "aarch64-apple-ios-sim",
    "aarch64-fuchsia",
    "aarch64-linux-android",
    "aarch64-pc-windows-msvc",
    "aarch64-unknown-hermit",
    "aarch64-unknown-linux-gnu",
    "aarch64-unknown-linux-musl",
    "aarch64-unknown-none",
    "aarch64-unknown-none-softfloat",
    "aarch64-unknown-redox",
    "arm-linux-androideabi",
    "arm-unknown-linux-gnueabi",
    "arm-unknown-linux-gnueabihf",
    "arm-unknown-linux-musleabi",
    "arm-unknown-linux-musleabihf",
    "armv5te-unknown-linux-gnueabi",
    "armv5te-unknown-linux-musleabi",
    "armv7-apple-ios",
    "armv7-linux-androideabi",
    "thumbv7neon-linux-androideabi",
    "armv7-unknown-linux-gnueabi",
    "armv7-unknown-linux-gnueabihf",
    "armv7a-none-eabi",
    "thumbv7neon-unknown-linux-gnueabihf",
    "armv7-unknown-linux-musleabi",
    "armv7-unknown-linux-musleabihf",
    "armebv7r-none-eabi",
    "armebv7r-none-eabihf",
    "armv7r-none-eabi",
    "armv7r-none-eabihf",
    "armv7s-apple-ios",
    "asmjs-unknown-emscripten",
    "bpfeb-unknown-none",
    "bpfel-unknown-none",
    "i386-apple-ios",
    "i586-pc-windows-msvc",
    "i586-unknown-linux-gnu",
    "i586-unknown-linux-musl",
    "i686-apple-darwin",
    "i686-linux-android",
    "i686-pc-windows-gnu",
    "i686-pc-windows-msvc",
    "i686-unknown-freebsd",
    "i686-unknown-linux-gnu",
    "i686-unknown-linux-musl",
    "m68k-unknown-linux-gnu",
    "mips-unknown-linux-gnu",
    "mips-unknown-linux-musl",
    "mips64-unknown-linux-gnuabi64",
    "mips64-unknown-linux-muslabi64",
    "mips64el-unknown-linux-gnuabi64",
    "mips64el-unknown-linux-muslabi64",
    "mipsisa32r6-unknown-linux-gnu",
    "mipsisa32r6el-unknown-linux-gnu",
    "mipsisa64r6-unknown-linux-gnuabi64",
    "mipsisa64r6el-unknown-linux-gnuabi64",
    "mipsel-unknown-linux-gnu",
    "mipsel-unknown-linux-musl",
    "nvptx64-nvidia-cuda",
    "powerpc-unknown-linux-gnu",
    "powerpc64-unknown-linux-gnu",
    "powerpc64le-unknown-linux-gnu",
    "riscv32i-unknown-none-elf",
    "riscv32imc-unknown-none-elf",
    "riscv32imac-unknown-none-elf",
    "riscv32gc-unknown-linux-gnu",
    "riscv64imac-unknown-none-elf",
    "riscv64gc-unknown-none-elf",
    "riscv64gc-unknown-linux-gnu",
    "s390x-unknown-linux-gnu",
    "sparc64-unknown-linux-gnu",
    "sparcv9-sun-solaris",
    "thumbv6m-none-eabi",
    "thumbv7em-none-eabi",
    "thumbv7em-none-eabihf",
    "thumbv7m-none-eabi",
    "thumbv8m.base-none-eabi",
    "thumbv8m.main-none-eabi",
    "thumbv8m.main-none-eabihf",
    "wasm32-unknown-emscripten",
    "wasm32-unknown-unknown",
    "wasm32-wasi",
    "x86_64-apple-darwin",
    "x86_64-apple-ios",
    "x86_64-fortanix-unknown-sgx",
    "x86_64-fuchsia",
    "x86_64-linux-android",
    "x86_64-pc-windows-gnu",
    "x86_64-pc-windows-msvc",
    "x86_64-sun-solaris",
    "x86_64-pc-solaris",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-illumos",
    "x86_64-unknown-linux-gnu",
    "x86_64-unknown-linux-gnux32",
    "x86_64-unknown-linux-musl",
    "x86_64-unknown-netbsd",
    "x86_64-unknown-redox",
    "x86_64-unknown-hermit",
];

/// This allows the manifest to contain rust-docs for hosts that don't build
/// docs.
///
/// Tuples of `(host_partial, host_instead)`. If the host does not have the
/// rust-docs component available, then if the host name contains
/// `host_partial`, it will use the docs from `host_instead` instead.
///
/// The order here matters, more specific entries should be first.
static DOCS_FALLBACK: &[(&str, &str)] = &[
    ("-apple-", "x86_64-apple-darwin"),
    ("aarch64", "aarch64-unknown-linux-gnu"),
    ("arm-", "aarch64-unknown-linux-gnu"),
    ("", "x86_64-unknown-linux-gnu"),
];

static MSI_INSTALLERS: &[&str] = &[
    "aarch64-pc-windows-msvc",
    "i686-pc-windows-gnu",
    "i686-pc-windows-msvc",
    "x86_64-pc-windows-gnu",
    "x86_64-pc-windows-msvc",
];

static PKG_INSTALLERS: &[&str] = &["x86_64-apple-darwin", "aarch64-apple-darwin"];

static MINGW: &[&str] = &["i686-pc-windows-gnu", "x86_64-pc-windows-gnu"];

static NIGHTLY_ONLY_COMPONENTS: &[&str] = &["miri-preview", "rust-analyzer-preview"];

macro_rules! t {
    ($e:expr) => {
        match $e {
            Ok(e) => e,
            Err(e) => panic!("{} failed with {}", stringify!($e), e),
        }
    };
}

struct Builder {
    versions: Versions,
    checksums: Checksums,
    shipped_files: HashSet<String>,

    input: PathBuf,
    output: PathBuf,
    s3_address: String,
    date: String,
}

fn main() {
    let num_threads = if let Some(num) = env::var_os("BUILD_MANIFEST_NUM_THREADS") {
        num.to_str().unwrap().parse().expect("invalid number for BUILD_MANIFEST_NUM_THREADS")
    } else {
        num_cpus::get()
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .expect("failed to initialize Rayon");

    let mut args = env::args().skip(1);
    let input = PathBuf::from(args.next().unwrap());
    let output = PathBuf::from(args.next().unwrap());
    let date = args.next().unwrap();
    let s3_address = args.next().unwrap();
    let channel = args.next().unwrap();

    Builder {
        versions: Versions::new(&channel, &input).unwrap(),
        checksums: t!(Checksums::new()),
        shipped_files: HashSet::new(),

        input,
        output,
        s3_address,
        date,
    }
    .build();
}

impl Builder {
    fn build(&mut self) {
        self.check_toolstate();
        let manifest = self.build_manifest();

        let channel = self.versions.channel().to_string();
        self.write_channel_files(&channel, &manifest);
        if channel == "stable" {
            // channel-rust-1.XX.YY.toml
            let rust_version = self.versions.rustc_version().to_string();
            self.write_channel_files(&rust_version, &manifest);

            // channel-rust-1.XX.toml
            let major_minor = rust_version.split('.').take(2).collect::<Vec<_>>().join(".");
            self.write_channel_files(&major_minor, &manifest);
        }

        if let Some(path) = std::env::var_os("BUILD_MANIFEST_SHIPPED_FILES_PATH") {
            self.write_shipped_files(&Path::new(&path));
        }

        t!(self.checksums.store_cache());
    }

    /// If a tool does not pass its tests on *any* of Linux and Windows, don't ship
    /// it on *all* targets, because tools like Miri can "cross-run" programs for
    /// different targets, for example, run a program for `x86_64-pc-windows-msvc`
    /// on `x86_64-unknown-linux-gnu`.
    /// Right now, we do this only for Miri.
    fn check_toolstate(&mut self) {
        for file in &["toolstates-linux.json", "toolstates-windows.json"] {
            let toolstates: Option<HashMap<String, String>> = File::open(self.input.join(file))
                .ok()
                .and_then(|f| serde_json::from_reader(&f).ok());
            let toolstates = toolstates.unwrap_or_else(|| {
                println!("WARNING: `{}` missing/malformed; assuming all tools failed", file);
                HashMap::default() // Use empty map if anything went wrong.
            });
            // Mark some tools as missing based on toolstate.
            if toolstates.get("miri").map(|s| &*s as &str) != Some("test-pass") {
                println!("Miri tests are not passing, removing component");
                self.versions.disable_version(&PkgType::Miri);
                break;
            }
        }
    }

    fn build_manifest(&mut self) -> Manifest {
        let mut manifest = Manifest {
            manifest_version: "2".to_string(),
            date: self.date.to_string(),
            pkg: BTreeMap::new(),
            artifacts: BTreeMap::new(),
            renames: BTreeMap::new(),
            profiles: BTreeMap::new(),
        };
        self.add_packages_to(&mut manifest);
        self.add_artifacts_to(&mut manifest);
        self.add_profiles_to(&mut manifest);
        self.add_renames_to(&mut manifest);
        manifest.pkg.insert("rust".to_string(), self.rust_package(&manifest));

        self.checksums.fill_missing_checksums(&mut manifest);

        manifest
    }

    fn add_packages_to(&mut self, manifest: &mut Manifest) {
        macro_rules! package {
            ($name:expr, $targets:expr) => {
                self.package($name, &mut manifest.pkg, $targets, &[])
            };
        }
        package!("rustc", HOSTS);
        package!("rustc-dev", HOSTS);
        package!("reproducible-artifacts", HOSTS);
        package!("rustc-docs", HOSTS);
        package!("cargo", HOSTS);
        package!("rust-mingw", MINGW);
        package!("rust-std", TARGETS);
        self.package("rust-docs", &mut manifest.pkg, HOSTS, DOCS_FALLBACK);
        package!("rust-src", &["*"]);
        package!("rls-preview", HOSTS);
        package!("rust-analyzer-preview", HOSTS);
        package!("clippy-preview", HOSTS);
        package!("miri-preview", HOSTS);
        package!("rustfmt-preview", HOSTS);
        package!("rust-analysis", TARGETS);
        package!("llvm-tools-preview", TARGETS);
    }

    fn add_artifacts_to(&mut self, manifest: &mut Manifest) {
        manifest.add_artifact("source-code", |artifact| {
            let tarball = self.versions.tarball_name(&PkgType::Rustc, "src").unwrap();
            artifact.add_tarball(self, "*", &tarball);
        });

        manifest.add_artifact("installer-msi", |artifact| {
            for target in MSI_INSTALLERS {
                let msi = self.versions.archive_name(&PkgType::Rust, target, "msi").unwrap();
                artifact.add_file(self, target, &msi);
            }
        });

        manifest.add_artifact("installer-pkg", |artifact| {
            for target in PKG_INSTALLERS {
                let pkg = self.versions.archive_name(&PkgType::Rust, target, "pkg").unwrap();
                artifact.add_file(self, target, &pkg);
            }
        });
    }

    fn add_profiles_to(&mut self, manifest: &mut Manifest) {
        let mut profile = |name, pkgs| self.profile(name, &mut manifest.profiles, pkgs);
        profile("minimal", &["rustc", "cargo", "rust-std", "rust-mingw"]);
        profile(
            "default",
            &[
                "rustc",
                "cargo",
                "rust-std",
                "rust-mingw",
                "rust-docs",
                "rustfmt-preview",
                "clippy-preview",
            ],
        );
        profile(
            "complete",
            &[
                "rustc",
                "cargo",
                "rust-std",
                "rust-mingw",
                "rust-docs",
                "rustfmt-preview",
                "clippy-preview",
                "rls-preview",
                "rust-analyzer-preview",
                "rust-src",
                "llvm-tools-preview",
                "rust-analysis",
                "miri-preview",
            ],
        );

        // The compiler libraries are not stable for end users, and they're also huge, so we only
        // `rustc-dev` for nightly users, and only in the "complete" profile. It's still possible
        // for users to install the additional component manually, if needed.
        if self.versions.channel() == "nightly" {
            self.extend_profile("complete", &mut manifest.profiles, &["rustc-dev"]);
            // Do not include the rustc-docs component for now, as it causes
            // conflicts with the rust-docs component when installed. See
            // #75833.
            // self.extend_profile("complete", &mut manifest.profiles, &["rustc-docs"]);
        }
    }

    fn add_renames_to(&self, manifest: &mut Manifest) {
        let mut rename = |from: &str, to: &str| {
            manifest.renames.insert(from.to_owned(), Rename { to: to.to_owned() })
        };
        rename("rls", "rls-preview");
        rename("rustfmt", "rustfmt-preview");
        rename("clippy", "clippy-preview");
        rename("miri", "miri-preview");
    }

    fn rust_package(&mut self, manifest: &Manifest) -> Package {
        let version_info = self.versions.version(&PkgType::Rust).expect("missing Rust tarball");
        let mut pkg = Package {
            version: version_info.version.expect("missing Rust version"),
            git_commit_hash: version_info.git_commit,
            target: BTreeMap::new(),
        };
        for host in HOSTS {
            if let Some(target) = self.target_host_combination(host, &manifest) {
                pkg.target.insert(host.to_string(), target);
            } else {
                pkg.target.insert(host.to_string(), Target::unavailable());
                continue;
            }
        }
        pkg
    }

    fn target_host_combination(&mut self, host: &str, manifest: &Manifest) -> Option<Target> {
        let filename = self.versions.tarball_name(&PkgType::Rust, host).unwrap();

        let mut target = Target::from_compressed_tar(self, &filename);
        if !target.available {
            return None;
        }

        let mut components = Vec::new();
        let mut extensions = Vec::new();

        let host_component = |pkg| Component::from_str(pkg, host);

        // rustc/rust-std/cargo/docs are all required,
        // and so is rust-mingw if it's available for the target.
        components.extend(vec![
            host_component("rustc"),
            host_component("rust-std"),
            host_component("cargo"),
            host_component("rust-docs"),
        ]);
        if host.contains("pc-windows-gnu") {
            components.push(host_component("rust-mingw"));
        }

        // Tools are always present in the manifest,
        // but might be marked as unavailable if they weren't built.
        extensions.extend(vec![
            host_component("clippy-preview"),
            host_component("miri-preview"),
            host_component("rls-preview"),
            host_component("rust-analyzer-preview"),
            host_component("rustfmt-preview"),
            host_component("llvm-tools-preview"),
            host_component("rust-analysis"),
        ]);

        extensions.extend(
            TARGETS
                .iter()
                .filter(|&&target| target != host)
                .map(|target| Component::from_str("rust-std", target)),
        );
        extensions.extend(HOSTS.iter().map(|target| Component::from_str("rustc-dev", target)));
        extensions.extend(HOSTS.iter().map(|target| Component::from_str("rustc-docs", target)));
        extensions.push(Component::from_str("rust-src", "*"));

        // If the components/extensions don't actually exist for this
        // particular host/target combination then nix it entirely from our
        // lists.
        let has_component = |c: &Component| {
            if c.target == "*" {
                return true;
            }
            let pkg = match manifest.pkg.get(&c.pkg) {
                Some(p) => p,
                None => return false,
            };
            pkg.target.get(&c.target).is_some()
        };
        extensions.retain(&has_component);
        components.retain(&has_component);

        target.components = Some(components);
        target.extensions = Some(extensions);
        Some(target)
    }

    fn profile(
        &mut self,
        profile_name: &str,
        dst: &mut BTreeMap<String, Vec<String>>,
        pkgs: &[&str],
    ) {
        dst.insert(profile_name.to_owned(), pkgs.iter().map(|s| (*s).to_owned()).collect());
    }

    fn extend_profile(
        &mut self,
        profile_name: &str,
        dst: &mut BTreeMap<String, Vec<String>>,
        pkgs: &[&str],
    ) {
        dst.get_mut(profile_name)
            .expect("existing profile")
            .extend(pkgs.iter().map(|s| (*s).to_owned()));
    }

    fn package(
        &mut self,
        pkgname: &str,
        dst: &mut BTreeMap<String, Package>,
        targets: &[&str],
        fallback: &[(&str, &str)],
    ) {
        let version_info = self
            .versions
            .version(&PkgType::from_component(pkgname))
            .expect("failed to load package version");
        let mut is_present = version_info.present;

        // Never ship nightly-only components for other trains.
        if self.versions.channel() != "nightly" && NIGHTLY_ONLY_COMPONENTS.contains(&pkgname) {
            is_present = false; // Pretend the component is entirely missing.
        }

        macro_rules! tarball_name {
            ($target_name:expr) => {
                self.versions.tarball_name(&PkgType::from_component(pkgname), $target_name).unwrap()
            };
        }
        let mut target_from_compressed_tar = |target_name| {
            let target = Target::from_compressed_tar(self, &tarball_name!(target_name));
            if target.available {
                return target;
            }
            for (substr, fallback_target) in fallback {
                if target_name.contains(substr) {
                    let t = Target::from_compressed_tar(self, &tarball_name!(fallback_target));
                    // Fallbacks must always be available.
                    assert!(t.available);
                    return t;
                }
            }
            Target::unavailable()
        };

        let targets = targets
            .iter()
            .map(|name| {
                let target = if is_present {
                    target_from_compressed_tar(name)
                } else {
                    // If the component is not present for this build add it anyway but mark it as
                    // unavailable -- this way rustup won't allow upgrades without --force
                    Target::unavailable()
                };
                (name.to_string(), target)
            })
            .collect();

        dst.insert(
            pkgname.to_string(),
            Package {
                version: version_info.version.unwrap_or_default(),
                git_commit_hash: version_info.git_commit,
                target: targets,
            },
        );
    }

    fn url(&self, path: &Path) -> String {
        let file_name = path.file_name().unwrap().to_str().unwrap();
        format!("{}/{}/{}", self.s3_address, self.date, file_name)
    }

    fn write_channel_files(&mut self, channel_name: &str, manifest: &Manifest) {
        self.write(&toml::to_string(&manifest).unwrap(), channel_name, ".toml");
        self.write(&manifest.date, channel_name, "-date.txt");
        self.write(
            manifest.pkg["rust"].git_commit_hash.as_ref().unwrap(),
            channel_name,
            "-git-commit-hash.txt",
        );
    }

    fn write(&mut self, contents: &str, channel_name: &str, suffix: &str) {
        let name = format!("channel-rust-{}{}", channel_name, suffix);
        self.shipped_files.insert(name.clone());

        let dst = self.output.join(name);
        t!(fs::write(&dst, contents));
    }

    fn write_shipped_files(&self, path: &Path) {
        let mut files = self.shipped_files.iter().map(|s| s.as_str()).collect::<Vec<_>>();
        files.sort();
        let content = format!("{}\n", files.join("\n"));

        t!(std::fs::write(path, content.as_bytes()));
    }
}
