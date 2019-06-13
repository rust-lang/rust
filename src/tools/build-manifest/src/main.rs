#![deny(rust_2018_idioms)]

use toml;
use serde::Serialize;

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::{self, Read, Write};
use std::path::{PathBuf, Path};
use std::process::{Command, Stdio};

static HOSTS: &[&str] = &[
    "aarch64-unknown-linux-gnu",
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
    "s390x-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-gnu",
    "x86_64-pc-windows-msvc",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "x86_64-unknown-netbsd",
];

static TARGETS: &[&str] = &[
    "aarch64-apple-ios",
    "aarch64-fuchsia",
    "aarch64-linux-android",
    "aarch64-pc-windows-msvc",
    "aarch64-unknown-cloudabi",
    "aarch64-unknown-linux-gnu",
    "aarch64-unknown-linux-musl",
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
    "armv7-unknown-linux-gnueabihf",
    "thumbv7neon-unknown-linux-gnueabihf",
    "armv7-unknown-linux-musleabihf",
    "armebv7r-none-eabi",
    "armebv7r-none-eabihf",
    "armv7r-none-eabi",
    "armv7r-none-eabihf",
    "armv7s-apple-ios",
    "asmjs-unknown-emscripten",
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
    "mips-unknown-linux-gnu",
    "mips-unknown-linux-musl",
    "mips64-unknown-linux-gnuabi64",
    "mips64el-unknown-linux-gnuabi64",
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
    "riscv32imc-unknown-none-elf",
    "riscv32imac-unknown-none-elf",
    "riscv64imac-unknown-none-elf",
    "riscv64gc-unknown-none-elf",
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
    "x86_64-rumprun-netbsd",
    "x86_64-sun-solaris",
    "x86_64-pc-solaris",
    "x86_64-unknown-cloudabi",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-linux-gnu",
    "x86_64-unknown-linux-gnux32",
    "x86_64-unknown-linux-musl",
    "x86_64-unknown-netbsd",
    "x86_64-unknown-redox",
];

static DOCS_TARGETS: &[&str] = &[
    "i686-apple-darwin",
    "i686-pc-windows-gnu",
    "i686-pc-windows-msvc",
    "i686-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-gnu",
    "x86_64-pc-windows-msvc",
    "x86_64-unknown-linux-gnu",
];

static MINGW: &[&str] = &[
    "i686-pc-windows-gnu",
    "x86_64-pc-windows-gnu",
];

#[derive(Serialize)]
#[serde(rename_all = "kebab-case")]
struct Manifest {
    manifest_version: String,
    date: String,
    pkg: BTreeMap<String, Package>,
    renames: BTreeMap<String, Rename>,
    profiles: BTreeMap<String, Vec<String>>,
}

#[derive(Serialize)]
struct Package {
    version: String,
    git_commit_hash: Option<String>,
    target: BTreeMap<String, Target>,
}

#[derive(Serialize)]
struct Rename {
    to: String,
}

#[derive(Serialize, Default)]
struct Target {
    available: bool,
    url: Option<String>,
    hash: Option<String>,
    xz_url: Option<String>,
    xz_hash: Option<String>,
    components: Option<Vec<Component>>,
    extensions: Option<Vec<Component>>,
}

impl Target {
    fn unavailable() -> Self { Self::default() }
}

#[derive(Serialize)]
struct Component {
    pkg: String,
    target: String,
}

impl Component {
    fn from_str(pkg: &str, target: &str) -> Self {
        Self { pkg: pkg.to_string(), target: target.to_string() }
    }
}

macro_rules! t {
    ($e:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed with {}", stringify!($e), e),
    })
}

struct Builder {
    rust_release: String,
    cargo_release: String,
    rls_release: String,
    clippy_release: String,
    rustfmt_release: String,
    llvm_tools_release: String,
    lldb_release: String,
    miri_release: String,

    input: PathBuf,
    output: PathBuf,
    gpg_passphrase: String,
    digests: BTreeMap<String, String>,
    s3_address: String,
    date: String,

    rust_version: Option<String>,
    cargo_version: Option<String>,
    rls_version: Option<String>,
    clippy_version: Option<String>,
    rustfmt_version: Option<String>,
    llvm_tools_version: Option<String>,
    lldb_version: Option<String>,
    miri_version: Option<String>,

    rust_git_commit_hash: Option<String>,
    cargo_git_commit_hash: Option<String>,
    rls_git_commit_hash: Option<String>,
    clippy_git_commit_hash: Option<String>,
    rustfmt_git_commit_hash: Option<String>,
    llvm_tools_git_commit_hash: Option<String>,
    lldb_git_commit_hash: Option<String>,
    miri_git_commit_hash: Option<String>,

    should_sign: bool,
}

fn main() {
    // Avoid signing packages while manually testing
    // Do NOT set this envvar in CI
    let should_sign = env::var("BUILD_MANIFEST_DISABLE_SIGNING").is_err();

    // Safety check to ensure signing is always enabled on CI
    // The CI environment variable is set by both Travis and AppVeyor
    if !should_sign && env::var("CI").is_ok() {
        println!("The 'BUILD_MANIFEST_DISABLE_SIGNING' env var can't be enabled on CI.");
        println!("If you're not running this on CI, unset the 'CI' env var.");
        panic!();
    }

    let mut args = env::args().skip(1);
    let input = PathBuf::from(args.next().unwrap());
    let output = PathBuf::from(args.next().unwrap());
    let date = args.next().unwrap();
    let rust_release = args.next().unwrap();
    let s3_address = args.next().unwrap();
    let cargo_release = args.next().unwrap();
    let rls_release = args.next().unwrap();
    let clippy_release = args.next().unwrap();
    let miri_release = args.next().unwrap();
    let rustfmt_release = args.next().unwrap();
    let llvm_tools_release = args.next().unwrap();
    let lldb_release = args.next().unwrap();

    // Do not ask for a passphrase while manually testing
    let mut passphrase = String::new();
    if should_sign {
        t!(io::stdin().read_to_string(&mut passphrase));
    }

    Builder {
        rust_release,
        cargo_release,
        rls_release,
        clippy_release,
        rustfmt_release,
        llvm_tools_release,
        lldb_release,
        miri_release,

        input,
        output,
        gpg_passphrase: passphrase,
        digests: BTreeMap::new(),
        s3_address,
        date,

        rust_version: None,
        cargo_version: None,
        rls_version: None,
        clippy_version: None,
        rustfmt_version: None,
        llvm_tools_version: None,
        lldb_version: None,
        miri_version: None,

        rust_git_commit_hash: None,
        cargo_git_commit_hash: None,
        rls_git_commit_hash: None,
        clippy_git_commit_hash: None,
        rustfmt_git_commit_hash: None,
        llvm_tools_git_commit_hash: None,
        lldb_git_commit_hash: None,
        miri_git_commit_hash: None,

        should_sign,
    }.build();
}

enum PkgType { RustSrc, Cargo, Rls, Clippy, Rustfmt, LlvmTools, Lldb, Miri, Other }

impl PkgType {
    fn from_component(component: &str) -> Self {
        use PkgType::*;
        match component {
            "rust-src" => RustSrc,
            "cargo" => Cargo,
            "rls" | "rls-preview" => Rls,
            "clippy" | "clippy-preview" => Clippy,
            "rustfmt" | "rustfmt-preview" => Rustfmt,
            "llvm-tools" | "llvm-tools-preview" => LlvmTools,
            "lldb" | "lldb-preview" => Lldb,
            "miri" | "miri-preview" => Miri,
            _ => Other,
        }
    }
}

impl Builder {
    fn build(&mut self) {
        self.rust_version = self.version("rust", "x86_64-unknown-linux-gnu");
        self.cargo_version = self.version("cargo", "x86_64-unknown-linux-gnu");
        self.rls_version = self.version("rls", "x86_64-unknown-linux-gnu");
        self.clippy_version = self.version("clippy", "x86_64-unknown-linux-gnu");
        self.rustfmt_version = self.version("rustfmt", "x86_64-unknown-linux-gnu");
        self.llvm_tools_version = self.version("llvm-tools", "x86_64-unknown-linux-gnu");
        // lldb is only built for macOS.
        self.lldb_version = self.version("lldb", "x86_64-apple-darwin");
        self.miri_version = self.version("miri", "x86_64-unknown-linux-gnu");

        self.rust_git_commit_hash = self.git_commit_hash("rust", "x86_64-unknown-linux-gnu");
        self.cargo_git_commit_hash = self.git_commit_hash("cargo", "x86_64-unknown-linux-gnu");
        self.rls_git_commit_hash = self.git_commit_hash("rls", "x86_64-unknown-linux-gnu");
        self.clippy_git_commit_hash = self.git_commit_hash("clippy", "x86_64-unknown-linux-gnu");
        self.rustfmt_git_commit_hash = self.git_commit_hash("rustfmt", "x86_64-unknown-linux-gnu");
        self.llvm_tools_git_commit_hash = self.git_commit_hash("llvm-tools",
                                                               "x86_64-unknown-linux-gnu");
        self.lldb_git_commit_hash = self.git_commit_hash("lldb", "x86_64-unknown-linux-gnu");
        self.miri_git_commit_hash = self.git_commit_hash("miri", "x86_64-unknown-linux-gnu");

        self.digest_and_sign();
        let manifest = self.build_manifest();
        self.write_channel_files(&self.rust_release, &manifest);

        if self.rust_release != "beta" && self.rust_release != "nightly" {
            self.write_channel_files("stable", &manifest);
        }
    }

    fn digest_and_sign(&mut self) {
        for file in t!(self.input.read_dir()).map(|e| t!(e).path()) {
            let filename = file.file_name().unwrap().to_str().unwrap();
            let digest = self.hash(&file);
            self.sign(&file);
            assert!(self.digests.insert(filename.to_string(), digest).is_none());
        }
    }

    fn build_manifest(&mut self) -> Manifest {
        let mut manifest = Manifest {
            manifest_version: "2".to_string(),
            date: self.date.to_string(),
            pkg: BTreeMap::new(),
            renames: BTreeMap::new(),
            profiles: BTreeMap::new(),
        };
        self.add_packages_to(&mut manifest);
        self.add_profiles_to(&mut manifest);
        self.add_renames_to(&mut manifest);
        manifest.pkg.insert("rust".to_string(), self.rust_package(&manifest));
        manifest
    }

    fn add_packages_to(&mut self, manifest: &mut Manifest) {
        let mut package = |name, targets| self.package(name, &mut manifest.pkg, targets);
        package("rustc", HOSTS);
        package("cargo", HOSTS);
        package("rust-mingw", MINGW);
        package("rust-std", TARGETS);
        package("rust-docs", DOCS_TARGETS);
        package("rust-src", &["*"]);
        package("rls-preview", HOSTS);
        package("clippy-preview", HOSTS);
        package("miri-preview", HOSTS);
        package("rustfmt-preview", HOSTS);
        package("rust-analysis", TARGETS);
        package("llvm-tools-preview", TARGETS);
        package("lldb-preview", TARGETS);
    }

    fn add_profiles_to(&mut self, manifest: &mut Manifest) {
        let mut profile = |name, pkgs| self.profile(name, &mut manifest.profiles, pkgs);
        profile("minimal", &["rustc", "cargo", "rust-std", "rust-mingw"]);
        profile("default", &[
            "rustc", "cargo", "rust-std", "rust-mingw",
            "rust-docs", "rustfmt-preview", "clippy-preview"
        ]);
        profile("complete", &[
            "rustc", "cargo", "rust-std", "rust-mingw",
            "rust-docs", "rustfmt-preview", "clippy-preview",
            "rls-preview", "rust-src", "llvm-tools-preview",
            "lldb-preview", "rust-analysis", "miri-preview"
        ]);
    }

    fn add_renames_to(&self, manifest: &mut Manifest) {
        let mut rename = |from: &str, to: &str| manifest.renames.insert(
            from.to_owned(),
            Rename { to: to.to_owned() }
        );
        rename("rls", "rls-preview");
        rename("rustfmt", "rustfmt-preview");
        rename("clippy", "clippy-preview");
        rename("miri", "miri-preview");
    }

    fn rust_package(&mut self, manifest: &Manifest) -> Package {
        let mut pkg = Package {
            version: self.cached_version("rust")
                         .as_ref()
                         .expect("Couldn't find Rust version")
                         .clone(),
            git_commit_hash: self.cached_git_commit_hash("rust").clone(),
            target: BTreeMap::new(),
        };
        for host in HOSTS {
            if let Some(target) = self.target_host_combination(host, &manifest) {
                pkg.target.insert(host.to_string(), target);
            } else {
                pkg.target.insert(host.to_string(), Target::unavailable());
                continue
            }
        }
        pkg
    }

    fn target_host_combination(&mut self, host: &str, manifest: &Manifest) -> Option<Target> {
        let filename = self.filename("rust", host);
        let digest = self.digests.remove(&filename)?;
        let xz_filename = filename.replace(".tar.gz", ".tar.xz");
        let xz_digest = self.digests.remove(&xz_filename);
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
            host_component("rustfmt-preview"),
            host_component("llvm-tools-preview"),
            host_component("lldb-preview"),
            host_component("rust-analysis"),
        ]);

        extensions.extend(
            TARGETS.iter()
                .filter(|&&target| target != host)
                .map(|target| Component::from_str("rust-std", target))
        );
        extensions.push(Component::from_str("rust-src", "*"));

        // If the components/extensions don't actually exist for this
        // particular host/target combination then nix it entirely from our
        // lists.
        let has_component = |c: &Component| {
            if c.target == "*" {
                return true
            }
            let pkg = match manifest.pkg.get(&c.pkg) {
                Some(p) => p,
                None => return false,
            };
            pkg.target.get(&c.target).is_some()
        };
        extensions.retain(&has_component);
        components.retain(&has_component);

        Some(Target {
            available: true,
            url: Some(self.url(&filename)),
            hash: Some(digest),
            xz_url: xz_digest.as_ref().map(|_| self.url(&xz_filename)),
            xz_hash: xz_digest,
            components: Some(components),
            extensions: Some(extensions),
        })
    }

    fn profile(&mut self,
               profile_name: &str,
               dst: &mut BTreeMap<String, Vec<String>>,
               pkgs: &[&str]) {
        dst.insert(profile_name.to_owned(), pkgs.iter().map(|s| (*s).to_owned()).collect());
    }

    fn package(&mut self,
               pkgname: &str,
               dst: &mut BTreeMap<String, Package>,
               targets: &[&str]) {
        let (version, mut is_present) = self.cached_version(pkgname)
            .as_ref()
            .cloned()
            .map(|version| (version, true))
            .unwrap_or_default();

        // miri needs to build std with xargo, which doesn't allow stable/beta:
        // <https://github.com/japaric/xargo/pull/204#issuecomment-374888868>
        if pkgname == "miri-preview" && self.rust_release != "nightly" {
            is_present = false; // ignore it
        }

        let targets = targets.iter().map(|name| {
            if is_present {
                let filename = self.filename(pkgname, name);
                let digest = match self.digests.remove(&filename) {
                    Some(digest) => digest,
                    None => return (name.to_string(), Target::unavailable()),
                };
                let xz_filename = filename.replace(".tar.gz", ".tar.xz");
                let xz_digest = self.digests.remove(&xz_filename);

                (name.to_string(), Target {
                    available: true,
                    url: Some(self.url(&filename)),
                    hash: Some(digest),
                    xz_url: xz_digest.as_ref().map(|_| self.url(&xz_filename)),
                    xz_hash: xz_digest,
                    components: None,
                    extensions: None,
                })
            } else {
                // If the component is not present for this build add it anyway but mark it as
                // unavailable -- this way rustup won't allow upgrades without --force
                (name.to_string(), Target::unavailable())
            }
        }).collect();

        dst.insert(pkgname.to_string(), Package {
            version,
            git_commit_hash: self.cached_git_commit_hash(pkgname).clone(),
            target: targets,
        });
    }

    fn url(&self, filename: &str) -> String {
        format!("{}/{}/{}",
                self.s3_address,
                self.date,
                filename)
    }

    fn filename(&self, component: &str, target: &str) -> String {
        use PkgType::*;
        match PkgType::from_component(component) {
            RustSrc => format!("rust-src-{}.tar.gz", self.rust_release),
            Cargo => format!("cargo-{}-{}.tar.gz", self.cargo_release, target),
            Rls => format!("rls-{}-{}.tar.gz", self.rls_release, target),
            Clippy => format!("clippy-{}-{}.tar.gz", self.clippy_release, target),
            Rustfmt => format!("rustfmt-{}-{}.tar.gz", self.rustfmt_release, target),
            LlvmTools => format!("llvm-tools-{}-{}.tar.gz", self.llvm_tools_release, target),
            Lldb => format!("lldb-{}-{}.tar.gz", self.lldb_release, target),
            Miri => format!("miri-{}-{}.tar.gz", self.miri_release, target),
            Other => format!("{}-{}-{}.tar.gz", component, self.rust_release, target),
        }
    }

    fn cached_version(&self, component: &str) -> &Option<String> {
        use PkgType::*;
        match PkgType::from_component(component) {
            Cargo => &self.cargo_version,
            Rls => &self.rls_version,
            Clippy => &self.clippy_version,
            Rustfmt => &self.rustfmt_version,
            LlvmTools => &self.llvm_tools_version,
            Lldb => &self.lldb_version,
            Miri => &self.miri_version,
            _ => &self.rust_version,
        }
    }

    fn cached_git_commit_hash(&self, component: &str) -> &Option<String> {
        use PkgType::*;
        match PkgType::from_component(component) {
            Cargo => &self.cargo_git_commit_hash,
            Rls => &self.rls_git_commit_hash,
            Clippy => &self.clippy_git_commit_hash,
            Rustfmt => &self.rustfmt_git_commit_hash,
            LlvmTools => &self.llvm_tools_git_commit_hash,
            Lldb => &self.lldb_git_commit_hash,
            Miri => &self.miri_git_commit_hash,
            _ => &self.rust_git_commit_hash,
        }
    }

    fn version(&self, component: &str, target: &str) -> Option<String> {
        self.untar(component, target, |filename| format!("{}/version", filename))
    }

    fn git_commit_hash(&self, component: &str, target: &str) -> Option<String> {
        self.untar(component, target, |filename| format!("{}/git-commit-hash", filename))
    }

    fn untar<F>(&self, component: &str, target: &str, dir: F) -> Option<String>
    where
        F: FnOnce(String) -> String
    {
        let mut cmd = Command::new("tar");
        let filename = self.filename(component, target);
        cmd.arg("xf")
           .arg(self.input.join(&filename))
           .arg(dir(filename.replace(".tar.gz", "")))
           .arg("-O");
        let output = t!(cmd.output());
        if output.status.success() {
            Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
        } else {
            None
        }
    }

    fn hash(&self, path: &Path) -> String {
        let sha = t!(Command::new("shasum")
                        .arg("-a").arg("256")
                        .arg(path.file_name().unwrap())
                        .current_dir(path.parent().unwrap())
                        .output());
        assert!(sha.status.success());

        let filename = path.file_name().unwrap().to_str().unwrap();
        let sha256 = self.output.join(format!("{}.sha256", filename));
        t!(fs::write(&sha256, &sha.stdout));

        let stdout = String::from_utf8_lossy(&sha.stdout);
        stdout.split_whitespace().next().unwrap().to_string()
    }

    fn sign(&self, path: &Path) {
        if !self.should_sign {
            return;
        }

        let filename = path.file_name().unwrap().to_str().unwrap();
        let asc = self.output.join(format!("{}.asc", filename));
        println!("signing: {:?}", path);
        let mut cmd = Command::new("gpg");
        cmd.arg("--pinentry-mode=loopback")
            .arg("--no-tty")
            .arg("--yes")
            .arg("--batch")
            .arg("--passphrase-fd").arg("0")
            .arg("--personal-digest-preferences").arg("SHA512")
            .arg("--armor")
            .arg("--output").arg(&asc)
            .arg("--detach-sign").arg(path)
            .stdin(Stdio::piped());
        let mut child = t!(cmd.spawn());
        t!(child.stdin.take().unwrap().write_all(self.gpg_passphrase.as_bytes()));
        assert!(t!(child.wait()).success());
    }

    fn write_channel_files(&self, channel_name: &str, manifest: &Manifest) {
        self.write(&toml::to_string(&manifest).unwrap(), channel_name, ".toml");
        self.write(&manifest.date, channel_name, "-date.txt");
        self.write(manifest.pkg["rust"].git_commit_hash.as_ref().unwrap(),
                   channel_name, "-git-commit-hash.txt");
    }

    fn write(&self, contents: &str, channel_name: &str, suffix: &str) {
        let dst = self.output.join(format!("channel-rust-{}{}", channel_name, suffix));
        t!(fs::write(&dst, contents));
        self.hash(&dst);
        self.sign(&dst);
    }
}
