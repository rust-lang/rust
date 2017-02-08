// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

extern crate toml;
extern crate rustc_serialize;

use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{PathBuf, Path};
use std::process::{Command, Stdio};

static HOSTS: &'static [&'static str] = &[
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
    "powerpc-unknown-linux-gnu",
    "powerpc64-unknown-linux-gnu",
    "powerpc64le-unknown-linux-gnu",
    "s390x-unknown-linux-gnu",
    "x86_64-apple-darwin",
    "x86_64-pc-windows-gnu",
    "x86_64-pc-windows-msvc",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-linux-gnu",
    "x86_64-unknown-netbsd",
];

static TARGETS: &'static [&'static str] = &[
    "aarch64-apple-ios",
    "aarch64-linux-android",
    "aarch64-unknown-linux-gnu",
    "arm-linux-androideabi",
    "arm-unknown-linux-gnueabi",
    "arm-unknown-linux-gnueabihf",
    "arm-unknown-linux-musleabi",
    "arm-unknown-linux-musleabihf",
    "armv7-apple-ios",
    "armv7-linux-androideabi",
    "armv7-unknown-linux-gnueabihf",
    "armv7-unknown-linux-musleabihf",
    "armv7s-apple-ios",
    "asmjs-unknown-emscripten",
    "i386-apple-ios",
    "i586-pc-windows-msvc",
    "i586-unknown-linux-gnu",
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
    "mipsel-unknown-linux-gnu",
    "mipsel-unknown-linux-musl",
    "powerpc-unknown-linux-gnu",
    "powerpc64-unknown-linux-gnu",
    "powerpc64le-unknown-linux-gnu",
    "s390x-unknown-linux-gnu",
    "wasm32-unknown-emscripten",
    "x86_64-apple-darwin",
    "x86_64-apple-ios",
    "x86_64-pc-windows-gnu",
    "x86_64-pc-windows-msvc",
    "x86_64-rumprun-netbsd",
    "x86_64-unknown-freebsd",
    "x86_64-unknown-linux-gnu",
    "x86_64-unknown-linux-musl",
    "x86_64-unknown-netbsd",
];

static MINGW: &'static [&'static str] = &[
    "i686-pc-windows-gnu",
    "x86_64-pc-windows-gnu",
];

#[derive(RustcEncodable)]
struct Manifest {
    manifest_version: String,
    date: String,
    pkg: HashMap<String, Package>,
}

#[derive(RustcEncodable)]
struct Package {
    version: String,
    target: HashMap<String, Target>,
}

#[derive(RustcEncodable)]
struct Target {
    available: bool,
    url: Option<String>,
    hash: Option<String>,
    components: Option<Vec<Component>>,
    extensions: Option<Vec<Component>>,
}

#[derive(RustcEncodable)]
struct Component {
    pkg: String,
    target: String,
}

macro_rules! t {
    ($e:expr) => (match $e {
        Ok(e) => e,
        Err(e) => panic!("{} failed with {}", stringify!($e), e),
    })
}

struct Builder {
    channel: String,
    input: PathBuf,
    output: PathBuf,
    gpg_passphrase: String,
    digests: HashMap<String, String>,
    s3_address: String,
    date: String,
    rust_version: String,
    cargo_version: String,
}

fn main() {
    let mut args = env::args().skip(1);
    let input = PathBuf::from(args.next().unwrap());
    let output = PathBuf::from(args.next().unwrap());
    let date = args.next().unwrap();
    let channel = args.next().unwrap();
    let s3_address = args.next().unwrap();
    let mut passphrase = String::new();
    t!(io::stdin().read_to_string(&mut passphrase));

    Builder {
        channel: channel,
        input: input,
        output: output,
        gpg_passphrase: passphrase,
        digests: HashMap::new(),
        s3_address: s3_address,
        date: date,
        rust_version: String::new(),
        cargo_version: String::new(),
    }.build();
}

impl Builder {
    fn build(&mut self) {
        self.rust_version = self.version("rust", "x86_64-unknown-linux-gnu");
        self.cargo_version = self.version("cargo", "x86_64-unknown-linux-gnu");

        self.digest_and_sign();
        let manifest = self.build_manifest();
        let manifest = toml::encode(&manifest).to_string();

        let filename = format!("channel-rust-{}.toml", self.channel);
        self.write_manifest(&manifest, &filename);

        if self.channel != "beta" && self.channel != "nightly" {
            self.write_manifest(&manifest, "channel-rust-stable.toml");
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
            pkg: HashMap::new(),
        };

        self.package("rustc", &mut manifest.pkg, HOSTS);
        self.package("cargo", &mut manifest.pkg, HOSTS);
        self.package("rust-mingw", &mut manifest.pkg, MINGW);
        self.package("rust-std", &mut manifest.pkg, TARGETS);
        self.package("rust-docs", &mut manifest.pkg, TARGETS);
        self.package("rust-src", &mut manifest.pkg, &["*"]);

        let mut pkg = Package {
            version: self.cached_version("rust").to_string(),
            target: HashMap::new(),
        };
        for host in HOSTS {
            let filename = self.filename("rust", host);
            let digest = match self.digests.remove(&filename) {
                Some(digest) => digest,
                None => {
                    pkg.target.insert(host.to_string(), Target {
                        available: false,
                        url: None,
                        hash: None,
                        components: None,
                        extensions: None,
                    });
                    continue
                }
            };
            let mut components = Vec::new();
            let mut extensions = Vec::new();

            // rustc/rust-std/cargo are all required, and so is rust-mingw if it's
            // available for the target.
            components.extend(vec![
                Component { pkg: "rustc".to_string(), target: host.to_string() },
                Component { pkg: "rust-std".to_string(), target: host.to_string() },
                Component { pkg: "cargo".to_string(), target: host.to_string() },
            ]);
            if host.contains("pc-windows-gnu") {
                components.push(Component {
                    pkg: "rust-mingw".to_string(),
                    target: host.to_string(),
                });
            }

            // Docs, other standard libraries, and the source package are all
            // optional.
            extensions.push(Component {
                pkg: "rust-docs".to_string(),
                target: host.to_string(),
            });
            for target in TARGETS {
                if target != host {
                    extensions.push(Component {
                        pkg: "rust-std".to_string(),
                        target: target.to_string(),
                    });
                }
            }
            extensions.push(Component {
                pkg: "rust-src".to_string(),
                target: "*".to_string(),
            });

            pkg.target.insert(host.to_string(), Target {
                available: true,
                url: Some(self.url("rust", host)),
                hash: Some(to_hex(digest.as_ref())),
                components: Some(components),
                extensions: Some(extensions),
            });
        }
        manifest.pkg.insert("rust".to_string(), pkg);

        return manifest
    }

    fn package(&mut self,
               pkgname: &str,
               dst: &mut HashMap<String, Package>,
               targets: &[&str]) {
        let targets = targets.iter().map(|name| {
            let filename = self.filename(pkgname, name);
            let digest = match self.digests.remove(&filename) {
                Some(digest) => digest,
                None => {
                    return (name.to_string(), Target {
                        available: false,
                        url: None,
                        hash: None,
                        components: None,
                        extensions: None,
                    })
                }
            };

            (name.to_string(), Target {
                available: true,
                url: Some(self.url(pkgname, name)),
                hash: Some(digest),
                components: None,
                extensions: None,
            })
        }).collect();

        dst.insert(pkgname.to_string(), Package {
            version: self.cached_version(pkgname).to_string(),
            target: targets,
        });
    }

    fn url(&self, component: &str, target: &str) -> String {
        format!("{}/{}/{}",
                self.s3_address,
                self.date,
                self.filename(component, target))
    }

    fn filename(&self, component: &str, target: &str) -> String {
        if component == "rust-src" {
            format!("rust-src-{}.tar.gz", self.channel)
        } else if component == "cargo" {
            format!("cargo-nightly-{}.tar.gz", target)
        } else {
            format!("{}-{}-{}.tar.gz", component, self.channel, target)
        }
    }

    fn cached_version(&self, component: &str) -> &str {
        if component == "cargo" {
            &self.cargo_version
        } else {
            &self.rust_version
        }
    }

    fn version(&self, component: &str, target: &str) -> String {
        let mut cmd = Command::new("tar");
        let filename = self.filename(component, target);
        cmd.arg("xf")
           .arg(self.input.join(&filename))
           .arg(format!("{}/version", filename.replace(".tar.gz", "")))
           .arg("-O");
        let version = t!(cmd.output());
        if !version.status.success() {
            panic!("failed to learn version:\n\n{:?}\n\n{}\n\n{}",
                   cmd,
                   String::from_utf8_lossy(&version.stdout),
                   String::from_utf8_lossy(&version.stderr));
        }
        String::from_utf8_lossy(&version.stdout).trim().to_string()
    }

    fn hash(&self, path: &Path) -> String {
        let sha = t!(Command::new("shasum")
                        .arg("-a").arg("256")
                        .arg(path)
                        .output());
        assert!(sha.status.success());

        let filename = path.file_name().unwrap().to_str().unwrap();
        let sha256 = self.output.join(format!("{}.sha256", filename));
        t!(t!(File::create(&sha256)).write_all(&sha.stdout));

        let stdout = String::from_utf8_lossy(&sha.stdout);
        stdout.split_whitespace().next().unwrap().to_string()
    }

    fn sign(&self, path: &Path) {
        let filename = path.file_name().unwrap().to_str().unwrap();
        let asc = self.output.join(format!("{}.asc", filename));
        println!("signing: {:?}", path);
        let mut cmd = Command::new("gpg");
        cmd.arg("--no-tty")
            .arg("--yes")
            .arg("--passphrase-fd").arg("0")
            .arg("--armor")
            .arg("--output").arg(&asc)
            .arg("--detach-sign").arg(path)
            .stdin(Stdio::piped());
        let mut child = t!(cmd.spawn());
        t!(child.stdin.take().unwrap().write_all(self.gpg_passphrase.as_bytes()));
        assert!(t!(child.wait()).success());
    }

    fn write_manifest(&self, manifest: &str, name: &str) {
        let dst = self.output.join(name);
        t!(t!(File::create(&dst)).write_all(manifest.as_bytes()));
        self.hash(&dst);
        self.sign(&dst);
    }
}

fn to_hex(digest: &[u8]) -> String {
    let mut ret = String::new();
    for byte in digest {
        ret.push(hex((byte & 0xf0) >> 4));
        ret.push(hex(byte & 0xf));
    }
    return ret;

    fn hex(b: u8) -> char {
        match b {
            0...9 => (b'0' + b) as char,
            _ => (b'a' + b - 10) as char,
        }
    }
}
