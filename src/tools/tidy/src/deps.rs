// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Check license of third-party deps by inspecting src/vendor

use std::collections::{BTreeSet, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::Command;

use serde_json;

static LICENSES: &'static [&'static str] = &[
    "MIT/Apache-2.0",
    "MIT / Apache-2.0",
    "Apache-2.0/MIT",
    "Apache-2.0 / MIT",
    "MIT OR Apache-2.0",
    "MIT",
    "Unlicense/MIT",
];

/// These are exceptions to Rust's permissive licensing policy, and
/// should be considered bugs. Exceptions are only allowed in Rust
/// tooling. It is _crucial_ that no exception crates be dependencies
/// of the Rust runtime (std / test).
static EXCEPTIONS: &'static [&'static str] = &[
    "mdbook",             // MPL2, mdbook
    "openssl",            // BSD+advertising clause, cargo, mdbook
    "pest",               // MPL2, mdbook via handlebars
    "thread-id",          // Apache-2.0, mdbook
    "toml-query",         // MPL-2.0, mdbook
    "is-match",           // MPL-2.0, mdbook
    "cssparser",          // MPL-2.0, rustdoc
    "smallvec",           // MPL-2.0, rustdoc
    "fuchsia-zircon-sys", // BSD-3-Clause, rustdoc, rustc, cargo
    "fuchsia-zircon",     // BSD-3-Clause, rustdoc, rustc, cargo (jobserver & tempdir)
    "cssparser-macros",   // MPL-2.0, rustdoc
    "selectors",          // MPL-2.0, rustdoc
    "clippy_lints",       // MPL-2.0, rls
    "colored",            // MPL-2.0, rustfmt
];

/// Which crates to check against the whitelist?
static WHITELIST_CRATES: &'static [CrateVersion] = &[
    CrateVersion("rustc", "0.0.0"),
    CrateVersion("rustc_codegen_llvm", "0.0.0"),
];

/// Whitelist of crates rustc is allowed to depend on. Avoid adding to the list if possible.
static WHITELIST: &'static [Crate] = &[
    Crate("aho-corasick"),
    Crate("ar"),
    Crate("arrayvec"),
    Crate("atty"),
    Crate("backtrace"),
    Crate("backtrace-sys"),
    Crate("bitflags"),
    Crate("byteorder"),
    Crate("cc"),
    Crate("chalk-engine"),
    Crate("chalk-macros"),
    Crate("cfg-if"),
    Crate("cmake"),
    Crate("crossbeam-deque"),
    Crate("crossbeam-epoch"),
    Crate("crossbeam-utils"),
    Crate("datafrog"),
    Crate("either"),
    Crate("ena"),
    Crate("env_logger"),
    Crate("filetime"),
    Crate("flate2"),
    Crate("fuchsia-zircon"),
    Crate("fuchsia-zircon-sys"),
    Crate("getopts"),
    Crate("humantime"),
    Crate("jobserver"),
    Crate("kernel32-sys"),
    Crate("lazy_static"),
    Crate("libc"),
    Crate("log"),
    Crate("log_settings"),
    Crate("memchr"),
    Crate("memoffset"),
    Crate("miniz-sys"),
    Crate("nodrop"),
    Crate("num_cpus"),
    Crate("owning_ref"),
    Crate("parking_lot"),
    Crate("parking_lot_core"),
    Crate("polonius-engine"),
    Crate("pkg-config"),
    Crate("quick-error"),
    Crate("rand"),
    Crate("redox_syscall"),
    Crate("redox_termios"),
    Crate("regex"),
    Crate("regex-syntax"),
    Crate("remove_dir_all"),
    Crate("rustc-demangle"),
    Crate("rustc-hash"),
    Crate("rustc-rayon"),
    Crate("rustc-rayon-core"),
    Crate("scoped-tls"),
    Crate("scopeguard"),
    Crate("smallvec"),
    Crate("stable_deref_trait"),
    Crate("tempfile"),
    Crate("termcolor"),
    Crate("terminon"),
    Crate("termion"),
    Crate("thread_local"),
    Crate("ucd-util"),
    Crate("unicode-width"),
    Crate("unreachable"),
    Crate("utf8-ranges"),
    Crate("void"),
    Crate("winapi"),
    Crate("winapi-build"),
    Crate("winapi-i686-pc-windows-gnu"),
    Crate("winapi-x86_64-pc-windows-gnu"),
    Crate("wincolor"),
];

// Some types for Serde to deserialize the output of `cargo metadata` to...

#[derive(Deserialize)]
struct Output {
    resolve: Resolve,
}

#[derive(Deserialize)]
struct Resolve {
    nodes: Vec<ResolveNode>,
}

#[derive(Deserialize)]
struct ResolveNode {
    id: String,
    dependencies: Vec<String>,
}

/// A unique identifier for a crate
#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Debug, Hash)]
struct Crate<'a>(&'a str); // (name,)

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Debug, Hash)]
struct CrateVersion<'a>(&'a str, &'a str); // (name, version)

impl<'a> Crate<'a> {
    pub fn id_str(&self) -> String {
        format!("{} ", self.0)
    }
}

impl<'a> CrateVersion<'a> {
    /// Returns the struct and whether or not the dep is in-tree
    pub fn from_str(s: &'a str) -> (Self, bool) {
        let mut parts = s.split(" ");
        let name = parts.next().unwrap();
        let version = parts.next().unwrap();
        let path = parts.next().unwrap();

        let is_path_dep = path.starts_with("(path+");

        (CrateVersion(name, version), is_path_dep)
    }

    pub fn id_str(&self) -> String {
        format!("{} {}", self.0, self.1)
    }
}

impl<'a> From<CrateVersion<'a>> for Crate<'a> {
    fn from(cv: CrateVersion<'a>) -> Crate<'a> {
        Crate(cv.0)
    }
}

/// Checks the dependency at the given path. Changes `bad` to `true` if a check failed.
///
/// Specifically, this checks that the license is correct.
pub fn check(path: &Path, bad: &mut bool) {
    // Check licences
    let path = path.join("vendor");
    assert!(path.exists(), "vendor directory missing");
    let mut saw_dir = false;
    for dir in t!(path.read_dir()) {
        saw_dir = true;
        let dir = t!(dir);

        // skip our exceptions
        if EXCEPTIONS.iter().any(|exception| {
            dir.path()
                .to_str()
                .unwrap()
                .contains(&format!("src/vendor/{}", exception))
        }) {
            continue;
        }

        let toml = dir.path().join("Cargo.toml");
        *bad = *bad || !check_license(&toml);
    }
    assert!(saw_dir, "no vendored source");
}

/// Checks the dependency of WHITELIST_CRATES at the given path. Changes `bad` to `true` if a check
/// failed.
///
/// Specifically, this checks that the dependencies are on the WHITELIST.
pub fn check_whitelist(path: &Path, cargo: &Path, bad: &mut bool) {
    // Get dependencies from cargo metadata
    let resolve = get_deps(path, cargo);

    // Get the whitelist into a convenient form
    let whitelist: HashSet<_> = WHITELIST.iter().cloned().collect();

    // Check dependencies
    let mut visited = BTreeSet::new();
    let mut unapproved = BTreeSet::new();
    for &krate in WHITELIST_CRATES.iter() {
        let mut bad = check_crate_whitelist(&whitelist, &resolve, &mut visited, krate, false);
        unapproved.append(&mut bad);
    }

    if unapproved.len() > 0 {
        println!("Dependencies not on the whitelist:");
        for dep in unapproved {
            println!("* {}", dep.id_str());
        }
        *bad = true;
    }
}

fn check_license(path: &Path) -> bool {
    if !path.exists() {
        panic!("{} does not exist", path.display());
    }
    let mut contents = String::new();
    t!(t!(File::open(path)).read_to_string(&mut contents));

    let mut found_license = false;
    for line in contents.lines() {
        if !line.starts_with("license") {
            continue;
        }
        let license = extract_license(line);
        if !LICENSES.contains(&&*license) {
            println!("invalid license {} in {}", license, path.display());
            return false;
        }
        found_license = true;
        break;
    }
    if !found_license {
        println!("no license in {}", path.display());
        return false;
    }

    true
}

fn extract_license(line: &str) -> String {
    let first_quote = line.find('"');
    let last_quote = line.rfind('"');
    if let (Some(f), Some(l)) = (first_quote, last_quote) {
        let license = &line[f + 1..l];
        license.into()
    } else {
        "bad-license-parse".into()
    }
}

/// Get the dependencies of the crate at the given path using `cargo metadata`.
fn get_deps(path: &Path, cargo: &Path) -> Resolve {
    // Run `cargo metadata` to get the set of dependencies
    let output = Command::new(cargo)
        .arg("metadata")
        .arg("--format-version")
        .arg("1")
        .arg("--manifest-path")
        .arg(path.join("Cargo.toml"))
        .output()
        .expect("Unable to run `cargo metadata`")
        .stdout;
    let output = String::from_utf8_lossy(&output);
    let output: Output = serde_json::from_str(&output).unwrap();

    output.resolve
}

/// Checks the dependencies of the given crate from the given cargo metadata to see if they are on
/// the whitelist. Returns a list of illegal dependencies.
fn check_crate_whitelist<'a, 'b>(
    whitelist: &'a HashSet<Crate>,
    resolve: &'a Resolve,
    visited: &'b mut BTreeSet<CrateVersion<'a>>,
    krate: CrateVersion<'a>,
    must_be_on_whitelist: bool,
) -> BTreeSet<Crate<'a>> {
    // Will contain bad deps
    let mut unapproved = BTreeSet::new();

    // Check if we have already visited this crate
    if visited.contains(&krate) {
        return unapproved;
    }

    visited.insert(krate);

    // If this path is in-tree, we don't require it to be on the whitelist
    if must_be_on_whitelist {
        // If this dependency is not on the WHITELIST, add to bad set
        if !whitelist.contains(&krate.into()) {
            unapproved.insert(krate.into());
        }
    }

    // Do a DFS in the crate graph (it's a DAG, so we know we have no cycles!)
    let to_check = resolve
        .nodes
        .iter()
        .find(|n| n.id.starts_with(&krate.id_str()))
        .expect("crate does not exist");

    for dep in to_check.dependencies.iter() {
        let (krate, is_path_dep) = CrateVersion::from_str(dep);

        let mut bad = check_crate_whitelist(whitelist, resolve, visited, krate, !is_path_dep);
        unapproved.append(&mut bad);
    }

    unapproved
}
