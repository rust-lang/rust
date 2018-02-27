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

use std::collections::HashSet;
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
    "clippy_lints",       // MPL-2.0 rls
];

/// Which crates to check against the whitelist?
static WHITELIST_CRATES: &'static [Crate] =
    &[Crate("rustc", "0.0.0"), Crate("rustc_trans", "0.0.0")];

/// Whitelist of crates rustc is allowed to depend on. Avoid adding to the list if possible.
static WHITELIST: &'static [Crate] = &[
//    Crate("ar", "0.3.1"),
//    Crate("arena", "0.0.0"),
//    Crate("backtrace", "0.3.5"),
//    Crate("backtrace-sys", "0.1.16"),
//    Crate("bitflags", "1.0.1"),
//    Crate("build_helper", "0.1.0"),
//    Crate("byteorder", "1.2.1"),
//    Crate("cc", "1.0.4"),
//    Crate("cfg-if", "0.1.2"),
//    Crate("cmake", "0.1.29"),
//    Crate("filetime", "0.1.15"),
//    Crate("flate2", "1.0.1"),
//    Crate("fmt_macros", "0.0.0"),
//    Crate("fuchsia-zircon", "0.3.3"),
//    Crate("fuchsia-zircon-sys", "0.3.3"),
//    Crate("graphviz", "0.0.0"),
//    Crate("jobserver", "0.1.9"),
//    Crate("kernel32-sys", "0.2.2"),
//    Crate("lazy_static", "0.2.11"),
//    Crate("libc", "0.2.36"),
//    Crate("log", "0.4.1"),
//    Crate("log_settings", "0.1.1"),
//    Crate("miniz-sys", "0.1.10"),
//    Crate("num_cpus", "1.8.0"),
//    Crate("owning_ref", "0.3.3"),
//    Crate("parking_lot", "0.5.3"),
//    Crate("parking_lot_core", "0.2.9"),
//    Crate("rand", "0.3.20"),
//    Crate("redox_syscall", "0.1.37"),
//    Crate("rustc", "0.0.0"),
//    Crate("rustc-demangle", "0.1.5"),
//    Crate("rustc_allocator", "0.0.0"),
//    Crate("rustc_apfloat", "0.0.0"),
//    Crate("rustc_back", "0.0.0"),
//    Crate("rustc_binaryen", "0.0.0"),
//    Crate("rustc_const_eval", "0.0.0"),
//    Crate("rustc_const_math", "0.0.0"),
//    Crate("rustc_cratesio_shim", "0.0.0"),
//    Crate("rustc_data_structures", "0.0.0"),
//    Crate("rustc_errors", "0.0.0"),
//    Crate("rustc_incremental", "0.0.0"),
//    Crate("rustc_llvm", "0.0.0"),
//    Crate("rustc_mir", "0.0.0"),
//    Crate("rustc_platform_intrinsics", "0.0.0"),
//    Crate("rustc_trans", "0.0.0"),
//    Crate("rustc_trans_utils", "0.0.0"),
//    Crate("serialize", "0.0.0"),
//    Crate("smallvec", "0.6.0"),
//    Crate("stable_deref_trait", "1.0.0"),
//    Crate("syntax", "0.0.0"),
//    Crate("syntax_pos", "0.0.0"),
//    Crate("tempdir", "0.3.5"),
//    Crate("unicode-width", "0.1.4"),
//    Crate("winapi", "0.2.8"),
//    Crate("winapi", "0.3.4"),
//    Crate("winapi-build", "0.1.1"),
//    Crate("winapi-i686-pc-windows-gnu", "0.4.0"),
//    Crate("winapi-x86_64-pc-windows-gnu", "0.4.0"),
];

// Some types for Serde to deserialize the output of `cargo metadata` to...

#[derive(Deserialize)]
struct Output {
    resolve: Resolve,

    // Not used, but needed to not confuse serde :P
    #[allow(dead_code)] packages: Vec<Package>,
}

// Not used, but needed to not confuse serde :P
#[allow(dead_code)]
#[derive(Deserialize)]
struct Package {
    name: String,
    version: String,
    id: String,
    source: Option<String>,
    manifest_path: String,
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
struct Crate<'a>(&'a str, &'a str); // (name, version)

impl<'a> Crate<'a> {
    pub fn from_str(s: &'a str) -> Self {
        let mut parts = s.split(" ");
        let name = parts.next().unwrap();
        let version = parts.next().unwrap();

        Crate(name, version)
    }

    pub fn id_str(&self) -> String {
        format!("{} {}", self.0, self.1)
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
    let mut unapproved = Vec::new();
    for &krate in WHITELIST_CRATES.iter() {
        let mut bad = check_crate_whitelist(&whitelist, &resolve, krate);
        unapproved.append(&mut bad);
    }

    // For ease of reading
    unapproved.sort_unstable();
    unapproved.dedup();

    if unapproved.len() > 0 {
        println!("Dependencies not on the whitelist:");
        for dep in unapproved {
            println!("* {} {}", dep.0, dep.1); // name version
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
fn check_crate_whitelist<'a>(
    whitelist: &'a HashSet<Crate>,
    resolve: &'a Resolve,
    krate: Crate<'a>,
) -> Vec<Crate<'a>> {
    // Will contain bad deps
    let mut unapproved = Vec::new();

    // If this dependency is not on the WHITELIST, add to bad set
    if !whitelist.contains(&krate) {
        unapproved.push(krate);
    }

    // Do a DFS in the crate graph (it's a DAG, so we know we have no cycles!)
    let to_check = resolve
        .nodes
        .iter()
        .find(|n| n.id.starts_with(&krate.id_str()))
        .expect("crate does not exist");

    for dep in to_check.dependencies.iter() {
        let krate = Crate::from_str(dep);
        let mut bad = check_crate_whitelist(whitelist, resolve, krate);

        unapproved.append(&mut bad);
    }

    // Remove duplicates
    unapproved.sort_unstable();
    unapproved.dedup();

    unapproved
}
