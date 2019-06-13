//! Checks the licenses of third-party dependencies by inspecting vendors.

use std::collections::{BTreeSet, HashSet, HashMap};
use std::fs;
use std::path::Path;
use std::process::Command;

use serde::Deserialize;
use serde_json;

const LICENSES: &[&str] = &[
    "MIT/Apache-2.0",
    "MIT / Apache-2.0",
    "Apache-2.0/MIT",
    "Apache-2.0 / MIT",
    "MIT OR Apache-2.0",
    "MIT",
    "Unlicense/MIT",
    "Unlicense OR MIT",
];

/// These are exceptions to Rust's permissive licensing policy, and
/// should be considered bugs. Exceptions are only allowed in Rust
/// tooling. It is _crucial_ that no exception crates be dependencies
/// of the Rust runtime (std/test).
const EXCEPTIONS: &[&str] = &[
    "mdbook",             // MPL2, mdbook
    "openssl",            // BSD+advertising clause, cargo, mdbook
    "pest",               // MPL2, mdbook via handlebars
    "arrayref",           // BSD-2-Clause, mdbook via handlebars via pest
    "thread-id",          // Apache-2.0, mdbook
    "toml-query",         // MPL-2.0, mdbook
    "is-match",           // MPL-2.0, mdbook
    "cssparser",          // MPL-2.0, rustdoc
    "smallvec",           // MPL-2.0, rustdoc
    "rdrand",             // ISC, mdbook, rustfmt
    "fuchsia-cprng",      // BSD-3-Clause, mdbook, rustfmt
    "fuchsia-zircon-sys", // BSD-3-Clause, rustdoc, rustc, cargo
    "fuchsia-zircon",     // BSD-3-Clause, rustdoc, rustc, cargo (jobserver & tempdir)
    "cssparser-macros",   // MPL-2.0, rustdoc
    "selectors",          // MPL-2.0, rustdoc
    "clippy_lints",       // MPL-2.0, rls
    "colored",            // MPL-2.0, rustfmt
    "ordslice",           // Apache-2.0, rls
    "cloudabi",           // BSD-2-Clause, (rls -> crossbeam-channel 0.2 -> rand 0.5)
    "ryu",                // Apache-2.0, rls/cargo/... (because of serde)
    "bytesize",           // Apache-2.0, cargo
    "im-rc",              // MPL-2.0+, cargo
    "adler32",            // BSD-3-Clause AND Zlib, cargo dep that isn't used
    "fortanix-sgx-abi",   // MPL-2.0+, libstd but only for `sgx` target
    "constant_time_eq",   // CC0-1.0, rustfmt
    "utf8parse",          // Apache-2.0 OR MIT, cargo via strip-ansi-escapes
    "vte",                // Apache-2.0 OR MIT, cargo via strip-ansi-escapes
    "sized-chunks",       // MPL-2.0+, cargo via im-rc
];

/// Which crates to check against the whitelist?
const WHITELIST_CRATES: &[CrateVersion<'_>] = &[
    CrateVersion("rustc", "0.0.0"),
    CrateVersion("rustc_codegen_llvm", "0.0.0"),
];

/// Whitelist of crates rustc is allowed to depend on. Avoid adding to the list if possible.
const WHITELIST: &[Crate<'_>] = &[
    Crate("adler32"),
    Crate("aho-corasick"),
    Crate("annotate-snippets"),
    Crate("ansi_term"),
    Crate("arrayvec"),
    Crate("atty"),
    Crate("autocfg"),
    Crate("backtrace"),
    Crate("backtrace-sys"),
    Crate("bitflags"),
    Crate("build_const"),
    Crate("byteorder"),
    Crate("cc"),
    Crate("cfg-if"),
    Crate("chalk-engine"),
    Crate("chalk-macros"),
    Crate("cloudabi"),
    Crate("cmake"),
    Crate("compiler_builtins"),
    Crate("crc"),
    Crate("crc32fast"),
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
    Crate("indexmap"),
    Crate("itertools"),
    Crate("jobserver"),
    Crate("kernel32-sys"),
    Crate("lazy_static"),
    Crate("libc"),
    Crate("libz-sys"),
    Crate("lock_api"),
    Crate("log"),
    Crate("log_settings"),
    Crate("measureme"),
    Crate("memchr"),
    Crate("memmap"),
    Crate("memoffset"),
    Crate("miniz-sys"),
    Crate("miniz_oxide"),
    Crate("miniz_oxide_c_api"),
    Crate("nodrop"),
    Crate("num_cpus"),
    Crate("owning_ref"),
    Crate("parking_lot"),
    Crate("parking_lot_core"),
    Crate("pkg-config"),
    Crate("polonius-engine"),
    Crate("proc-macro2"),
    Crate("quick-error"),
    Crate("quote"),
    Crate("rand"),
    Crate("rand_chacha"),
    Crate("rand_core"),
    Crate("rand_hc"),
    Crate("rand_isaac"),
    Crate("rand_pcg"),
    Crate("rand_xorshift"),
    Crate("redox_syscall"),
    Crate("redox_termios"),
    Crate("regex"),
    Crate("regex-syntax"),
    Crate("remove_dir_all"),
    Crate("rustc-demangle"),
    Crate("rustc-hash"),
    Crate("rustc-rayon"),
    Crate("rustc-rayon-core"),
    Crate("rustc_version"),
    Crate("scoped-tls"),
    Crate("scopeguard"),
    Crate("semver"),
    Crate("semver-parser"),
    Crate("serde"),
    Crate("serde_derive"),
    Crate("smallvec"),
    Crate("stable_deref_trait"),
    Crate("syn"),
    Crate("synstructure"),
    Crate("tempfile"),
    Crate("termcolor"),
    Crate("terminon"),
    Crate("termion"),
    Crate("thread_local"),
    Crate("ucd-util"),
    Crate("unicode-width"),
    Crate("unicode-xid"),
    Crate("unreachable"),
    Crate("utf8-ranges"),
    Crate("vcpkg"),
    Crate("version_check"),
    Crate("void"),
    Crate("winapi"),
    Crate("winapi-build"),
    Crate("winapi-i686-pc-windows-gnu"),
    Crate("winapi-util"),
    Crate("winapi-x86_64-pc-windows-gnu"),
    Crate("wincolor"),
];

// Some types for Serde to deserialize the output of `cargo metadata` to.

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

/// A unique identifier for a crate.
#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Debug, Hash)]
struct Crate<'a>(&'a str); // (name)

#[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Debug, Hash)]
struct CrateVersion<'a>(&'a str, &'a str); // (name, version)

impl Crate<'_> {
    pub fn id_str(&self) -> String {
        format!("{} ", self.0)
    }
}

impl<'a> CrateVersion<'a> {
    /// Returns the struct and whether or not the dependency is in-tree.
    pub fn from_str(s: &'a str) -> (Self, bool) {
        let mut parts = s.split(' ');
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
    // Check licences.
    let path = path.join("../vendor");
    assert!(path.exists(), "vendor directory missing");
    let mut saw_dir = false;
    for dir in t!(path.read_dir()) {
        saw_dir = true;
        let dir = t!(dir);

        // Skip our exceptions.
        let is_exception = EXCEPTIONS.iter().any(|exception| {
            dir.path()
                .to_str()
                .unwrap()
                .contains(&format!("vendor/{}", exception))
        });
        if is_exception {
            continue;
        }

        let toml = dir.path().join("Cargo.toml");
        *bad = !check_license(&toml) || *bad;
    }
    assert!(saw_dir, "no vendored source");
}

/// Checks the dependency of `WHITELIST_CRATES` at the given path. Changes `bad` to `true` if a
/// check failed.
///
/// Specifically, this checks that the dependencies are on the `WHITELIST`.
pub fn check_whitelist(path: &Path, cargo: &Path, bad: &mut bool) {
    // Get dependencies from Cargo metadata.
    let resolve = get_deps(path, cargo);

    // Get the whitelist in a convenient form.
    let whitelist: HashSet<_> = WHITELIST.iter().cloned().collect();

    // Check dependencies.
    let mut visited = BTreeSet::new();
    let mut unapproved = BTreeSet::new();
    for &krate in WHITELIST_CRATES.iter() {
        let mut bad = check_crate_whitelist(&whitelist, &resolve, &mut visited, krate, false);
        unapproved.append(&mut bad);
    }

    if !unapproved.is_empty() {
        println!("Dependencies not on the whitelist:");
        for dep in unapproved {
            println!("* {}", dep.id_str());
        }
        *bad = true;
    }

    check_crate_duplicate(&resolve, bad);
}

fn check_license(path: &Path) -> bool {
    if !path.exists() {
        panic!("{} does not exist", path.display());
    }
    let contents = t!(fs::read_to_string(&path));

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

/// Gets the dependencies of the crate at the given path using `cargo metadata`.
fn get_deps(path: &Path, cargo: &Path) -> Resolve {
    // Run `cargo metadata` to get the set of dependencies.
    let output = Command::new(cargo)
        .arg("metadata")
        .arg("--format-version")
        .arg("1")
        .arg("--manifest-path")
        .arg(path.join("../Cargo.toml"))
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
    whitelist: &'a HashSet<Crate<'_>>,
    resolve: &'a Resolve,
    visited: &mut BTreeSet<CrateVersion<'a>>,
    krate: CrateVersion<'a>,
    must_be_on_whitelist: bool,
) -> BTreeSet<Crate<'a>> {
    // This will contain bad deps.
    let mut unapproved = BTreeSet::new();

    // Check if we have already visited this crate.
    if visited.contains(&krate) {
        return unapproved;
    }

    visited.insert(krate);

    // If this path is in-tree, we don't require it to be on the whitelist.
    if must_be_on_whitelist {
        // If this dependency is not on `WHITELIST`, add to bad set.
        if !whitelist.contains(&krate.into()) {
            unapproved.insert(krate.into());
        }
    }

    // Do a DFS in the crate graph (it's a DAG, so we know we have no cycles!).
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

fn check_crate_duplicate(resolve: &Resolve, bad: &mut bool) {
    const FORBIDDEN_TO_HAVE_DUPLICATES: &[&str] = &[
        // These two crates take quite a long time to build, so don't allow two versions of them
        // to accidentally sneak into our dependency graph, in order to ensure we keep our CI times
        // under control.

        "cargo",
        "rustc-ap-syntax",
    ];
    let mut name_to_id: HashMap<_, Vec<_>> = HashMap::new();
    for node in resolve.nodes.iter() {
        name_to_id.entry(node.id.split_whitespace().next().unwrap())
            .or_default()
            .push(&node.id);
    }

    for name in FORBIDDEN_TO_HAVE_DUPLICATES {
        if name_to_id[name].len() <= 1 {
            continue
        }
        println!("crate `{}` is duplicated in `Cargo.lock`", name);
        for id in name_to_id[name].iter() {
            println!("  * {}", id);
        }
        *bad = true;
    }
}
