//! Checks the licenses of third-party dependencies.

use cargo_metadata::{Metadata, Package, PackageId, Resolve};
use std::collections::{BTreeSet, HashSet};
use std::path::Path;

/// These are licenses that are allowed for all crates, including the runtime,
/// rustc, tools, etc.
const LICENSES: &[&str] = &[
    "MIT/Apache-2.0",
    "MIT / Apache-2.0",
    "Apache-2.0/MIT",
    "Apache-2.0 / MIT",
    "MIT OR Apache-2.0",
    "Apache-2.0 OR MIT",
    "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT", // wasi license
    "MIT",
    "ISC",
    "Unlicense/MIT",
    "Unlicense OR MIT",
    "0BSD OR MIT OR Apache-2.0", // adler license
    "Zlib OR Apache-2.0 OR MIT", // tinyvec
];

/// These are exceptions to Rust's permissive licensing policy, and
/// should be considered bugs. Exceptions are only allowed in Rust
/// tooling. It is _crucial_ that no exception crates be dependencies
/// of the Rust runtime (std/test).
const EXCEPTIONS: &[(&str, &str)] = &[
    ("mdbook", "MPL-2.0"),            // mdbook
    ("openssl", "Apache-2.0"),        // cargo, mdbook
    ("colored", "MPL-2.0"),           // rustfmt
    ("ordslice", "Apache-2.0"),       // rls
    ("ryu", "Apache-2.0 OR BSL-1.0"), // rls/cargo/... (because of serde)
    ("bytesize", "Apache-2.0"),       // cargo
    ("im-rc", "MPL-2.0+"),            // cargo
    ("sized-chunks", "MPL-2.0+"),     // cargo via im-rc
    ("bitmaps", "MPL-2.0+"),          // cargo via im-rc
    ("instant", "BSD-3-Clause"),      // rustc_driver/tracing-subscriber/parking_lot
    ("snap", "BSD-3-Clause"),         // rustc
    // FIXME: this dependency violates the documentation comment above:
    ("fortanix-sgx-abi", "MPL-2.0"), // libstd but only for `sgx` target
];

const EXCEPTIONS_CRANELIFT: &[(&str, &str)] = &[
    ("cranelift-bforest", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-codegen", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-codegen-meta", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-codegen-shared", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-entity", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-frontend", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-jit", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-module", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-native", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-object", "Apache-2.0 WITH LLVM-exception"),
    ("mach", "BSD-2-Clause"),
    ("regalloc", "Apache-2.0 WITH LLVM-exception"),
    ("target-lexicon", "Apache-2.0 WITH LLVM-exception"),
];

/// These are the root crates that are part of the runtime. The licenses for
/// these and all their dependencies *must not* be in the exception list.
const RUNTIME_CRATES: &[&str] = &["std", "core", "alloc", "test", "panic_abort", "panic_unwind"];

/// Crates whose dependencies must be explicitly permitted.
const RESTRICTED_DEPENDENCY_CRATES: &[&str] = &["rustc_driver", "rustc_codegen_llvm"];

/// Crates rustc is allowed to depend on. Avoid adding to the list if possible.
///
/// This list is here to provide a speed-bump to adding a new dependency to
/// rustc. Please check with the compiler team before adding an entry.
const PERMITTED_DEPENDENCIES: &[&str] = &[
    "addr2line",
    "adler",
    "ahash",
    "aho-corasick",
    "annotate-snippets",
    "ansi_term",
    "arrayvec",
    "atty",
    "autocfg",
    "bitflags",
    "block-buffer",
    "block-padding",
    "byte-tools",
    "byteorder",
    "cc",
    "cfg-if",
    "chalk-derive",
    "chalk-engine",
    "chalk-ir",
    "chalk-solve",
    "chrono",
    "cmake",
    "compiler_builtins",
    "cpufeatures",
    "crc32fast",
    "crossbeam-deque",
    "crossbeam-epoch",
    "crossbeam-utils",
    "crypto-common",
    "cstr",
    "datafrog",
    "difference",
    "digest",
    "dlmalloc",
    "either",
    "ena",
    "env_logger",
    "expect-test",
    "fake-simd",
    "fallible-iterator", // dependency of `thorin`
    "filetime",
    "fixedbitset",
    "flate2",
    "fortanix-sgx-abi",
    "generic-array",
    "getopts",
    "getrandom",
    "gimli",
    "gsgdt",
    "hashbrown",
    "hermit-abi",
    "humantime",
    "if_chain",
    "indexmap",
    "instant",
    "itertools",
    "itoa",
    "jobserver",
    "lazy_static",
    "libc",
    "libloading",
    "libz-sys",
    "lock_api",
    "log",
    "matchers",
    "md-5",
    "measureme",
    "memchr",
    "memmap2",
    "memoffset",
    "miniz_oxide",
    "num-integer",
    "num-traits",
    "num_cpus",
    "object",
    "odht",
    "once_cell",
    "opaque-debug",
    "parking_lot",
    "parking_lot_core",
    "pathdiff",
    "perf-event-open-sys",
    "petgraph",
    "pin-project-lite",
    "pkg-config",
    "polonius-engine",
    "ppv-lite86",
    "proc-macro2",
    "psm",
    "punycode",
    "quick-error",
    "quote",
    "rand",
    "rand_chacha",
    "rand_core",
    "rand_hc",
    "rand_pcg",
    "rand_xorshift",
    "rand_xoshiro",
    "redox_syscall",
    "regex",
    "regex-automata",
    "regex-syntax",
    "remove_dir_all",
    "rls-data",
    "rls-span",
    "rustc-demangle",
    "rustc-hash",
    "rustc-rayon",
    "rustc-rayon-core",
    "rustc_version",
    "ryu",
    "scoped-tls",
    "scopeguard",
    "semver",
    "serde",
    "serde_derive",
    "serde_json",
    "sha-1",
    "sha2",
    "sharded-slab",
    "smallvec",
    "snap",
    "stable_deref_trait",
    "stacker",
    "syn",
    "synstructure",
    "tempfile",
    "termcolor",
    "termize",
    "thorin-dwp",
    "thread_local",
    "time",
    "tinyvec",
    "tracing",
    "tracing-attributes",
    "tracing-core",
    "tracing-log",
    "tracing-subscriber",
    "tracing-tree",
    "typenum",
    "unic-char-property",
    "unic-char-range",
    "unic-common",
    "unic-emoji-char",
    "unic-ucd-version",
    "unicode-normalization",
    "unicode-script",
    "unicode-security",
    "unicode-width",
    "unicode-xid",
    "vcpkg",
    "version_check",
    "wasi",
    "winapi",
    "winapi-i686-pc-windows-gnu",
    "winapi-util",
    "winapi-x86_64-pc-windows-gnu",
    // this is a false-positive: it's only used by rustfmt, but because it's enabled through a feature, tidy thinks it's used by rustc as well.
    "yansi-term",
];

const PERMITTED_CRANELIFT_DEPENDENCIES: &[&str] = &[
    "anyhow",
    "ar",
    "autocfg",
    "bitflags",
    "cfg-if",
    "cranelift-bforest",
    "cranelift-codegen",
    "cranelift-codegen-meta",
    "cranelift-codegen-shared",
    "cranelift-entity",
    "cranelift-frontend",
    "cranelift-jit",
    "cranelift-module",
    "cranelift-native",
    "cranelift-object",
    "crc32fast",
    "gimli",
    "hashbrown",
    "indexmap",
    "libc",
    "libloading",
    "log",
    "mach",
    "memchr",
    "object",
    "regalloc",
    "region",
    "rustc-hash",
    "smallvec",
    "target-lexicon",
    "winapi",
    "winapi-i686-pc-windows-gnu",
    "winapi-x86_64-pc-windows-gnu",
];

const FORBIDDEN_TO_HAVE_DUPLICATES: &[&str] = &[
    // These two crates take quite a long time to build, so don't allow two versions of them
    // to accidentally sneak into our dependency graph, in order to ensure we keep our CI times
    // under control.
    "cargo",
];

/// Dependency checks.
///
/// `root` is path to the directory with the root `Cargo.toml` (for the workspace). `cargo` is path
/// to the cargo executable.
pub fn check(root: &Path, cargo: &Path, bad: &mut bool) {
    let mut cmd = cargo_metadata::MetadataCommand::new();
    cmd.cargo_path(cargo)
        .manifest_path(root.join("Cargo.toml"))
        .features(cargo_metadata::CargoOpt::AllFeatures);
    let metadata = t!(cmd.exec());
    let runtime_ids = compute_runtime_crates(&metadata);
    check_exceptions(&metadata, EXCEPTIONS, runtime_ids, bad);
    check_dependencies(&metadata, PERMITTED_DEPENDENCIES, RESTRICTED_DEPENDENCY_CRATES, bad);
    check_crate_duplicate(&metadata, FORBIDDEN_TO_HAVE_DUPLICATES, bad);
    check_rustfix(&metadata, bad);

    // Check rustc_codegen_cranelift independently as it has it's own workspace.
    let mut cmd = cargo_metadata::MetadataCommand::new();
    cmd.cargo_path(cargo)
        .manifest_path(root.join("compiler/rustc_codegen_cranelift/Cargo.toml"))
        .features(cargo_metadata::CargoOpt::AllFeatures);
    let metadata = t!(cmd.exec());
    let runtime_ids = HashSet::new();
    check_exceptions(&metadata, EXCEPTIONS_CRANELIFT, runtime_ids, bad);
    check_dependencies(
        &metadata,
        PERMITTED_CRANELIFT_DEPENDENCIES,
        &["rustc_codegen_cranelift"],
        bad,
    );
    check_crate_duplicate(&metadata, &[], bad);
}

/// Check that all licenses are in the valid list in `LICENSES`.
///
/// Packages listed in `EXCEPTIONS` are allowed for tools.
fn check_exceptions(
    metadata: &Metadata,
    exceptions: &[(&str, &str)],
    runtime_ids: HashSet<&PackageId>,
    bad: &mut bool,
) {
    // Validate the EXCEPTIONS list hasn't changed.
    for (name, license) in exceptions {
        // Check that the package actually exists.
        if !metadata.packages.iter().any(|p| p.name == *name) {
            tidy_error!(
                bad,
                "could not find exception package `{}`\n\
                Remove from EXCEPTIONS list if it is no longer used.",
                name
            );
        }
        // Check that the license hasn't changed.
        for pkg in metadata.packages.iter().filter(|p| p.name == *name) {
            match &pkg.license {
                None => {
                    tidy_error!(
                        bad,
                        "dependency exception `{}` does not declare a license expression",
                        pkg.id
                    );
                }
                Some(pkg_license) => {
                    if pkg_license.as_str() != *license {
                        println!("dependency exception `{}` license has changed", name);
                        println!("    previously `{}` now `{}`", license, pkg_license);
                        println!("    update EXCEPTIONS for the new license");
                        *bad = true;
                    }
                }
            }
        }
    }

    let exception_names: Vec<_> = exceptions.iter().map(|(name, _license)| *name).collect();

    // Check if any package does not have a valid license.
    for pkg in &metadata.packages {
        if pkg.source.is_none() {
            // No need to check local packages.
            continue;
        }
        if !runtime_ids.contains(&pkg.id) && exception_names.contains(&pkg.name.as_str()) {
            continue;
        }
        let license = match &pkg.license {
            Some(license) => license,
            None => {
                tidy_error!(bad, "dependency `{}` does not define a license expression", pkg.id);
                continue;
            }
        };
        if !LICENSES.contains(&license.as_str()) {
            if pkg.name == "fortanix-sgx-abi" {
                // This is a specific exception because SGX is considered
                // "third party". See
                // https://github.com/rust-lang/rust/issues/62620 for more. In
                // general, these should never be added.
                continue;
            }
            tidy_error!(bad, "invalid license `{}` in `{}`", license, pkg.id);
        }
    }
}

/// Checks the dependency of `RESTRICTED_DEPENDENCY_CRATES` at the given path. Changes `bad` to
/// `true` if a check failed.
///
/// Specifically, this checks that the dependencies are on the `PERMITTED_DEPENDENCIES`.
fn check_dependencies(
    metadata: &Metadata,
    permitted_dependencies: &[&'static str],
    restricted_dependency_crates: &[&'static str],
    bad: &mut bool,
) {
    // Check that the PERMITTED_DEPENDENCIES does not have unused entries.
    for name in permitted_dependencies {
        if !metadata.packages.iter().any(|p| p.name == *name) {
            tidy_error!(
                bad,
                "could not find allowed package `{}`\n\
                Remove from PERMITTED_DEPENDENCIES list if it is no longer used.",
                name
            );
        }
    }
    // Get the list in a convenient form.
    let permitted_dependencies: HashSet<_> = permitted_dependencies.iter().cloned().collect();

    // Check dependencies.
    let mut visited = BTreeSet::new();
    let mut unapproved = BTreeSet::new();
    for &krate in restricted_dependency_crates.iter() {
        let pkg = pkg_from_name(metadata, krate);
        let mut bad =
            check_crate_dependencies(&permitted_dependencies, metadata, &mut visited, pkg);
        unapproved.append(&mut bad);
    }

    if !unapproved.is_empty() {
        tidy_error!(bad, "Dependencies not explicitly permitted:");
        for dep in unapproved {
            println!("* {}", dep);
        }
    }
}

/// Checks the dependencies of the given crate from the given cargo metadata to see if they are on
/// the list of permitted dependencies. Returns a list of disallowed dependencies.
fn check_crate_dependencies<'a>(
    permitted_dependencies: &'a HashSet<&'static str>,
    metadata: &'a Metadata,
    visited: &mut BTreeSet<&'a PackageId>,
    krate: &'a Package,
) -> BTreeSet<&'a PackageId> {
    // This will contain bad deps.
    let mut unapproved = BTreeSet::new();

    // Check if we have already visited this crate.
    if visited.contains(&krate.id) {
        return unapproved;
    }

    visited.insert(&krate.id);

    // If this path is in-tree, we don't require it to be explicitly permitted.
    if krate.source.is_some() {
        // If this dependency is not on `PERMITTED_DEPENDENCIES`, add to bad set.
        if !permitted_dependencies.contains(krate.name.as_str()) {
            unapproved.insert(&krate.id);
        }
    }

    // Do a DFS in the crate graph.
    let to_check = deps_of(metadata, &krate.id);

    for dep in to_check {
        let mut bad = check_crate_dependencies(permitted_dependencies, metadata, visited, dep);
        unapproved.append(&mut bad);
    }

    unapproved
}

/// Prevents multiple versions of some expensive crates.
fn check_crate_duplicate(
    metadata: &Metadata,
    forbidden_to_have_duplicates: &[&str],
    bad: &mut bool,
) {
    for &name in forbidden_to_have_duplicates {
        let matches: Vec<_> = metadata.packages.iter().filter(|pkg| pkg.name == name).collect();
        match matches.len() {
            0 => {
                tidy_error!(
                    bad,
                    "crate `{}` is missing, update `check_crate_duplicate` \
                    if it is no longer used",
                    name
                );
            }
            1 => {}
            _ => {
                tidy_error!(
                    bad,
                    "crate `{}` is duplicated in `Cargo.lock`, \
                    it is too expensive to build multiple times, \
                    so make sure only one version appears across all dependencies",
                    name
                );
                for pkg in matches {
                    println!("  * {}", pkg.id);
                }
            }
        }
    }
}

/// Returns a list of dependencies for the given package.
fn deps_of<'a>(metadata: &'a Metadata, pkg_id: &'a PackageId) -> Vec<&'a Package> {
    let resolve = metadata.resolve.as_ref().unwrap();
    let node = resolve
        .nodes
        .iter()
        .find(|n| &n.id == pkg_id)
        .unwrap_or_else(|| panic!("could not find `{}` in resolve", pkg_id));
    node.deps
        .iter()
        .map(|dep| {
            metadata.packages.iter().find(|pkg| pkg.id == dep.pkg).unwrap_or_else(|| {
                panic!("could not find dep `{}` for pkg `{}` in resolve", dep.pkg, pkg_id)
            })
        })
        .collect()
}

/// Finds a package with the given name.
fn pkg_from_name<'a>(metadata: &'a Metadata, name: &'static str) -> &'a Package {
    let mut i = metadata.packages.iter().filter(|p| p.name == name);
    let result =
        i.next().unwrap_or_else(|| panic!("could not find package `{}` in package list", name));
    assert!(i.next().is_none(), "more than one package found for `{}`", name);
    result
}

/// Finds all the packages that are in the rust runtime.
fn compute_runtime_crates<'a>(metadata: &'a Metadata) -> HashSet<&'a PackageId> {
    let resolve = metadata.resolve.as_ref().unwrap();
    let mut result = HashSet::new();
    for name in RUNTIME_CRATES {
        let id = &pkg_from_name(metadata, name).id;
        normal_deps_of_r(resolve, id, &mut result);
    }
    result
}

/// Recursively find all normal dependencies.
fn normal_deps_of_r<'a>(
    resolve: &'a Resolve,
    pkg_id: &'a PackageId,
    result: &mut HashSet<&'a PackageId>,
) {
    if !result.insert(pkg_id) {
        return;
    }
    let node = resolve
        .nodes
        .iter()
        .find(|n| &n.id == pkg_id)
        .unwrap_or_else(|| panic!("could not find `{}` in resolve", pkg_id));
    for dep in &node.deps {
        normal_deps_of_r(resolve, &dep.pkg, result);
    }
}

fn check_rustfix(metadata: &Metadata, bad: &mut bool) {
    let cargo = pkg_from_name(metadata, "cargo");
    let compiletest = pkg_from_name(metadata, "compiletest");
    let cargo_deps = deps_of(metadata, &cargo.id);
    let compiletest_deps = deps_of(metadata, &compiletest.id);
    let cargo_rustfix = cargo_deps.iter().find(|p| p.name == "rustfix").unwrap();
    let compiletest_rustfix = compiletest_deps.iter().find(|p| p.name == "rustfix").unwrap();
    if cargo_rustfix.version != compiletest_rustfix.version {
        tidy_error!(
            bad,
            "cargo's rustfix version {} does not match compiletest's rustfix version {}\n\
             rustfix should be kept in sync, update the cargo side first, and then update \
             compiletest along with cargo.",
            cargo_rustfix.version,
            compiletest_rustfix.version
        );
    }
}
