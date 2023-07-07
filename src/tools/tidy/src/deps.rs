//! Checks the licenses of third-party dependencies.

use cargo_metadata::{DepKindInfo, Metadata, Package, PackageId};
use std::collections::HashSet;
use std::path::Path;

/// These are licenses that are allowed for all crates, including the runtime,
/// rustc, tools, etc.
#[rustfmt::skip]
const LICENSES: &[&str] = &[
    // tidy-alphabetical-start
    "(MIT OR Apache-2.0) AND Unicode-DFS-2016",            // unicode_ident
    "0BSD OR MIT OR Apache-2.0",                           // adler license
    "0BSD",
    "Apache-2.0 / MIT",
    "Apache-2.0 OR MIT",
    "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT", // wasi license
    "Apache-2.0/MIT",
    "ISC",
    "MIT / Apache-2.0",
    "MIT OR Apache-2.0 OR Zlib",                           // tinyvec_macros
    "MIT OR Apache-2.0",
    "MIT OR Zlib OR Apache-2.0",                           // miniz_oxide
    "MIT",
    "MIT/Apache-2.0",
    "Unicode-DFS-2016",                                    // tinystr and icu4x
    "Unlicense OR MIT",
    "Unlicense/MIT",
    "Zlib OR Apache-2.0 OR MIT",                           // tinyvec
    // tidy-alphabetical-end
];

/// These are exceptions to Rust's permissive licensing policy, and
/// should be considered bugs. Exceptions are only allowed in Rust
/// tooling. It is _crucial_ that no exception crates be dependencies
/// of the Rust runtime (std/test).
#[rustfmt::skip]
const EXCEPTIONS: &[(&str, &str)] = &[
    // tidy-alphabetical-start
    ("ar_archive_writer", "Apache-2.0 WITH LLVM-exception"), // rustc
    ("colored", "MPL-2.0"),                                  // rustfmt
    ("dissimilar", "Apache-2.0"),                            // rustdoc, rustc_lexer (few tests) via expect-test, (dev deps)
    ("fluent-langneg", "Apache-2.0"),                        // rustc (fluent translations)
    ("fortanix-sgx-abi", "MPL-2.0"),                         // libstd but only for `sgx` target. FIXME: this dependency violates the documentation comment above.
    ("instant", "BSD-3-Clause"),                             // rustc_driver/tracing-subscriber/parking_lot
    ("mdbook", "MPL-2.0"),                                   // mdbook
    ("ryu", "Apache-2.0 OR BSL-1.0"),                        // cargo/... (because of serde)
    ("self_cell", "Apache-2.0"),                             // rustc (fluent translations)
    ("snap", "BSD-3-Clause"),                                // rustc
    // tidy-alphabetical-end
];

const EXCEPTIONS_CARGO: &[(&str, &str)] = &[
    // tidy-alphabetical-start
    ("bitmaps", "MPL-2.0+"),
    ("bytesize", "Apache-2.0"),
    ("dunce", "CC0-1.0 OR MIT-0 OR Apache-2.0"),
    ("fiat-crypto", "MIT OR Apache-2.0 OR BSD-1-Clause"),
    ("im-rc", "MPL-2.0+"),
    ("imara-diff", "Apache-2.0"),
    ("instant", "BSD-3-Clause"),
    ("normalize-line-endings", "Apache-2.0"),
    ("openssl", "Apache-2.0"),
    ("ryu", "Apache-2.0 OR BSL-1.0"),
    ("sha1_smol", "BSD-3-Clause"),
    ("similar", "Apache-2.0"),
    ("sized-chunks", "MPL-2.0+"),
    ("subtle", "BSD-3-Clause"),
    ("unicode-bom", "Apache-2.0"),
    // tidy-alphabetical-end
];

const EXCEPTIONS_CRANELIFT: &[(&str, &str)] = &[
    // tidy-alphabetical-start
    ("cranelift-bforest", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-codegen", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-codegen-meta", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-codegen-shared", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-control", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-entity", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-frontend", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-isle", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-jit", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-module", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-native", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-object", "Apache-2.0 WITH LLVM-exception"),
    ("mach", "BSD-2-Clause"),
    ("regalloc2", "Apache-2.0 WITH LLVM-exception"),
    ("target-lexicon", "Apache-2.0 WITH LLVM-exception"),
    ("wasmtime-jit-icache-coherence", "Apache-2.0 WITH LLVM-exception"),
    // tidy-alphabetical-end
];

const EXCEPTIONS_BOOTSTRAP: &[(&str, &str)] = &[
    ("ryu", "Apache-2.0 OR BSL-1.0"), // through serde
];

/// These are the root crates that are part of the runtime. The licenses for
/// these and all their dependencies *must not* be in the exception list.
const RUNTIME_CRATES: &[&str] = &["std", "core", "alloc", "test", "panic_abort", "panic_unwind"];

const PERMITTED_DEPS_LOCATION: &str = concat!(file!(), ":", line!());

/// Crates rustc is allowed to depend on. Avoid adding to the list if possible.
///
/// This list is here to provide a speed-bump to adding a new dependency to
/// rustc. Please check with the compiler team before adding an entry.
const PERMITTED_RUSTC_DEPENDENCIES: &[&str] = &[
    // tidy-alphabetical-start
    "addr2line",
    "adler",
    "ahash",
    "aho-corasick",
    "allocator-api2", // FIXME: only appears in Cargo.lock due to https://github.com/rust-lang/cargo/issues/10801
    "annotate-snippets",
    "ar_archive_writer",
    "arrayvec",
    "atty",
    "autocfg",
    "bitflags",
    "block-buffer",
    "byteorder", // via ruzstd in object in thorin-dwp
    "cc",
    "cfg-if",
    "compiler_builtins",
    "convert_case", // dependency of derive_more
    "cpufeatures",
    "crc32fast",
    "crossbeam-channel",
    "crossbeam-deque",
    "crossbeam-epoch",
    "crossbeam-utils",
    "crypto-common",
    "cstr",
    "datafrog",
    "derive_more",
    "digest",
    "displaydoc",
    "dissimilar",
    "dlmalloc",
    "either",
    "elsa",
    "ena",
    "equivalent",
    "expect-test",
    "fallible-iterator", // dependency of `thorin`
    "fastrand",
    "field-offset",
    "flate2",
    "fluent-bundle",
    "fluent-langneg",
    "fluent-syntax",
    "fortanix-sgx-abi",
    "generic-array",
    "getopts",
    "getrandom",
    "gimli",
    "gsgdt",
    "hashbrown",
    "hermit-abi",
    "icu_list",
    "icu_locid",
    "icu_provider",
    "icu_provider_adapters",
    "icu_provider_macros",
    "indexmap",
    "instant",
    "intl-memoizer",
    "intl_pluralrules",
    "io-lifetimes",
    "itertools",
    "itoa",
    "jobserver",
    "lazy_static",
    "libc",
    "libloading",
    "linux-raw-sys",
    "litemap",
    "lock_api",
    "log",
    "matchers",
    "md-5",
    "measureme",
    "memchr",
    "memmap2",
    "memoffset",
    "miniz_oxide",
    "nu-ansi-term",
    "num_cpus",
    "object",
    "odht",
    "once_cell",
    "overload",
    "parking_lot",
    "parking_lot_core",
    "pathdiff",
    "perf-event-open-sys",
    "pin-project-lite",
    "polonius-engine",
    "ppv-lite86",
    "proc-macro-hack",
    "proc-macro2",
    "psm",
    "pulldown-cmark",
    "punycode",
    "quote",
    "rand",
    "rand_chacha",
    "rand_core",
    "rand_xorshift",
    "rand_xoshiro",
    "redox_syscall",
    "regex",
    "regex-automata",
    "regex-syntax",
    "rustc-demangle",
    "rustc-hash",
    "rustc-rayon",
    "rustc-rayon-core",
    "rustc_version",
    "rustix",
    "ruzstd", // via object in thorin-dwp
    "ryu",
    "scoped-tls",
    "scopeguard",
    "self_cell",
    "semver",
    "serde",
    "serde_derive",
    "serde_json",
    "sha1",
    "sha2",
    "sharded-slab",
    "smallvec",
    "snap",
    "stable_deref_trait",
    "stacker",
    "static_assertions",
    "syn",
    "synstructure",
    "tempfile",
    "termcolor",
    "termize",
    "thin-vec",
    "thiserror",
    "thiserror-impl",
    "thorin-dwp",
    "thread_local",
    "tinystr",
    "tinyvec",
    "tinyvec_macros",
    "tracing",
    "tracing-attributes",
    "tracing-core",
    "tracing-log",
    "tracing-subscriber",
    "tracing-tree",
    "twox-hash",
    "type-map",
    "typenum",
    "unic-char-property",
    "unic-char-range",
    "unic-common",
    "unic-emoji-char",
    "unic-langid",
    "unic-langid-impl",
    "unic-langid-macros",
    "unic-langid-macros-impl",
    "unic-ucd-version",
    "unicase",
    "unicode-ident",
    "unicode-normalization",
    "unicode-script",
    "unicode-security",
    "unicode-width",
    "unicode-xid",
    "valuable",
    "version_check",
    "wasi",
    "winapi",
    "winapi-i686-pc-windows-gnu",
    "winapi-util",
    "winapi-x86_64-pc-windows-gnu",
    "windows",
    "windows-sys",
    "windows-targets",
    "windows_aarch64_gnullvm",
    "windows_aarch64_msvc",
    "windows_i686_gnu",
    "windows_i686_msvc",
    "windows_x86_64_gnu",
    "windows_x86_64_gnullvm",
    "windows_x86_64_msvc",
    "writeable",
    "yansi-term", // this is a false-positive: it's only used by rustfmt, but because it's enabled through a feature, tidy thinks it's used by rustc as well.
    "yoke",
    "yoke-derive",
    "zerofrom",
    "zerofrom-derive",
    "zerovec",
    "zerovec-derive",
    // tidy-alphabetical-end
];

const PERMITTED_CRANELIFT_DEPENDENCIES: &[&str] = &[
    // tidy-alphabetical-start
    "ahash",
    "anyhow",
    "arbitrary",
    "autocfg",
    "bitflags",
    "bumpalo",
    "cfg-if",
    "cranelift-bforest",
    "cranelift-codegen",
    "cranelift-codegen-meta",
    "cranelift-codegen-shared",
    "cranelift-control",
    "cranelift-entity",
    "cranelift-frontend",
    "cranelift-isle",
    "cranelift-jit",
    "cranelift-module",
    "cranelift-native",
    "cranelift-object",
    "crc32fast",
    "fallible-iterator",
    "gimli",
    "hashbrown",
    "indexmap",
    "libc",
    "libloading",
    "log",
    "mach",
    "memchr",
    "object",
    "regalloc2",
    "region",
    "rustc-hash",
    "slice-group-by",
    "smallvec",
    "stable_deref_trait",
    "target-lexicon",
    "version_check",
    "wasmtime-jit-icache-coherence",
    "winapi",
    "winapi-i686-pc-windows-gnu",
    "winapi-x86_64-pc-windows-gnu",
    "windows-sys",
    "windows-targets",
    "windows_aarch64_gnullvm",
    "windows_aarch64_msvc",
    "windows_i686_gnu",
    "windows_i686_msvc",
    "windows_x86_64_gnu",
    "windows_x86_64_gnullvm",
    "windows_x86_64_msvc",
    // tidy-alphabetical-end
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
    check_license_exceptions(&metadata, EXCEPTIONS, runtime_ids, bad);
    check_permitted_dependencies(
        &metadata,
        "rustc",
        PERMITTED_RUSTC_DEPENDENCIES,
        &["rustc_driver", "rustc_codegen_llvm"],
        bad,
    );

    // Check cargo independently as it has it's own workspace.
    let mut cmd = cargo_metadata::MetadataCommand::new();
    cmd.cargo_path(cargo)
        .manifest_path(root.join("src/tools/cargo/Cargo.toml"))
        .features(cargo_metadata::CargoOpt::AllFeatures);
    let cargo_metadata = t!(cmd.exec());
    let runtime_ids = HashSet::new();
    check_license_exceptions(&cargo_metadata, EXCEPTIONS_CARGO, runtime_ids, bad);
    check_rustfix(&metadata, &cargo_metadata, bad);

    // Check rustc_codegen_cranelift independently as it has it's own workspace.
    let mut cmd = cargo_metadata::MetadataCommand::new();
    cmd.cargo_path(cargo)
        .manifest_path(root.join("compiler/rustc_codegen_cranelift/Cargo.toml"))
        .features(cargo_metadata::CargoOpt::AllFeatures);
    let metadata = t!(cmd.exec());
    let runtime_ids = HashSet::new();
    check_license_exceptions(&metadata, EXCEPTIONS_CRANELIFT, runtime_ids, bad);
    check_permitted_dependencies(
        &metadata,
        "cranelift",
        PERMITTED_CRANELIFT_DEPENDENCIES,
        &["rustc_codegen_cranelift"],
        bad,
    );

    let mut cmd = cargo_metadata::MetadataCommand::new();
    cmd.cargo_path(cargo)
        .manifest_path(root.join("src/bootstrap/Cargo.toml"))
        .features(cargo_metadata::CargoOpt::AllFeatures);
    let metadata = t!(cmd.exec());
    let runtime_ids = HashSet::new();
    check_license_exceptions(&metadata, EXCEPTIONS_BOOTSTRAP, runtime_ids, bad);
}

/// Check that all licenses are in the valid list in `LICENSES`.
///
/// Packages listed in `exceptions` are allowed for tools.
fn check_license_exceptions(
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
                        println!("dependency exception `{name}` license has changed");
                        println!("    previously `{license}` now `{pkg_license}`");
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

/// Checks the dependency of `restricted_dependency_crates` at the given path. Changes `bad` to
/// `true` if a check failed.
///
/// Specifically, this checks that the dependencies are on the `permitted_dependencies`.
fn check_permitted_dependencies(
    metadata: &Metadata,
    descr: &str,
    permitted_dependencies: &[&'static str],
    restricted_dependency_crates: &[&'static str],
    bad: &mut bool,
) {
    let mut has_permitted_dep_error = false;
    let mut deps = HashSet::new();
    for to_check in restricted_dependency_crates {
        let to_check = pkg_from_name(metadata, to_check);
        use cargo_platform::Cfg;
        use std::str::FromStr;
        // We don't expect the compiler to ever run on wasm32, so strip
        // out those dependencies to avoid polluting the permitted list.
        deps_of_filtered(metadata, &to_check.id, &mut deps, &|dep_kinds| {
            dep_kinds.iter().any(|dep_kind| {
                dep_kind
                    .target
                    .as_ref()
                    .map(|target| {
                        !target.matches(
                            "wasm32-unknown-unknown",
                            &[
                                Cfg::from_str("target_arch=\"wasm32\"").unwrap(),
                                Cfg::from_str("target_os=\"unknown\"").unwrap(),
                            ],
                        )
                    })
                    .unwrap_or(true)
            })
        });
    }

    // Check that the PERMITTED_DEPENDENCIES does not have unused entries.
    for permitted in permitted_dependencies {
        if !deps.iter().any(|dep_id| &pkg_from_id(metadata, dep_id).name == permitted) {
            tidy_error!(
                bad,
                "could not find allowed package `{permitted}`\n\
                Remove from PERMITTED_DEPENDENCIES list if it is no longer used.",
            );
            has_permitted_dep_error = true;
        }
    }

    // Get in a convenient form.
    let permitted_dependencies: HashSet<_> = permitted_dependencies.iter().cloned().collect();

    for dep in deps {
        let dep = pkg_from_id(metadata, dep);
        // If this path is in-tree, we don't require it to be explicitly permitted.
        if dep.source.is_some() {
            if !permitted_dependencies.contains(dep.name.as_str()) {
                tidy_error!(bad, "Dependency for {descr} not explicitly permitted: {}", dep.id);
                has_permitted_dep_error = true;
            }
        }
    }

    if has_permitted_dep_error {
        eprintln!("Go to `{PERMITTED_DEPS_LOCATION}` for the list.");
    }
}

/// Finds a package with the given name.
fn pkg_from_name<'a>(metadata: &'a Metadata, name: &'static str) -> &'a Package {
    let mut i = metadata.packages.iter().filter(|p| p.name == name);
    let result =
        i.next().unwrap_or_else(|| panic!("could not find package `{name}` in package list"));
    assert!(i.next().is_none(), "more than one package found for `{name}`");
    result
}

fn pkg_from_id<'a>(metadata: &'a Metadata, id: &PackageId) -> &'a Package {
    metadata.packages.iter().find(|p| &p.id == id).unwrap()
}

/// Finds all the packages that are in the rust runtime.
fn compute_runtime_crates<'a>(metadata: &'a Metadata) -> HashSet<&'a PackageId> {
    let mut result = HashSet::new();
    for name in RUNTIME_CRATES {
        let id = &pkg_from_name(metadata, name).id;
        deps_of_filtered(metadata, id, &mut result, &|_| true);
    }
    result
}

/// Recursively find all dependencies.
fn deps_of_filtered<'a>(
    metadata: &'a Metadata,
    pkg_id: &'a PackageId,
    result: &mut HashSet<&'a PackageId>,
    filter: &dyn Fn(&[DepKindInfo]) -> bool,
) {
    if !result.insert(pkg_id) {
        return;
    }
    let node = metadata
        .resolve
        .as_ref()
        .unwrap()
        .nodes
        .iter()
        .find(|n| &n.id == pkg_id)
        .unwrap_or_else(|| panic!("could not find `{pkg_id}` in resolve"));
    for dep in &node.deps {
        if !filter(&dep.dep_kinds) {
            continue;
        }
        deps_of_filtered(metadata, &dep.pkg, result, filter);
    }
}

fn direct_deps_of<'a>(
    metadata: &'a Metadata,
    pkg_id: &'a PackageId,
) -> impl Iterator<Item = &'a Package> {
    let resolve = metadata.resolve.as_ref().unwrap();
    let node = resolve.nodes.iter().find(|n| &n.id == pkg_id).unwrap();
    node.deps.iter().map(|dep| pkg_from_id(metadata, &dep.pkg))
}

fn check_rustfix(rust_metadata: &Metadata, cargo_metadata: &Metadata, bad: &mut bool) {
    let cargo = pkg_from_name(cargo_metadata, "cargo");
    let cargo_rustfix =
        direct_deps_of(cargo_metadata, &cargo.id).find(|p| p.name == "rustfix").unwrap();

    let compiletest = pkg_from_name(rust_metadata, "compiletest");
    let compiletest_rustfix =
        direct_deps_of(rust_metadata, &compiletest.id).find(|p| p.name == "rustfix").unwrap();

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
