//! Checks the licenses of third-party dependencies.

use std::collections::{HashMap, HashSet};
use std::fs::{File, read_dir};
use std::io::Write;
use std::path::Path;

use build_helper::ci::CiEnv;
use cargo_metadata::semver::Version;
use cargo_metadata::{Metadata, Package, PackageId};

#[path = "../../../bootstrap/src/utils/proc_macro_deps.rs"]
mod proc_macro_deps;

/// These are licenses that are allowed for all crates, including the runtime,
/// rustc, tools, etc.
#[rustfmt::skip]
const LICENSES: &[&str] = &[
    // tidy-alphabetical-start
    "(MIT OR Apache-2.0) AND Unicode-3.0",                 // unicode_ident (1.0.14)
    "(MIT OR Apache-2.0) AND Unicode-DFS-2016",            // unicode_ident (1.0.12)
    "0BSD OR MIT OR Apache-2.0",                           // adler2 license
    "0BSD",
    "Apache-2.0 / MIT",
    "Apache-2.0 OR ISC OR MIT",
    "Apache-2.0 OR MIT",
    "Apache-2.0 WITH LLVM-exception OR Apache-2.0 OR MIT", // wasi license
    "Apache-2.0",
    "Apache-2.0/MIT",
    "BSD-2-Clause OR Apache-2.0 OR MIT",                   // zerocopy
    "ISC",
    "MIT / Apache-2.0",
    "MIT AND (MIT OR Apache-2.0)",
    "MIT AND Apache-2.0 WITH LLVM-exception AND (MIT OR Apache-2.0)", // compiler-builtins
    "MIT OR Apache-2.0 OR LGPL-2.1-or-later",              // r-efi, r-efi-alloc
    "MIT OR Apache-2.0 OR Zlib",                           // tinyvec_macros
    "MIT OR Apache-2.0",
    "MIT OR Zlib OR Apache-2.0",                           // miniz_oxide
    "MIT",
    "MIT/Apache-2.0",
    "Unicode-3.0",                                         // icu4x
    "Unicode-DFS-2016",                                    // tinystr
    "Unlicense OR MIT",
    "Unlicense/MIT",
    "Zlib OR Apache-2.0 OR MIT",                           // tinyvec
    // tidy-alphabetical-end
];

type ExceptionList = &'static [(&'static str, &'static str)];

/// The workspaces to check for licensing and optionally permitted dependencies.
///
/// Each entry consists of a tuple with the following elements:
///
/// * The path to the workspace root Cargo.toml file.
/// * The list of license exceptions.
/// * Optionally a tuple of:
///     * A list of crates for which dependencies need to be explicitly allowed.
///     * The list of allowed dependencies.
/// * Submodules required for the workspace.
// FIXME auto detect all cargo workspaces
pub(crate) const WORKSPACES: &[(&str, ExceptionList, Option<(&[&str], &[&str])>, &[&str])] = &[
    // The root workspace has to be first for check_rustfix to work.
    (".", EXCEPTIONS, Some((&["rustc-main"], PERMITTED_RUSTC_DEPENDENCIES)), &[]),
    ("library", EXCEPTIONS_STDLIB, Some((&["sysroot"], PERMITTED_STDLIB_DEPENDENCIES)), &[]),
    // Outside of the alphabetical section because rustfmt formats it using multiple lines.
    (
        "compiler/rustc_codegen_cranelift",
        EXCEPTIONS_CRANELIFT,
        Some((&["rustc_codegen_cranelift"], PERMITTED_CRANELIFT_DEPENDENCIES)),
        &[],
    ),
    // tidy-alphabetical-start
    ("compiler/rustc_codegen_gcc", EXCEPTIONS_GCC, None, &[]),
    ("src/bootstrap", EXCEPTIONS_BOOTSTRAP, None, &[]),
    ("src/ci/docker/host-x86_64/test-various/uefi_qemu_test", EXCEPTIONS_UEFI_QEMU_TEST, None, &[]),
    ("src/tools/cargo", EXCEPTIONS_CARGO, None, &["src/tools/cargo"]),
    //("src/tools/miri/test-cargo-miri", &[], None), // FIXME uncomment once all deps are vendored
    //("src/tools/miri/test_dependencies", &[], None), // FIXME uncomment once all deps are vendored
    ("src/tools/rust-analyzer", EXCEPTIONS_RUST_ANALYZER, None, &[]),
    ("src/tools/rustbook", EXCEPTIONS_RUSTBOOK, None, &["src/doc/book", "src/doc/reference"]),
    ("src/tools/rustc-perf", EXCEPTIONS_RUSTC_PERF, None, &["src/tools/rustc-perf"]),
    ("src/tools/test-float-parse", EXCEPTIONS, None, &[]),
    // tidy-alphabetical-end
];

/// These are exceptions to Rust's permissive licensing policy, and
/// should be considered bugs. Exceptions are only allowed in Rust
/// tooling. It is _crucial_ that no exception crates be dependencies
/// of the Rust runtime (std/test).
#[rustfmt::skip]
const EXCEPTIONS: ExceptionList = &[
    // tidy-alphabetical-start
    ("ar_archive_writer", "Apache-2.0 WITH LLVM-exception"), // rustc
    ("arrayref", "BSD-2-Clause"),                            // rustc
    ("blake3", "CC0-1.0 OR Apache-2.0 OR Apache-2.0 WITH LLVM-exception"),  // rustc
    ("colored", "MPL-2.0"),                                  // rustfmt
    ("constant_time_eq", "CC0-1.0 OR MIT-0 OR Apache-2.0"),  // rustc
    ("dissimilar", "Apache-2.0"),                            // rustdoc, rustc_lexer (few tests) via expect-test, (dev deps)
    ("fluent-langneg", "Apache-2.0"),                        // rustc (fluent translations)
    ("foldhash", "Zlib"),                                    // rustc
    ("option-ext", "MPL-2.0"),                               // cargo-miri (via `directories`)
    ("rustc_apfloat", "Apache-2.0 WITH LLVM-exception"),     // rustc (license is the same as LLVM uses)
    ("ryu", "Apache-2.0 OR BSL-1.0"), // BSL is not acceptble, but we use it under Apache-2.0                       // cargo/... (because of serde)
    ("self_cell", "Apache-2.0"),                             // rustc (fluent translations)
    ("wasi-preview1-component-adapter-provider", "Apache-2.0 WITH LLVM-exception"), // rustc
    // tidy-alphabetical-end
];

/// These are exceptions to Rust's permissive licensing policy, and
/// should be considered bugs. Exceptions are only allowed in Rust
/// tooling. It is _crucial_ that no exception crates be dependencies
/// of the Rust runtime (std/test).
#[rustfmt::skip]
const EXCEPTIONS_STDLIB: ExceptionList = &[
    // tidy-alphabetical-start
    ("fortanix-sgx-abi", "MPL-2.0"), // libstd but only for `sgx` target. FIXME: this dependency violates the documentation comment above.
    // tidy-alphabetical-end
];

const EXCEPTIONS_CARGO: ExceptionList = &[
    // tidy-alphabetical-start
    ("arrayref", "BSD-2-Clause"),
    ("bitmaps", "MPL-2.0+"),
    ("blake3", "CC0-1.0 OR Apache-2.0 OR Apache-2.0 WITH LLVM-exception"),
    ("ciborium", "Apache-2.0"),
    ("ciborium-io", "Apache-2.0"),
    ("ciborium-ll", "Apache-2.0"),
    ("constant_time_eq", "CC0-1.0 OR MIT-0 OR Apache-2.0"),
    ("dunce", "CC0-1.0 OR MIT-0 OR Apache-2.0"),
    ("encoding_rs", "(Apache-2.0 OR MIT) AND BSD-3-Clause"),
    ("fiat-crypto", "MIT OR Apache-2.0 OR BSD-1-Clause"),
    ("foldhash", "Zlib"),
    ("im-rc", "MPL-2.0+"),
    ("libz-rs-sys", "Zlib"),
    ("normalize-line-endings", "Apache-2.0"),
    ("openssl", "Apache-2.0"),
    ("ryu", "Apache-2.0 OR BSL-1.0"), // BSL is not acceptble, but we use it under Apache-2.0
    ("similar", "Apache-2.0"),
    ("sized-chunks", "MPL-2.0+"),
    ("subtle", "BSD-3-Clause"),
    ("supports-hyperlinks", "Apache-2.0"),
    ("unicode-bom", "Apache-2.0"),
    ("zlib-rs", "Zlib"),
    // tidy-alphabetical-end
];

const EXCEPTIONS_RUST_ANALYZER: ExceptionList = &[
    // tidy-alphabetical-start
    ("dissimilar", "Apache-2.0"),
    ("foldhash", "Zlib"),
    ("notify", "CC0-1.0"),
    ("option-ext", "MPL-2.0"),
    ("pulldown-cmark-to-cmark", "Apache-2.0"),
    ("rustc_apfloat", "Apache-2.0 WITH LLVM-exception"),
    ("ryu", "Apache-2.0 OR BSL-1.0"), // BSL is not acceptble, but we use it under Apache-2.0
    ("scip", "Apache-2.0"),
    // tidy-alphabetical-end
];

const EXCEPTIONS_RUSTC_PERF: ExceptionList = &[
    // tidy-alphabetical-start
    ("alloc-no-stdlib", "BSD-3-Clause"),
    ("alloc-stdlib", "BSD-3-Clause"),
    ("brotli", "BSD-3-Clause/MIT"),
    ("brotli-decompressor", "BSD-3-Clause/MIT"),
    ("encoding_rs", "(Apache-2.0 OR MIT) AND BSD-3-Clause"),
    ("inferno", "CDDL-1.0"),
    ("ring", NON_STANDARD_LICENSE), // see EXCEPTIONS_NON_STANDARD_LICENSE_DEPS for more.
    ("ryu", "Apache-2.0 OR BSL-1.0"),
    ("snap", "BSD-3-Clause"),
    ("subtle", "BSD-3-Clause"),
    // tidy-alphabetical-end
];

const EXCEPTIONS_RUSTBOOK: ExceptionList = &[
    // tidy-alphabetical-start
    ("cssparser", "MPL-2.0"),
    ("cssparser-macros", "MPL-2.0"),
    ("dtoa-short", "MPL-2.0"),
    ("mdbook", "MPL-2.0"),
    ("ryu", "Apache-2.0 OR BSL-1.0"),
    // tidy-alphabetical-end
];

const EXCEPTIONS_CRANELIFT: ExceptionList = &[
    // tidy-alphabetical-start
    ("cranelift-assembler-x64", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-assembler-x64-meta", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-bforest", "Apache-2.0 WITH LLVM-exception"),
    ("cranelift-bitset", "Apache-2.0 WITH LLVM-exception"),
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
    ("cranelift-srcgen", "Apache-2.0 WITH LLVM-exception"),
    ("foldhash", "Zlib"),
    ("mach2", "BSD-2-Clause OR MIT OR Apache-2.0"),
    ("regalloc2", "Apache-2.0 WITH LLVM-exception"),
    ("target-lexicon", "Apache-2.0 WITH LLVM-exception"),
    ("wasmtime-jit-icache-coherence", "Apache-2.0 WITH LLVM-exception"),
    ("wasmtime-math", "Apache-2.0 WITH LLVM-exception"),
    // tidy-alphabetical-end
];

const EXCEPTIONS_GCC: ExceptionList = &[
    // tidy-alphabetical-start
    ("gccjit", "GPL-3.0"),
    ("gccjit_sys", "GPL-3.0"),
    // tidy-alphabetical-end
];

const EXCEPTIONS_BOOTSTRAP: ExceptionList = &[
    ("ryu", "Apache-2.0 OR BSL-1.0"), // through serde. BSL is not acceptble, but we use it under Apache-2.0
];

const EXCEPTIONS_UEFI_QEMU_TEST: ExceptionList = &[
    ("r-efi", "MIT OR Apache-2.0 OR LGPL-2.1-or-later"), // LGPL is not acceptable, but we use it under MIT OR Apache-2.0
];

/// Placeholder for non-standard license file.
const NON_STANDARD_LICENSE: &str = "NON_STANDARD_LICENSE";

/// These dependencies have non-standard licenses but are genenrally permitted.
const EXCEPTIONS_NON_STANDARD_LICENSE_DEPS: &[&str] = &[
    // `ring` is included because it is an optional dependency of `hyper`,
    // which is a training data in rustc-perf for optimized build.
    // The license of it is generally `ISC AND MIT AND OpenSSL`,
    // though the `package.license` field is not set.
    //
    // See https://github.com/briansmith/ring/issues/902
    "ring",
];

const PERMITTED_DEPS_LOCATION: &str = concat!(file!(), ":", line!());

/// Crates rustc is allowed to depend on. Avoid adding to the list if possible.
///
/// This list is here to provide a speed-bump to adding a new dependency to
/// rustc. Please check with the compiler team before adding an entry.
const PERMITTED_RUSTC_DEPENDENCIES: &[&str] = &[
    // tidy-alphabetical-start
    "adler2",
    "aho-corasick",
    "allocator-api2", // FIXME: only appears in Cargo.lock due to https://github.com/rust-lang/cargo/issues/10801
    "annotate-snippets",
    "anstyle",
    "ar_archive_writer",
    "arrayref",
    "arrayvec",
    "autocfg",
    "bitflags",
    "blake3",
    "block-buffer",
    "bstr",
    "cc",
    "cfg-if",
    "cfg_aliases",
    "constant_time_eq",
    "cpufeatures",
    "crc32fast",
    "crossbeam-deque",
    "crossbeam-epoch",
    "crossbeam-utils",
    "crypto-common",
    "ctrlc",
    "darling",
    "darling_core",
    "darling_macro",
    "datafrog",
    "derive-where",
    "derive_setters",
    "digest",
    "displaydoc",
    "dissimilar",
    "either",
    "elsa",
    "ena",
    "equivalent",
    "errno",
    "expect-test",
    "fallible-iterator", // dependency of `thorin`
    "fastrand",
    "flate2",
    "fluent-bundle",
    "fluent-langneg",
    "fluent-syntax",
    "fnv",
    "foldhash",
    "generic-array",
    "getopts",
    "getrandom",
    "gimli",
    "gsgdt",
    "hashbrown",
    "icu_list",
    "icu_list_data",
    "icu_locid",
    "icu_locid_transform",
    "icu_locid_transform_data",
    "icu_provider",
    "icu_provider_adapters",
    "icu_provider_macros",
    "ident_case",
    "indexmap",
    "intl-memoizer",
    "intl_pluralrules",
    "itertools",
    "itoa",
    "jiff",
    "jiff-static",
    "jobserver",
    "lazy_static",
    "leb128",
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
    "miniz_oxide",
    "nix",
    "nu-ansi-term",
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
    "portable-atomic", // dependency for platforms doesn't support `AtomicU64` in std
    "portable-atomic-util",
    "ppv-lite86",
    "proc-macro-hack",
    "proc-macro2",
    "psm",
    "pulldown-cmark",
    "pulldown-cmark-escape",
    "punycode",
    "quote",
    "r-efi",
    "rand",
    "rand_chacha",
    "rand_core",
    "rand_xorshift", // dependency for doc-tests in rustc_thread_pool
    "rand_xoshiro",
    "redox_syscall",
    "regex",
    "regex-automata",
    "regex-syntax",
    "rustc-demangle",
    "rustc-hash",
    "rustc-literal-escaper",
    "rustc-stable-hash",
    "rustc_apfloat",
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
    "shlex",
    "smallvec",
    "stable_deref_trait",
    "stacker",
    "static_assertions",
    "strsim",
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
    "tikv-jemalloc-sys",
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
    "unic-langid",
    "unic-langid-impl",
    "unic-langid-macros",
    "unic-langid-macros-impl",
    "unicase",
    "unicode-ident",
    "unicode-normalization",
    "unicode-properties",
    "unicode-script",
    "unicode-security",
    "unicode-width",
    "unicode-xid",
    "valuable",
    "version_check",
    "wasi",
    "wasm-encoder",
    "wasmparser",
    "winapi",
    "winapi-i686-pc-windows-gnu",
    "winapi-util",
    "winapi-x86_64-pc-windows-gnu",
    "windows",
    "windows-collections",
    "windows-core",
    "windows-future",
    "windows-implement",
    "windows-interface",
    "windows-link",
    "windows-numerics",
    "windows-result",
    "windows-strings",
    "windows-sys",
    "windows-targets",
    "windows-threading",
    "windows_aarch64_gnullvm",
    "windows_aarch64_msvc",
    "windows_i686_gnu",
    "windows_i686_gnullvm",
    "windows_i686_msvc",
    "windows_x86_64_gnu",
    "windows_x86_64_gnullvm",
    "windows_x86_64_msvc",
    "wit-bindgen-rt@0.39.0", // pinned to a specific version due to using a binary blob: <https://github.com/rust-lang/rust/pull/136395#issuecomment-2692769062>
    "writeable",
    "yoke",
    "yoke-derive",
    "zerocopy",
    "zerocopy-derive",
    "zerofrom",
    "zerofrom-derive",
    "zerovec",
    "zerovec-derive",
    // tidy-alphabetical-end
];

const PERMITTED_STDLIB_DEPENDENCIES: &[&str] = &[
    // tidy-alphabetical-start
    "addr2line",
    "adler2",
    "cc",
    "cfg-if",
    "compiler_builtins",
    "dlmalloc",
    "fortanix-sgx-abi",
    "getopts",
    "gimli",
    "hashbrown",
    "hermit-abi",
    "libc",
    "memchr",
    "miniz_oxide",
    "object",
    "r-efi",
    "r-efi-alloc",
    "rand",
    "rand_core",
    "rand_xorshift",
    "rustc-demangle",
    "rustc-literal-escaper",
    "shlex",
    "unicode-width",
    "unwinding",
    "wasi",
    "windows-sys",
    "windows-targets",
    "windows_aarch64_gnullvm",
    "windows_aarch64_msvc",
    "windows_i686_gnu",
    "windows_i686_gnullvm",
    "windows_i686_msvc",
    "windows_x86_64_gnu",
    "windows_x86_64_gnullvm",
    "windows_x86_64_msvc",
    // tidy-alphabetical-end
];

const PERMITTED_CRANELIFT_DEPENDENCIES: &[&str] = &[
    // tidy-alphabetical-start
    "allocator-api2",
    "anyhow",
    "arbitrary",
    "bitflags",
    "bumpalo",
    "cfg-if",
    "cranelift-assembler-x64",
    "cranelift-assembler-x64-meta",
    "cranelift-bforest",
    "cranelift-bitset",
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
    "cranelift-srcgen",
    "crc32fast",
    "equivalent",
    "fallible-iterator",
    "foldhash",
    "gimli",
    "hashbrown",
    "indexmap",
    "libc",
    "libloading",
    "libm",
    "log",
    "mach2",
    "memchr",
    "object",
    "proc-macro2",
    "quote",
    "regalloc2",
    "region",
    "rustc-hash",
    "serde",
    "serde_derive",
    "smallvec",
    "stable_deref_trait",
    "syn",
    "target-lexicon",
    "unicode-ident",
    "wasmtime-jit-icache-coherence",
    "wasmtime-math",
    "windows-sys",
    "windows-targets",
    "windows_aarch64_gnullvm",
    "windows_aarch64_msvc",
    "windows_i686_gnu",
    "windows_i686_gnullvm",
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
pub fn check(root: &Path, cargo: &Path, bless: bool, bad: &mut bool) {
    let mut checked_runtime_licenses = false;

    check_proc_macro_dep_list(root, cargo, bless, bad);

    for &(workspace, exceptions, permitted_deps, submodules) in WORKSPACES {
        if has_missing_submodule(root, submodules) {
            continue;
        }

        if !root.join(workspace).join("Cargo.lock").exists() {
            tidy_error!(bad, "the `{workspace}` workspace doesn't have a Cargo.lock");
            continue;
        }

        let mut cmd = cargo_metadata::MetadataCommand::new();
        cmd.cargo_path(cargo)
            .manifest_path(root.join(workspace).join("Cargo.toml"))
            .features(cargo_metadata::CargoOpt::AllFeatures)
            .other_options(vec!["--locked".to_owned()]);
        let metadata = t!(cmd.exec());

        check_license_exceptions(&metadata, exceptions, bad);
        if let Some((crates, permitted_deps)) = permitted_deps {
            check_permitted_dependencies(&metadata, workspace, permitted_deps, crates, bad);
        }

        if workspace == "library" {
            check_runtime_license_exceptions(&metadata, bad);
            checked_runtime_licenses = true;
        }
    }

    // Sanity check to ensure we don't accidentally remove the workspace containing the runtime
    // crates.
    assert!(checked_runtime_licenses);
}

/// Ensure the list of proc-macro crate transitive dependencies is up to date
fn check_proc_macro_dep_list(root: &Path, cargo: &Path, bless: bool, bad: &mut bool) {
    let mut cmd = cargo_metadata::MetadataCommand::new();
    cmd.cargo_path(cargo)
        .manifest_path(root.join("Cargo.toml"))
        .features(cargo_metadata::CargoOpt::AllFeatures)
        .other_options(vec!["--locked".to_owned()]);
    let metadata = t!(cmd.exec());
    let is_proc_macro_pkg = |pkg: &Package| pkg.targets.iter().any(|target| target.is_proc_macro());

    let mut proc_macro_deps = HashSet::new();
    for pkg in metadata.packages.iter().filter(|pkg| is_proc_macro_pkg(*pkg)) {
        deps_of(&metadata, &pkg.id, &mut proc_macro_deps);
    }
    // Remove the proc-macro crates themselves
    proc_macro_deps.retain(|pkg| !is_proc_macro_pkg(&metadata[pkg]));

    let proc_macro_deps: HashSet<_> =
        proc_macro_deps.into_iter().map(|dep| metadata[dep].name.clone()).collect();
    let expected = proc_macro_deps::CRATES.iter().map(|s| s.to_string()).collect::<HashSet<_>>();

    let needs_blessing = proc_macro_deps.difference(&expected).next().is_some()
        || expected.difference(&proc_macro_deps).next().is_some();

    if needs_blessing && bless {
        let mut proc_macro_deps: Vec<_> = proc_macro_deps.into_iter().collect();
        proc_macro_deps.sort();
        let mut file = File::create(root.join("src/bootstrap/src/utils/proc_macro_deps.rs"))
            .expect("`proc_macro_deps` should exist");
        writeln!(
            &mut file,
            "/// Do not update manually - use `./x.py test tidy --bless`
/// Holds all direct and indirect dependencies of proc-macro crates in tree.
/// See <https://github.com/rust-lang/rust/issues/134863>
pub static CRATES: &[&str] = &[
    // tidy-alphabetical-start"
        )
        .unwrap();
        for dep in proc_macro_deps {
            writeln!(&mut file, "    {dep:?},").unwrap();
        }
        writeln!(
            &mut file,
            "    // tidy-alphabetical-end
];"
        )
        .unwrap();
    } else {
        let old_bad = *bad;

        for missing in proc_macro_deps.difference(&expected) {
            tidy_error!(
                bad,
                "proc-macro crate dependency `{missing}` is not registered in `src/bootstrap/src/utils/proc_macro_deps.rs`",
            );
        }
        for extra in expected.difference(&proc_macro_deps) {
            tidy_error!(
                bad,
                "`{extra}` is registered in `src/bootstrap/src/utils/proc_macro_deps.rs`, but is not a proc-macro crate dependency",
            );
        }
        if *bad != old_bad {
            eprintln!("Run `./x.py test tidy --bless` to regenerate the list");
        }
    }
}

/// Used to skip a check if a submodule is not checked out, and not in a CI environment.
///
/// This helps prevent enforcing developers to fetch submodules for tidy.
pub fn has_missing_submodule(root: &Path, submodules: &[&str]) -> bool {
    !CiEnv::is_ci()
        && submodules.iter().any(|submodule| {
            let path = root.join(submodule);
            !path.exists()
            // If the directory is empty, we can consider it as an uninitialized submodule.
            || read_dir(path).unwrap().next().is_none()
        })
}

/// Check that all licenses of runtime dependencies are in the valid list in `LICENSES`.
///
/// Unlike for tools we don't allow exceptions to the `LICENSES` list for the runtime with the sole
/// exception of `fortanix-sgx-abi` which is only used on x86_64-fortanix-unknown-sgx.
fn check_runtime_license_exceptions(metadata: &Metadata, bad: &mut bool) {
    for pkg in &metadata.packages {
        if pkg.source.is_none() {
            // No need to check local packages.
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
            // This is a specific exception because SGX is considered "third party".
            // See https://github.com/rust-lang/rust/issues/62620 for more.
            // In general, these should never be added and this exception
            // should not be taken as precedent for any new target.
            if pkg.name == "fortanix-sgx-abi" && pkg.license.as_deref() == Some("MPL-2.0") {
                continue;
            }

            tidy_error!(bad, "invalid license `{}` in `{}`", license, pkg.id);
        }
    }
}

/// Check that all licenses of tool dependencies are in the valid list in `LICENSES`.
///
/// Packages listed in `exceptions` are allowed for tools.
fn check_license_exceptions(metadata: &Metadata, exceptions: &[(&str, &str)], bad: &mut bool) {
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
                    if *license == NON_STANDARD_LICENSE
                        && EXCEPTIONS_NON_STANDARD_LICENSE_DEPS.contains(&pkg.name.as_str())
                    {
                        continue;
                    }
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
        if exception_names.contains(&pkg.name.as_str()) {
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
        deps_of(metadata, &to_check.id, &mut deps);
    }

    // Check that the PERMITTED_DEPENDENCIES does not have unused entries.
    for permitted in permitted_dependencies {
        fn compare(pkg: &Package, permitted: &str) -> bool {
            if let Some((name, version)) = permitted.split_once("@") {
                let Ok(version) = Version::parse(version) else {
                    return false;
                };
                pkg.name == name && pkg.version == version
            } else {
                pkg.name == permitted
            }
        }
        if !deps.iter().any(|dep_id| compare(pkg_from_id(metadata, dep_id), permitted)) {
            tidy_error!(
                bad,
                "could not find allowed package `{permitted}`\n\
                Remove from PERMITTED_DEPENDENCIES list if it is no longer used.",
            );
            has_permitted_dep_error = true;
        }
    }

    // Get in a convenient form.
    let permitted_dependencies: HashMap<_, _> = permitted_dependencies
        .iter()
        .map(|s| {
            if let Some((name, version)) = s.split_once('@') {
                (name, Version::parse(version).ok())
            } else {
                (*s, None)
            }
        })
        .collect();

    for dep in deps {
        let dep = pkg_from_id(metadata, dep);
        // If this path is in-tree, we don't require it to be explicitly permitted.
        if dep.source.is_some() {
            let is_eq = if let Some(version) = permitted_dependencies.get(dep.name.as_str()) {
                if let Some(version) = version { version == &dep.version } else { true }
            } else {
                false
            };
            if !is_eq {
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

/// Recursively find all dependencies.
fn deps_of<'a>(metadata: &'a Metadata, pkg_id: &'a PackageId, result: &mut HashSet<&'a PackageId>) {
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
        deps_of(metadata, &dep.pkg, result);
    }
}
