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

// These are exceptions to Rust's permissive licensing policy, and
// should be considered bugs. Exceptions are only allowed in Rust
// tooling. It is _crucial_ that no exception crates be dependencies
// of the Rust runtime (std / test).
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

// Whitelist of crates rustc is allowed to depend on. Avoid adding to the list if possible.
static WHITELIST: &'static [(&'static str, &'static str)] = &[
//    ("advapi32-sys", "0.2.0"),
//    ("aho-corasick", "0.5.3"),
//    ("aho-corasick", "0.6.4"),
//    ("alloc", "0.0.0"),
//    ("alloc_jemalloc", "0.0.0"),
//    ("alloc_system", "0.0.0"),
//    ("ansi_term", "0.10.2"),
//    ("ar", "0.3.1"),
//    ("arena", "0.0.0"),
//    ("atty", "0.2.6"),
//    ("backtrace", "0.3.5"),
//    ("backtrace-sys", "0.1.16"),
//    ("bin_lib", "0.1.0"),
//    ("bitflags", "0.7.0"),
//    ("bitflags", "0.9.1"),
//    ("bitflags", "1.0.1"),
//    ("bootstrap", "0.0.0"),
//    ("borrow_error", "0.1.0"),
//    ("bufstream", "0.1.3"),
//    ("build-manifest", "0.1.0"),
//    ("build_helper", "0.1.0"),
//    ("byteorder", "1.2.1"),
//    ("cargo", "0.26.0"),
//    ("cargo_metadata", "0.2.3"),
//    ("cargo_metadata", "0.4.0"),
//    ("cargotest", "0.1.0"),
//    ("cargotest2", "0.1.0"),
//    ("cc", "1.0.4"),
//    ("cfg-if", "0.1.2"),
//    ("chrono", "0.4.0"),
//    ("clap", "2.29.0"),
//    ("clippy", "0.0.186"),
//    ("clippy-mini-macro-test", "0.2.0"),
//    ("clippy_lints", "0.0.186"),
//    ("cmake", "0.1.29"),
//    ("coco", "0.1.1"),
//    ("commoncrypto", "0.2.0"),
//    ("commoncrypto-sys", "0.2.0"),
//    ("compiler_builtins", "0.0.0"),
//    ("compiletest", "0.0.0"),
//    ("compiletest_rs", "0.3.6"),
//    ("completion", "0.1.0"),
//    ("core", "0.0.0"),
//    ("core-foundation", "0.4.6"),
//    ("core-foundation-sys", "0.4.6"),
//    ("crates-io", "0.15.0"),
//    ("crossbeam", "0.2.12"),
//    ("crossbeam", "0.3.2"),
//    ("crypto-hash", "0.3.0"),
//    ("curl", "0.4.11"),
//    ("curl-sys", "0.4.1"),
//    ("deglob", "0.1.0"),
//    ("derive-new", "0.5.0"),
//    ("diff", "0.1.11"),
//    ("dlmalloc", "0.0.0"),
//    ("docopt", "0.8.3"),
//    ("dtoa", "0.4.2"),
//    ("duct", "0.8.2"),
//    ("either", "1.4.0"),
//    ("endian-type", "0.1.2"),
//    ("enum_primitive", "0.1.1"),
//    ("env_logger", "0.3.5"),
//    ("env_logger", "0.4.3"),
//    ("env_logger", "0.5.3"),
//    ("error-chain", "0.11.0"),
//    ("error-chain", "0.8.1"),
//    ("error_index_generator", "0.0.0"),
//    ("failure", "0.1.1"),
//    ("failure_derive", "0.1.1"),
//    ("features", "0.1.0"),
//    ("filetime", "0.1.15"),
//    ("find_all_refs_no_cfg_test", "0.1.0"),
//    ("find_impls", "0.1.0"),
//    ("flate2", "1.0.1"),
//    ("fmt_macros", "0.0.0"),
//    ("fnv", "1.0.6"),
//    ("foreign-types", "0.3.2"),
//    ("foreign-types-shared", "0.1.1"),
//    ("fs2", "0.4.3"),
//    ("fuchsia-zircon", "0.3.3"),
//    ("fuchsia-zircon-sys", "0.3.3"),
//    ("futures", "0.1.17"),
//    ("getopts", "0.2.15"),
//    ("git2", "0.6.11"),
//    ("git2-curl", "0.7.0"),
//    ("glob", "0.2.11"),
//    ("globset", "0.2.1"),
//    ("graphviz", "0.0.0"),
//    ("hamcrest", "0.1.1"),
//    ("handlebars", "0.29.1"),
//    ("hex", "0.2.0"),
//    ("hex", "0.3.1"),
//    ("home", "0.3.0"),
//    ("idna", "0.1.4"),
//    ("if_chain", "0.1.2"),
//    ("ignore", "0.3.1"),
//    ("infer_bin", "0.1.0"),
//    ("infer_custom_bin", "0.1.0"),
//    ("infer_lib", "0.1.0"),
//    ("installer", "0.0.0"),
//    ("is-match", "0.1.0"),
//    ("itertools", "0.6.5"),
//    ("itertools", "0.7.6"),
//    ("itoa", "0.3.4"),
//    ("jobserver", "0.1.9"),
//    ("json", "0.11.12"),
//    ("jsonrpc-core", "8.0.1"),
//    ("kernel32-sys", "0.2.2"),
//    ("languageserver-types", "0.30.0"),
//    ("lazy_static", "0.2.11"),
//    ("lazy_static", "1.0.0"),
//    ("lazycell", "0.5.1"),
//    ("libc", "0.0.0"),
//    ("libc", "0.2.36"),
//    ("libgit2-sys", "0.6.19"),
//    ("libssh2-sys", "0.2.6"),
//    ("libz-sys", "1.0.18"),
//    ("linkchecker", "0.1.0"),
//    ("log", "0.3.9"),
//    ("log", "0.4.1"),
//    ("log_settings", "0.1.1"),
//    ("lzma-sys", "0.1.9"),
//    ("matches", "0.1.6"),
//    ("mdbook", "0.1.2"),
//    ("memchr", "0.1.11"),
//    ("memchr", "2.0.1"),
//    ("miniz-sys", "0.1.10"),
//    ("miow", "0.2.1"),
//    ("miri", "0.1.0"),
//    ("multiple_bins", "0.1.0"),
//    ("net2", "0.2.31"),
//    ("nibble_vec", "0.0.3"),
//    ("nix", "0.8.1"),
//    ("num", "0.1.41"),
//    ("num-bigint", "0.1.41"),
//    ("num-complex", "0.1.41"),
//    ("num-integer", "0.1.35"),
//    ("num-iter", "0.1.34"),
//    ("num-rational", "0.1.40"),
//    ("num-traits", "0.1.41"),
//    ("num_cpus", "1.8.0"),
//    ("open", "1.2.1"),
//    ("openssl", "0.9.23"),
//    ("openssl-probe", "0.1.2"),
//    ("openssl-sys", "0.9.24"),
//    ("os_pipe", "0.5.1"),
//    ("owning_ref", "0.3.3"),
//    ("panic_abort", "0.0.0"),
//    ("panic_unwind", "0.0.0"),
//    ("parking_lot", "0.5.3"),
//    ("parking_lot_core", "0.2.9"),
//    ("percent-encoding", "1.0.1"),
//    ("pest", "0.3.3"),
//    ("pkg-config", "0.3.9"),
//    ("proc_macro", "0.0.0"),
//    ("profiler_builtins", "0.0.0"),
//    ("pulldown-cmark", "0.0.15"),
//    ("pulldown-cmark", "0.1.0"),
//    ("quick-error", "1.2.1"),
//    ("quine-mc_cluskey", "0.2.4"),
//    ("quote", "0.3.15"),
//    ("racer", "2.0.12"),
//    ("radix_trie", "0.1.2"),
//    ("rand", "0.3.20"),
//    ("rayon", "0.9.0"),
//    ("rayon-core", "1.3.0"),
//    ("redox_syscall", "0.1.37"),
//    ("redox_termios", "0.1.1"),
//    ("reformat", "0.1.0"),
//    ("reformat_with_range", "0.1.0"),
//    ("regex", "0.1.80"),
//    ("regex", "0.2.5"),
//    ("regex-syntax", "0.3.9"),
//    ("regex-syntax", "0.4.2"),
//    ("remote-test-client", "0.1.0"),
//    ("remote-test-server", "0.1.0"),
//    ("rls", "0.125.0"),
//    ("rls-analysis", "0.11.0"),
//    ("rls-blacklist", "0.1.0"),
//    ("rls-data", "0.15.0"),
//    ("rls-rustc", "0.2.1"),
//    ("rls-span", "0.4.0"),
//    ("rls-vfs", "0.4.4"),
//    ("rustbook", "0.1.0"),
//    ("rustc", "0.0.0"),
//    ("rustc-ap-rustc_cratesio_shim", "29.0.0"),
//    ("rustc-ap-rustc_data_structures", "29.0.0"),
//    ("rustc-ap-rustc_errors", "29.0.0"),
//    ("rustc-ap-serialize", "29.0.0"),
//    ("rustc-ap-syntax", "29.0.0"),
//    ("rustc-ap-syntax_pos", "29.0.0"),
//    ("rustc-demangle", "0.1.5"),
//    ("rustc-main", "0.0.0"),
//    ("rustc-serialize", "0.3.24"),
//    ("rustc_allocator", "0.0.0"),
//    ("rustc_apfloat", "0.0.0"),
//    ("rustc_asan", "0.0.0"),
//    ("rustc_back", "0.0.0"),
//    ("rustc_binaryen", "0.0.0"),
//    ("rustc_borrowck", "0.0.0"),
//    ("rustc_const_eval", "0.0.0"),
//    ("rustc_const_math", "0.0.0"),
//    ("rustc_cratesio_shim", "0.0.0"),
//    ("rustc_data_structures", "0.0.0"),
//    ("rustc_driver", "0.0.0"),
//    ("rustc_errors", "0.0.0"),
//    ("rustc_incremental", "0.0.0"),
//    ("rustc_lint", "0.0.0"),
//    ("rustc_llvm", "0.0.0"),
//    ("rustc_lsan", "0.0.0"),
//    ("rustc_metadata", "0.0.0"),
//    ("rustc_mir", "0.0.0"),
//    ("rustc_msan", "0.0.0"),
//    ("rustc_passes", "0.0.0"),
//    ("rustc_platform_intrinsics", "0.0.0"),
//    ("rustc_plugin", "0.0.0"),
//    ("rustc_privacy", "0.0.0"),
//    ("rustc_resolve", "0.0.0"),
//    ("rustc_save_analysis", "0.0.0"),
//    ("rustc_trans", "0.0.0"),
//    ("rustc_trans_utils", "0.0.0"),
//    ("rustc_tsan", "0.0.0"),
//    ("rustc_typeck", "0.0.0"),
//    ("rustdoc", "0.0.0"),
//    ("rustdoc-themes", "0.1.0"),
//    ("rustdoc-tool", "0.0.0"),
//    ("rustfmt-nightly", "0.3.8"),
//    ("same-file", "0.1.3"),
//    ("same-file", "1.0.2"),
//    ("schannel", "0.1.10"),
//    ("scoped-tls", "0.1.0"),
//    ("scopeguard", "0.1.2"),
//    ("scopeguard", "0.3.3"),
//    ("semver", "0.6.0"),
//    ("semver", "0.8.0"),
//    ("semver", "0.9.0"),
//    ("semver-parser", "0.7.0"),
//    ("serde", "1.0.27"),
//    ("serde_derive", "1.0.27"),
//    ("serde_derive_internals", "0.19.0"),
//    ("serde_ignored", "0.0.4"),
//    ("serde_json", "1.0.9"),
//    ("serialize", "0.0.0"),
//    ("shared_child", "0.2.1"),
//    ("shell-escape", "0.1.3"),
//    ("shlex", "0.1.1"),
//    ("smallvec", "0.6.0"),
//    ("socket2", "0.3.0"),
//    ("stable_deref_trait", "1.0.0"),
//    ("std", "0.0.0"),
//    ("std_unicode", "0.0.0"),
//    ("strsim", "0.6.0"),
//    ("syn", "0.11.11"),
//    ("synom", "0.11.3"),
//    ("synstructure", "0.6.1"),
//    ("syntax", "0.0.0"),
//    ("syntax_ext", "0.0.0"),
//    ("syntax_pos", "0.0.0"),
//    ("syntex_errors", "0.52.0"),
//    ("syntex_pos", "0.52.0"),
//    ("syntex_syntax", "0.52.0"),
//    ("tar", "0.4.14"),
//    ("tempdir", "0.3.5"),
//    ("term", "0.0.0"),
//    ("term", "0.4.6"),
//    ("termcolor", "0.3.3"),
//    ("termion", "1.5.1"),
//    ("test", "0.0.0"),
//    ("textwrap", "0.9.0"),
//    ("thread-id", "2.0.0"),
//    ("thread_local", "0.2.7"),
//    ("thread_local", "0.3.5"),
//    ("tidy", "0.1.0"),
//    ("time", "0.1.39"),
//    ("toml", "0.2.1"),
//    ("toml", "0.4.5"),
//    ("toml-query", "0.6.0"),
//    ("unicode-bidi", "0.3.4"),
//    ("unicode-normalization", "0.1.5"),
//    ("unicode-segmentation", "1.2.0"),
//    ("unicode-width", "0.1.4"),
//    ("unicode-xid", "0.0.3"),
//    ("unicode-xid", "0.0.4"),
//    ("unreachable", "1.0.0"),
//    ("unstable-book-gen", "0.1.0"),
//    ("unwind", "0.0.0"),
//    ("url", "1.6.0"),
//    ("url_serde", "0.2.0"),
//    ("userenv-sys", "0.2.0"),
//    ("utf8-ranges", "0.1.3"),
//    ("utf8-ranges", "1.0.0"),
//    ("vcpkg", "0.2.2"),
//    ("vec_map", "0.8.0"),
//    ("void", "1.0.2"),
//    ("walkdir", "1.0.7"),
//    ("walkdir", "2.0.1"),
//    ("winapi", "0.2.8"),
//    ("winapi", "0.3.4"),
//    ("winapi-build", "0.1.1"),
//    ("winapi-i686-pc-windows-gnu", "0.4.0"),
//    ("winapi-x86_64-pc-windows-gnu", "0.4.0"),
//    ("wincolor", "0.1.4"),
//    ("workspace_symbol", "0.1.0"),
//    ("ws2_32-sys", "0.2.1"),
//    ("xattr", "0.1.11"),
//    ("xz2", "0.1.4"),
//    ("yaml-rust", "0.3.5"),
];

// Some types for Serde to deserialize the output of `cargo metadata` to...

#[derive(Deserialize)]
struct Output {
    packages: Vec<Package>,

    // Not used, but needed to not confuse serde :P
    #[allow(dead_code)] resolve: Resolve,
}

#[derive(Deserialize)]
struct Package {
    name: String,
    version: String,

    // Not used, but needed to not confuse serde :P
    #[allow(dead_code)] id: String,
    #[allow(dead_code)] source: Option<String>,
    #[allow(dead_code)] manifest_path: String,
}

// Not used, but needed to not confuse serde :P
#[allow(dead_code)]
#[derive(Deserialize)]
struct Resolve {
    nodes: Vec<ResolveNode>,
}

// Not used, but needed to not confuse serde :P
#[allow(dead_code)]
#[derive(Deserialize)]
struct ResolveNode {
    id: String,
    dependencies: Vec<String>,
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

/// Checks the dependency at the given path. Changes `bad` to `true` if a check failed.
///
/// Specifically, this checks that the dependencies are on the whitelist.
pub fn check_whitelist(path: &Path, bad: &mut bool) {
    // Check dependencies
    let deps: HashSet<_> = get_deps(&path)
        .into_iter()
        .map(|Package { name, version, .. }| (name, version))
        .collect();
    let whitelist: HashSet<(String, String)> = WHITELIST
        .iter()
        .map(|&(n, v)| (n.to_owned(), v.to_owned()))
        .collect();

    // Dependencies not in the whitelist
    let mut unapproved: Vec<_> = deps.difference(&whitelist).collect();

    // For ease of reading
    unapproved.sort();

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
fn get_deps(path: &Path) -> Vec<Package> {
    // Run `cargo metadata` to get the set of dependencies
    println!("Getting metadata from {:?}", path.join("Cargo.toml"));
    let output = Command::new("cargo")
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

    output.packages
}
