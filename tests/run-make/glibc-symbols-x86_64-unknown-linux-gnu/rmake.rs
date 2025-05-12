// Check that the compiler toolchain (rustc) that we distribute is not using newer glibc
// symbols than a specified minimum.
// This test should only be executed on an extracted dist archive or in a dist-* CI job.

//@ only-dist
//@ only-x86_64-unknown-linux-gnu
//@ ignore-cross-compile

use std::path::{Path, PathBuf};

use run_make_support::{cmd, llvm_objdump, regex, rustc_path};

fn main() {
    // This is the maximum glibc version that we are *permitted* to use for the
    // x86_64-unknown-linux-gnu target.
    // All glibc symbols used in the compiler must be lower or equal than this version.
    // So that if a given machine only has glibc 2.17, it is able to run the compiler.
    let max_supported = (2, 17, 99);

    let rustc = PathBuf::from(rustc_path());
    // Check symbols directly in rustc
    check_symbols(&rustc, max_supported);

    // Find dynamic libraries referenced by rustc that come from our lib directory
    let lib_path = rustc.parent().unwrap().parent().unwrap().join("lib");
    let dynamic_libs = find_dynamic_libs(&rustc)
        .into_iter()
        .filter_map(|path| path.canonicalize().ok())
        .filter(|lib| lib.starts_with(&lib_path))
        .collect::<Vec<_>>();
    for lib in dynamic_libs {
        check_symbols(&lib, max_supported);
    }
}

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq)]
struct GlibcSymbol {
    name: String,
    version: (u32, u32, u32),
}

fn find_dynamic_libs(path: &Path) -> Vec<PathBuf> {
    cmd("ldd")
        .arg(path)
        .run()
        .stdout_utf8()
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let Some((_, line)) = line.split_once(" => ") else {
                return None;
            };
            line.split_ascii_whitespace().next().map(|path| PathBuf::from(path))
        })
        .collect()
}

fn check_symbols(file: &Path, max_supported: (u32, u32, u32)) {
    println!("Checking {}", file.display());
    let mut invalid: Vec<GlibcSymbol> = get_glibc_symbols(file)
        .into_iter()
        .filter(|symbol| symbol.version > max_supported)
        .collect();
    if !invalid.is_empty() {
        invalid.sort();
        panic!(
            "Found invalid glibc symbols in {}:\n{}",
            file.display(),
            invalid
                .into_iter()
                .map(|symbol| format!(
                    "{} ({:?} higher than max allowed {:?})",
                    symbol.name, symbol.version, max_supported
                ))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }
}

fn get_glibc_symbols(file: &Path) -> Vec<GlibcSymbol> {
    let regex = regex::Regex::new(r#"GLIBC_(\d)+\.(\d+)(:?\.(\d+))?"#).unwrap();

    // FIXME(kobzol): llvm-objdump currently chokes on the BOLTed librustc_driver.so file.
    // Use objdump instead, since it seems to work, and we only run this test in a specific
    // CI environment anyway.
    cmd("objdump")
        .arg("--dynamic-syms")
        .arg(file)
        .run()
        .stdout_utf8()
        .lines()
        .filter_map(|line| {
            // Example line
            // 0000000000000000 DF *UND* 0000000000000000 (GLIBC_2.2.5) sbrk
            let mut parts = line.split(" ").collect::<Vec<_>>().into_iter().rev();
            let Some(name) = parts.next() else {
                return None;
            };
            let Some(lib) = parts.next() else {
                return None;
            };
            let Some(version) = regex.captures(lib) else {
                return None;
            };
            let major = version.get(1).and_then(|m| m.as_str().parse().ok()).unwrap_or(0);
            let minor = version.get(2).and_then(|m| m.as_str().parse().ok()).unwrap_or(0);
            let patch = version.get(3).and_then(|m| m.as_str().parse().ok()).unwrap_or(0);
            Some(GlibcSymbol { version: (major, minor, patch), name: name.to_string() })
        })
        .collect()
}
