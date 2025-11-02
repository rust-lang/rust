//! Tool used by CI to inspect compiler-builtins archives and help ensure we won't run into any
//! linking errors.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use object::read::archive::{ArchiveFile, ArchiveMember};
use object::{
    File as ObjFile, Object, ObjectSection, ObjectSymbol, Symbol, SymbolKind, SymbolScope,
};
use serde_json::Value;

const CHECK_LIBRARIES: &[&str] = &["compiler_builtins", "builtins_test_intrinsics"];
const CHECK_EXTENSIONS: &[Option<&str>] = &[Some("rlib"), Some("a"), Some("exe"), None];

const USAGE: &str = "Usage:

    symbol-check build-and-check [TARGET] -- CARGO_BUILD_ARGS ...

Cargo will get invoked with `CARGO_ARGS` and the specified target. All output
`compiler_builtins*.rlib` files will be checked.

If TARGET is not specified, the host target is used.
";

fn main() {
    // Create a `&str` vec so we can match on it.
    let args = std::env::args().collect::<Vec<_>>();
    let args_ref = args.iter().map(String::as_str).collect::<Vec<_>>();

    match &args_ref[1..] {
        ["build-and-check", target, "--", args @ ..] if !args.is_empty() => {
            check_cargo_args(args);
            run_build_and_check(target, args);
        }
        ["build-and-check", "--", args @ ..] if !args.is_empty() => {
            check_cargo_args(args);
            run_build_and_check(&host_target(), args);
        }
        _ => {
            println!("{USAGE}");
            std::process::exit(1);
        }
    }
}

/// Make sure `--target` isn't passed to avoid confusion (since it should be proivded only once,
/// positionally).
fn check_cargo_args(args: &[&str]) {
    for arg in args {
        assert!(
            !arg.contains("--target"),
            "target must be passed positionally. {USAGE}"
        );
    }
}

fn run_build_and_check(target: &str, args: &[&str]) {
    let paths = exec_cargo_with_args(target, args);
    for path in paths {
        println!("Checking {}", path.display());
        let archive = Archive::from_path(&path);

        verify_no_duplicates(&archive);
        verify_core_symbols(&archive);
    }
}

fn host_target() -> String {
    let out = Command::new("rustc")
        .arg("--version")
        .arg("--verbose")
        .output()
        .unwrap();
    assert!(out.status.success());
    let out = String::from_utf8(out.stdout).unwrap();
    out.lines()
        .find_map(|s| s.strip_prefix("host: "))
        .unwrap()
        .to_owned()
}

/// Run `cargo build` with the provided additional arguments, collecting the list of created
/// libraries.
fn exec_cargo_with_args(target: &str, args: &[&str]) -> Vec<PathBuf> {
    let mut cmd = Command::new("cargo");
    cmd.args([
        "build",
        "--target",
        target,
        "--message-format=json-diagnostic-rendered-ansi",
    ])
    .args(args)
    .stdout(Stdio::piped());

    println!("running: {cmd:?}");
    let mut child = cmd.spawn().expect("failed to launch Cargo");

    let stdout = child.stdout.take().unwrap();
    let reader = BufReader::new(stdout);
    let mut check_files = Vec::new();

    for line in reader.lines() {
        let line = line.expect("failed to read line");
        let j: Value = serde_json::from_str(&line).expect("failed to deserialize");
        let reason = &j["reason"];

        // Forward output that is meant to be user-facing
        if reason == "compiler-message" {
            println!("{}", j["message"]["rendered"].as_str().unwrap());
        } else if reason == "build-finished" {
            println!("build finshed. success: {}", j["success"]);
        } else if reason == "build-script-executed" {
            let pretty = serde_json::to_string_pretty(&j).unwrap();
            println!("build script output: {pretty}",);
        }

        // Only interested in the artifact list now
        if reason != "compiler-artifact" {
            continue;
        }

        // Find rlibs in the created file list that match our expected library names and
        // extensions.
        for fpath in j["filenames"].as_array().expect("filenames not an array") {
            let path = fpath.as_str().expect("file name not a string");
            let path = PathBuf::from(path);

            if CHECK_EXTENSIONS.contains(&path.extension().map(|ex| ex.to_str().unwrap())) {
                let fname = path.file_name().unwrap().to_str().unwrap();

                if CHECK_LIBRARIES.iter().any(|lib| fname.contains(lib)) {
                    check_files.push(path);
                }
            }
        }
    }

    assert!(child.wait().expect("failed to wait on Cargo").success());

    assert!(!check_files.is_empty(), "no compiler_builtins rlibs found");
    println!("Collected the following rlibs to check: {check_files:#?}");

    check_files
}

/// Information collected from `object`, for convenience.
#[expect(unused)] // only for printing
#[derive(Clone, Debug)]
struct SymInfo {
    name: String,
    kind: SymbolKind,
    scope: SymbolScope,
    section: String,
    is_undefined: bool,
    is_global: bool,
    is_local: bool,
    is_weak: bool,
    is_common: bool,
    address: u64,
    object: String,
}

impl SymInfo {
    fn new(sym: &Symbol, obj: &ObjFile, member: &ArchiveMember) -> Self {
        // Include the section name if possible. Fall back to the `Section` debug impl if not.
        let section = sym.section();
        let section_name = sym
            .section()
            .index()
            .and_then(|idx| obj.section_by_index(idx).ok())
            .and_then(|sec| sec.name().ok())
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("{section:?}"));

        Self {
            name: sym.name().expect("missing name").to_owned(),
            kind: sym.kind(),
            scope: sym.scope(),
            section: section_name,
            is_undefined: sym.is_undefined(),
            is_global: sym.is_global(),
            is_local: sym.is_local(),
            is_weak: sym.is_weak(),
            is_common: sym.is_common(),
            address: sym.address(),
            object: String::from_utf8_lossy(member.name()).into_owned(),
        }
    }
}

/// Ensure that the same global symbol isn't defined in multiple object files within an archive.
///
/// Note that this will also locate cases where a symbol is weakly defined in more than one place.
/// Technically there are no linker errors that will come from this, but it keeps our binary more
/// straightforward and saves some distribution size.
fn verify_no_duplicates(archive: &Archive) {
    let mut syms = BTreeMap::<String, SymInfo>::new();
    let mut dups = Vec::new();
    let mut found_any = false;

    archive.for_each_symbol(|symbol, obj, member| {
        // Only check defined globals
        if !symbol.is_global() || symbol.is_undefined() {
            return;
        }

        let sym = SymInfo::new(&symbol, obj, member);

        // x86-32 includes multiple copies of thunk symbols
        if sym.name.starts_with("__x86.get_pc_thunk") {
            return;
        }

        // GDB pretty printing symbols may show up more than once but are weak.
        if sym.section == ".debug_gdb_scripts" && sym.is_weak {
            return;
        }

        // Windows has symbols for literal numeric constants, string literals, and MinGW pseudo-
        // relocations. These are allowed to have repeated definitions.
        let win_allowed_dup_pfx = ["__real@", "__xmm@", "__ymm@", "??_C@_", ".refptr"];
        if win_allowed_dup_pfx
            .iter()
            .any(|pfx| sym.name.starts_with(pfx))
        {
            return;
        }

        match syms.get(&sym.name) {
            Some(existing) => {
                dups.push(sym);
                dups.push(existing.clone());
            }
            None => {
                syms.insert(sym.name.clone(), sym);
            }
        }

        found_any = true;
    });

    assert!(found_any, "no symbols found");

    if !dups.is_empty() {
        dups.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        panic!("found duplicate symbols: {dups:#?}");
    }

    println!("    success: no duplicate symbols found");
}

/// Ensure that there are no references to symbols from `core` that aren't also (somehow) defined.
fn verify_core_symbols(archive: &Archive) {
    let mut defined = BTreeSet::new();
    let mut undefined = Vec::new();
    let mut has_symbols = false;

    archive.for_each_symbol(|symbol, obj, member| {
        has_symbols = true;

        // Find only symbols from `core`
        if !symbol.name().unwrap().contains("_ZN4core") {
            return;
        }

        let sym = SymInfo::new(&symbol, obj, member);
        if sym.is_undefined {
            undefined.push(sym);
        } else {
            defined.insert(sym.name);
        }
    });

    assert!(has_symbols, "no symbols found");

    // Discard any symbols that are defined somewhere in the archive
    undefined.retain(|sym| !defined.contains(&sym.name));

    if !undefined.is_empty() {
        undefined.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        panic!("found undefined symbols from core: {undefined:#?}");
    }

    println!("    success: no undefined references to core found");
}

/// Thin wrapper for owning data used by `object`.
struct Archive {
    data: Vec<u8>,
}

impl Archive {
    fn from_path(path: &Path) -> Self {
        Self {
            data: fs::read(path).expect("reading file failed"),
        }
    }

    fn file(&self) -> ArchiveFile<'_> {
        ArchiveFile::parse(self.data.as_slice()).expect("archive parse failed")
    }

    /// For a given archive, do something with each object file.
    fn for_each_object(&self, mut f: impl FnMut(ObjFile, &ArchiveMember)) {
        let archive = self.file();

        for member in archive.members() {
            let member = member.expect("failed to access member");
            let obj_data = member
                .data(self.data.as_slice())
                .expect("failed to access object");
            let obj = ObjFile::parse(obj_data).expect("failed to parse object");
            f(obj, &member);
        }
    }

    /// For a given archive, do something with each symbol.
    fn for_each_symbol(&self, mut f: impl FnMut(Symbol, &ObjFile, &ArchiveMember)) {
        self.for_each_object(|obj, member| {
            obj.symbols().for_each(|sym| f(sym, &obj, member));
        });
    }
}
