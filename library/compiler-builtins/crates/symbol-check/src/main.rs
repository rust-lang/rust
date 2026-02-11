//! Tool used by CI to inspect compiler-builtins archives and help ensure we won't run into any
//! linking errors.
//!
//! Note that symcheck is a "hostprog", i.e. is built and run on the host target even when the
//! actual target is cross compiled.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, exit};
use std::sync::LazyLock;

use object::read::archive::ArchiveFile;
use object::{
    BinaryFormat, File as ObjFile, Object, ObjectSection, ObjectSymbol, Result as ObjResult,
    Symbol, SymbolKind, SymbolScope,
};
use regex::Regex;
use serde_json::Value;

const CHECK_LIBRARIES: &[&str] = &["compiler_builtins", "builtins_test_intrinsics"];
const CHECK_EXTENSIONS: &[Option<&str>] = &[Some("rlib"), Some("a"), Some("exe"), None];

const USAGE: &str = "Usage:

    symbol-check --build-and-check [--target TARGET] -- CARGO_BUILD_ARGS ...
    symbol-check --check PATHS ...\
";

fn main() {
    let mut opts = getopts::Options::new();

    // Ideally these would be subcommands but that isn't supported.
    opts.optflag("h", "help", "Print this help message");
    opts.optflag(
        "",
        "build-and-check",
        "Cargo will get invoked with `CARGO_BUILD_ARGS` and the specified target. All output \
        `compiler_builtins*.rlib` files will be checked.",
    );
    opts.optopt(
        "",
        "target",
        "Set the target for build-and-check. Falls back to the host target otherwise.",
        "TARGET",
    );
    opts.optflag(
        "",
        "check",
        "Run checks on the given set of paths, without invoking Cargo. Paths \
        may be either archives or object files.",
    );

    let print_usage_and_exit = |code: i32| -> ! {
        eprintln!("{}", opts.usage(USAGE));
        exit(code);
    };

    let m = opts.parse(std::env::args().skip(1)).unwrap_or_else(|e| {
        eprintln!("{e}");
        print_usage_and_exit(1);
    });

    if m.opt_present("help") {
        print_usage_and_exit(0);
    }

    let free_args = m.free.iter().map(String::as_str).collect::<Vec<_>>();
    for arg in &free_args {
        assert!(
            !arg.contains("--target"),
            "target must be passed to symbol-check"
        );
    }

    if m.opt_present("build-and-check") {
        let target = m.opt_str("target").unwrap_or(env!("HOST").to_string());
        let paths = exec_cargo_with_args(&target, &free_args);
        check_paths(&paths);
    } else if m.opt_present("check") {
        if free_args.is_empty() {
            print_usage_and_exit(1);
        }
        check_paths(&free_args);
    } else {
        print_usage_and_exit(1);
    }
}

fn check_paths<P: AsRef<Path>>(paths: &[P]) {
    for path in paths {
        let path = path.as_ref();
        println!("Checking {}", path.display());
        let archive = BinFile::from_path(path);

        verify_no_duplicates(&archive);
        verify_core_symbols(&archive);
    }
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
    fn new(sym: &Symbol, obj: &ObjFile, obj_path: &str) -> Self {
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
            object: obj_path.to_owned(),
        }
    }
}

/// Ensure that the same global symbol isn't defined in multiple object files within an archive.
///
/// Note that this will also locate cases where a symbol is weakly defined in more than one place.
/// Technically there are no linker errors that will come from this, but it keeps our binary more
/// straightforward and saves some distribution size.
fn verify_no_duplicates(archive: &BinFile) {
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

    if archive.has_symbol_tables() {
        assert!(found_any, "no symbols found");
    }

    if !dups.is_empty() {
        let count = dups.iter().map(|x| &x.name).collect::<HashSet<_>>().len();
        dups.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        panic!("found {count} duplicate symbols: {dups:#?}");
    }

    println!("    success: no duplicate symbols found");
}

/// Ensure that there are no references to symbols from `core` that aren't also (somehow) defined.
fn verify_core_symbols(archive: &BinFile) {
    // Match both mangling styles:
    //
    // * `_ZN4core3str8converts9from_utf817hd4454ac14cbbb790E` (old)
    // * `_RNvNtNtCscK9O3IwVk7N_4core3str8converts9from_utf8` (v0)
    //
    // Also account for the Apple leading `_`.
    static RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^_?_[RZ].*4core").unwrap());

    let mut defined = BTreeSet::new();
    let mut undefined = Vec::new();
    let mut has_symbols = false;

    archive.for_each_symbol(|symbol, obj, member| {
        has_symbols = true;

        // Find only symbols from `core`
        if !RE.is_match(symbol.name().unwrap()) {
            return;
        }

        let sym = SymInfo::new(&symbol, obj, member);
        if sym.is_undefined {
            undefined.push(sym);
        } else {
            defined.insert(sym.name);
        }
    });

    if archive.has_symbol_tables() {
        assert!(has_symbols, "no symbols found");
    }

    // Discard any symbols that are defined somewhere in the archive
    undefined.retain(|sym| !defined.contains(&sym.name));

    if !undefined.is_empty() {
        undefined.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        panic!(
            "found {} undefined symbols from core: {undefined:#?}",
            undefined.len()
        );
    }

    println!("    success: no undefined references to core found");
}

/// Thin wrapper for owning data used by `object`.
struct BinFile {
    path: PathBuf,
    data: Vec<u8>,
}

impl BinFile {
    fn from_path(path: &Path) -> Self {
        Self {
            path: path.to_owned(),
            data: fs::read(path).expect("reading file failed"),
        }
    }

    fn as_archive_file(&self) -> ObjResult<ArchiveFile<'_>> {
        ArchiveFile::parse(self.data.as_slice())
    }

    fn as_obj_file(&self) -> ObjResult<ObjFile<'_>> {
        ObjFile::parse(self.data.as_slice())
    }

    /// For a given archive, do something with each object file. For an object file, do
    /// something once.
    fn for_each_object(&self, mut f: impl FnMut(ObjFile, &str)) {
        // Try as an archive first.
        let as_archive = self.as_archive_file();
        if let Ok(archive) = as_archive {
            for member in archive.members() {
                let member = member.expect("failed to access member");
                let obj_data = member
                    .data(self.data.as_slice())
                    .expect("failed to access object");
                let obj = ObjFile::parse(obj_data).expect("failed to parse object");
                f(obj, &String::from_utf8_lossy(member.name()));
            }

            return;
        }

        // Fall back to parsing as an object file.
        let as_obj = self.as_obj_file();
        if let Ok(obj) = as_obj {
            f(obj, &self.path.to_string_lossy());
            return;
        }

        panic!(
            "failed to parse as either archive or object file: {:?}, {:?}",
            as_archive.unwrap_err(),
            as_obj.unwrap_err(),
        );
    }

    /// D something with each symbol in an archive or object file.
    fn for_each_symbol(&self, mut f: impl FnMut(Symbol, &ObjFile, &str)) {
        self.for_each_object(|obj, obj_path| {
            obj.symbols().for_each(|sym| f(sym, &obj, obj_path));
        });
    }

    /// PE executable files don't have the same kind of symbol tables. This isn't a perfectly
    /// accurate check, but at least tells us whether we can skip erroring if we don't find any
    /// symbols.
    fn has_symbol_tables(&self) -> bool {
        let mut empty = true;
        let mut ret = false;

        self.for_each_object(|obj, _obj_path| {
            if !matches!(obj.format(), BinaryFormat::Pe) {
                // Any non-PE objects should have symbol tables.
                ret = true;
            }
            empty = false;
        });

        // If empty, assume there should be tables.
        empty || ret
    }
}
