//! Tool used by CI to inspect compiler-builtins archives and help ensure we won't run into any
//! linking errors.
//!
//! Note that symcheck is a "hostprog", i.e. is built and run on the host target even when the
//! actual target is cross compiled.

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, exit};
use std::sync::LazyLock;
use std::{env, fs};

use object::read::archive::ArchiveFile;
use object::{
    Architecture, BinaryFormat, Endianness, File as ObjFile, Object, ObjectSection, ObjectSymbol,
    Result as ObjResult, SectionFlags, Symbol, SymbolKind, SymbolScope, U32, elf,
};
use regex::Regex;
use serde_json::Value;

const CHECK_LIBRARIES: &[&str] = &["compiler_builtins", "builtins_test_intrinsics"];
const CHECK_EXTENSIONS: &[Option<&str>] = &[Some("rlib"), Some("a"), Some("exe"), None];
const GNU_STACK: &str = ".note.GNU-stack";

const USAGE: &str = "Usage:

    symbol-check --build-and-check [--target TARGET] [--no-os] -- CARGO_BUILD_ARGS ...
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
    opts.optflag(
        "",
        "no-os",
        "The binaries will not be checked for executable stacks. Used for embedded targets which \
        don't set `.note.GNU-stack` since there is no protection.",
    );
    opts.optflag("", "no-visibility", "Don't check visibility.");

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

    let no_os_target = m.opt_present("no-os");
    let check_visibility = !m.opt_present("no-visibility");
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
        check_paths(&paths, no_os_target, check_visibility);
    } else if m.opt_present("check") {
        if free_args.is_empty() {
            print_usage_and_exit(1);
        }
        check_paths(&free_args, no_os_target, check_visibility);
    } else {
        print_usage_and_exit(1);
    }
}

fn check_paths<P: AsRef<Path>>(paths: &[P], no_os_target: bool, check_visibility: bool) {
    for path in paths {
        let path = path.as_ref();
        println!("Checking {}", path.display());
        let archive = BinFile::from_path(path);

        verify_no_duplicates(&archive);
        verify_core_symbols(&archive);
        verify_no_exec_stack(&archive, no_os_target);
        if check_visibility {
            verify_hidden_visibility(&archive);
        }
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

/// Check for symbols with default visibility.
fn verify_hidden_visibility(archive: &BinFile) {
    let mut visible = Vec::new();
    let mut found_any = false;

    archive.for_each_symbol(|symbol, obj, member| {
        // Only check defined globals.
        if !symbol.is_global() || symbol.is_undefined() {
            return;
        }

        let sym = SymInfo::new(&symbol, obj, member);
        if sym.scope == SymbolScope::Dynamic {
            visible.push(sym);
        }

        found_any = true
    });

    if archive.has_symbol_tables() {
        assert!(found_any, "no symbols found");
    }

    if !visible.is_empty() {
        visible.sort_unstable_by(|a, b| a.name.cmp(&b.name));
        let num = visible.len();
        panic!("found {num:#?} visible symbols: {visible:#?}");
    }

    println!("    success: no visible symbols found");
}

/// Reasons a binary is considered to have an executable stack.
enum ExeStack {
    MissingGnuStackSec,
    ExeGnuStackSec,
    ExePtGnuStack,
}

/// Ensure that the object/archive will not require an executable stack.
fn verify_no_exec_stack(archive: &BinFile, no_os_target: bool) {
    if no_os_target {
        // We don't really have a good way of knowing whether or not an elf file is for a
        // no-os environment so we rely on a CLI arg (note.GNU-stack doesn't get emitted if
        // there is no OS to protect the stack).
        println!("    skipping check for writeable+executable stack on no-os target");
        return;
    }

    let mut problem_objfiles = Vec::new();

    archive.for_each_object(|obj, obj_path| match check_obj_exe_stack(&obj) {
        Ok(()) => (),
        Err(exe) => problem_objfiles.push((obj_path.to_owned(), exe)),
    });

    if problem_objfiles.is_empty() {
        println!("    success: no writeable+executable stack indicators found");
        return;
    }

    eprintln!("the following object files require an executable stack:");

    for (obj, exe) in problem_objfiles {
        let reason = match exe {
            ExeStack::MissingGnuStackSec => "no .note.GNU-stack section",
            ExeStack::ExeGnuStackSec => ".note.GNU-stack section marked SHF_EXECINSTR",
            ExeStack::ExePtGnuStack => "PT_GNU_STACK program header marked PF_X",
        };
        eprintln!("    {obj} ({reason})");
    }

    exit(1);
}

/// `Err` if the section/flag combination indicates that the object file should be linked with an
/// executable stack.
fn check_obj_exe_stack(obj: &ObjFile) -> Result<(), ExeStack> {
    match obj.format() {
        BinaryFormat::Elf => check_elf_exe_stack(obj),
        // Technically has the `MH_ALLOW_STACK_EXECUTION` flag but I can't get the compiler to
        // emit it (`-allow_stack_execute` doesn't seem to work in recent versions).
        BinaryFormat::MachO => Ok(()),
        // Can't find much information about Windows stack executability.
        BinaryFormat::Coff | BinaryFormat::Pe => Ok(()),
        // Also not sure about wasm.
        BinaryFormat::Wasm => Ok(()),
        BinaryFormat::Xcoff | _ => {
            unimplemented!("binary format {:?} is not supported", obj.format())
        }
    }
}

/// Check for an executable stack in elf binaries.
///
/// If the `PT_GNU_STACK` header on a binary is present and marked executable, the binary will
/// have an executable stack (RWE rather than the desired RW). If any object file has the right
/// `.note.GNU-stack` logic, the final binary will get `PT_GNU_STACK`.
///
/// Individual object file logic is as follows, paraphrased from [1]:
///
/// - A `.note.GNU-stack` section with the exe flag means this needs an executable stack
/// - A `.note.GNU-stack` section without the exe flag means there is no executable stack needed
/// - Without the section, behavior is target-specific. Historically it usually means an executable
///   stack is required.
///
/// Per [2], it is now deprecated behavior for a missing `.note.GNU-stack` section to imply an
/// executable stack. However, we shouldn't assume that tooling has caught up to this.
///
/// [1]: https://www.man7.org/linux/man-pages/man1/ld.1.html
/// [2]: https://sourceware.org/git/gitweb.cgi?p=binutils-gdb.git;h=0d38576a34ec64a1b4500c9277a8e9d0f07e6774>
fn check_elf_exe_stack(obj: &ObjFile) -> Result<(), ExeStack> {
    let end = obj.endianness();

    // Check for PT_GNU_STACK marked executable
    let mut is_obj_exe = false;
    let mut found_gnu_stack = false;
    let mut check_ph = |p_type: U32<Endianness>, p_flags: U32<Endianness>| {
        let ty = p_type.get(end);
        let flags = p_flags.get(end);

        // Presence of PT_INTERP indicates that this is an executable rather than a standalone
        // object file.
        if ty == elf::PT_INTERP {
            is_obj_exe = true;
        }

        if ty == elf::PT_GNU_STACK {
            assert!(!found_gnu_stack, "multiple PT_GNU_STACK sections");
            found_gnu_stack = true;
            if flags & elf::PF_X != 0 {
                return Err(ExeStack::ExePtGnuStack);
            }
        }

        Ok(())
    };

    match obj {
        ObjFile::Elf32(f) => {
            for ph in f.elf_program_headers() {
                check_ph(ph.p_type, ph.p_flags)?;
            }
        }
        ObjFile::Elf64(f) => {
            for ph in f.elf_program_headers() {
                check_ph(ph.p_type, ph.p_flags)?;
            }
        }
        _ => panic!("should only be called with elf objects"),
    }

    if is_obj_exe {
        return Ok(());
    }

    // The remaining are checks for individual object files, which wind up controlling PT_GNU_STACK
    // in the final binary.
    let mut gnu_stack_exe = None;
    let mut has_exe_sections = false;
    for sec in obj.sections() {
        let SectionFlags::Elf { sh_flags } = sec.flags() else {
            unreachable!("only elf files are being checked");
        };

        let is_sec_exe = sh_flags & u64::from(elf::SHF_EXECINSTR) != 0;

        // If the magic section is present, its exe bit tells us whether or not the object
        // file requires an executable stack.
        if sec.name().unwrap_or_default() == GNU_STACK {
            assert!(gnu_stack_exe.is_none(), "multiple {GNU_STACK} sections");
            if is_sec_exe {
                gnu_stack_exe = Some(Err(ExeStack::ExeGnuStackSec));
            } else {
                gnu_stack_exe = Some(Ok(()));
            }
        }

        // Otherwise, just keep track of whether or not we have exeuctable sections
        has_exe_sections |= is_sec_exe;
    }

    // GNU_STACK sets the executability if specified.
    if let Some(exe) = gnu_stack_exe {
        return exe;
    }

    // Ignore object files that have no executable sections, like rmeta.
    if !has_exe_sections {
        return Ok(());
    }

    // If there is no `.note.GNU-stack` and no executable sections, behavior differs by platform.
    match obj.architecture() {
        // PPC64 doesn't set `.note.GNU-stack` since GNU nested functions don't need a trampoline,
        // <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=21098>. From experimentation, it seems
        // like this only applies to big endian.
        Architecture::PowerPc64 if obj.endianness() == Endianness::Big => Ok(()),

        _ => Err(ExeStack::MissingGnuStackSec),
    }
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
