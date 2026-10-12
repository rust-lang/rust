//! Tests that `-Zstaticlib-rename-internal-symbols` renames internal symbols on COFF targets
//! while leaving exported symbols untouched, taking COFF decoration (i686 cdecl `_` /
//! stdcall `_@N` / fastcall `@N` / vectorcall `@@N`, Arm64EC text `#`) into account so decorated
//! exported symbols are neither renamed nor missed.

//@ only-windows
//@ ignore-cross-compile

use std::collections::HashSet;

use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::read::coff::ImageSymbol as _;
use run_make_support::object::{File, pe};
use run_make_support::path_helpers::source_root;
use run_make_support::{
    cc, extra_c_flags, is_windows_msvc, rfs, run, rustc, static_lib_name, target,
};

/// The undecorated exported names of the base fixture (`lib.rs`).
const BASE_EXPORTED: &[&str] = &["my_add", "my_hash_lookup", "call_internal", "my_safe_div"];

fn main() {
    let hide_sibling = source_root().join("tests/run-make/staticlib-hide-internal-symbols");
    let rename_sibling = source_root().join("tests/run-make/staticlib-rename-internal-symbols");
    rfs::copy(hide_sibling.join("lib.rs"), "lib.rs");
    rfs::copy(hide_sibling.join("main.c"), "main.c");
    rfs::copy(rename_sibling.join("liba.rs"), "liba.rs");
    rfs::copy(rename_sibling.join("libb.rs"), "libb.rs");
    rfs::copy(rename_sibling.join("dual_main.c"), "dual_main.c");

    test_basic_functionality();
    test_rs_suffix_present();
    test_dual_staticlib_linking();
    test_hide_and_rename();
    test_decorated_symbols();
}

/// Compile `<crate_name>.rs` into a staticlib under `-Zstaticlib-rename-internal-symbols`.
fn compile_staticlib(crate_name: &str) -> String {
    let lib_name = static_lib_name(crate_name);
    rustc()
        .input(format!("{crate_name}.rs"))
        .crate_type("staticlib")
        .arg("-Zstaticlib-rename-internal-symbols")
        .opt()
        .run();
    lib_name
}

/// Link `main_c` against `libs` into `exe`, then run it.
fn link_and_run(main_c: &str, libs: &[&str], exe: &str) {
    let mut cmd = cc();
    cmd.input(main_c);
    for lib in libs {
        cmd.input(*lib);
    }
    cmd.out_exe(exe).args(extra_c_flags()).run();
    run(exe);
}

fn test_basic_functionality() {
    let lib = compile_staticlib("lib");
    link_and_run("main.c", &[lib.as_str()], "main");
    rfs::remove_file(&lib);
}

fn test_rs_suffix_present() {
    let lib = compile_staticlib("lib");
    let data = rfs::read(&lib);
    check_rename_symbols(&data, BASE_EXPORTED, MatchMode::Undecorated);
    rfs::remove_file(&lib);
}

fn test_dual_staticlib_linking() {
    let liba = compile_staticlib("liba");
    let libb = compile_staticlib("libb");
    link_and_run("dual_main.c", &[liba.as_str(), libb.as_str()], "dual_main");
}

/// On COFF, hiding is unsupported and must only produce a warning, while
/// renaming still applies.
fn test_hide_and_rename() {
    let lib_name = static_lib_name("lib");

    rustc()
        .input("lib.rs")
        .crate_type("staticlib")
        .arg("-Zstaticlib-hide-internal-symbols")
        .arg("-Zstaticlib-rename-internal-symbols")
        .opt()
        .run_unchecked()
        .assert_stderr_contains(
            "-Zstaticlib-hide-internal-symbols only supports ELF and Mach-O targets",
        )
        .assert_exit_code(0);

    let data = rfs::read(&lib_name);
    check_rename_symbols(&data, BASE_EXPORTED, MatchMode::Undecorated);

    link_and_run("main.c", &[lib_name.as_str()], "main");
    rfs::remove_file(&lib_name);
}

/// Assert the stdcall/fastcall/vectorcall exports (`_@N` / `@N` / `@@N`) don't get renamed
/// into their undecorated form.
fn test_decorated_symbols() {
    let lib = compile_staticlib("decorated_lib");
    let data = rfs::read(&lib);
    check_rename_symbols(&data, &decorated_exports(), MatchMode::Raw);
    rfs::remove_file(&lib);
}

/// Raw decorated names `decorated_lib.rs` emits: i686 cdecl `_`, stdcall `_@N`, fastcall `@N`,
/// and vectorcall `@@N` (MSVC x86/x86_64 only).
fn decorated_exports() -> Vec<&'static str> {
    let mut exports = if target().starts_with("i686") {
        vec![
            "_decorated_cdecl",
            "_decorated_calls_internal",
            "_decorated_stdcall@8",
            "@decorated_fastcall@8",
        ]
    } else {
        vec!["decorated_cdecl", "decorated_calls_internal"]
    };
    if is_windows_msvc() && (target().starts_with("i686") || target().starts_with("x86_64")) {
        exports.push("decorated_vectorcall@@8");
    }
    exports
}

/// How exported symbol names are matched against the object file's symbol table.
#[derive(Clone, Copy)]
enum MatchMode {
    /// Strip the target's leading prefix (`_` i686 cdecl, `#` Arm64EC text) before matching.
    Undecorated,
    /// Match raw object-file names verbatim.
    Raw,
}

fn check_rename_symbols(archive_data: &[u8], exported: &[&str], mode: MatchMode) {
    let archive = ArchiveFile::parse(archive_data).unwrap();
    let mut found_exported = HashSet::new();
    let mut found_rs_suffix = false;

    for member in archive.members() {
        let member = member.unwrap();
        if !member.name().ends_with(b".rcgu.o") {
            continue;
        }
        // COFF header/symbol types have alignment 1, so odd-offset members
        // parse directly from the borrowed slice.
        let data = member.data(archive_data).unwrap();
        match File::parse(data) {
            Ok(File::Coff(f)) => check_coff_symbols(
                f.coff_header(),
                data,
                exported,
                mode,
                &mut found_exported,
                &mut found_rs_suffix,
            ),
            Ok(File::CoffBig(f)) => check_coff_symbols(
                f.coff_header(),
                data,
                exported,
                mode,
                &mut found_exported,
                &mut found_rs_suffix,
            ),
            Ok(_) => panic!("unexpected object file format in archive member"),
            Err(e) => panic!("failed to parse archive member: {e}"),
        }
    }

    assert!(found_rs_suffix, "expected to find at least one renamed symbol with .rs suffix");
    for expected in exported {
        assert!(
            found_exported.contains(*expected),
            "expected to find exported symbol `{expected}` in archive"
        );
    }
}

fn check_coff_symbols<Coff: run_make_support::object::read::coff::CoffHeader>(
    header: &Coff,
    data: &[u8],
    exported: &[&str],
    mode: MatchMode,
    found_exported: &mut HashSet<String>,
    found_rs_suffix: &mut bool,
) {
    // ImageSymbol is 18 bytes; ImageSymbolEx (bigobj) is 20.
    let sym_size = std::mem::size_of::<Coff::ImageSymbolBytes>();
    let Ok(symbols) = header.symbols(data) else { return };
    let strings = symbols.strings();
    let symtab_base = header.pointer_to_symbol_table() as usize;

    for (index, symbol) in symbols.iter() {
        let storage_class = symbol.storage_class();
        if storage_class != pe::IMAGE_SYM_CLASS_EXTERNAL
            && storage_class != pe::IMAGE_SYM_CLASS_WEAK_EXTERNAL
        {
            continue;
        }
        if symbol.section_number() <= 0 {
            continue;
        }
        // String-table references keep all four leading name bytes zero.
        let name_field = symtab_base + index.0 * sym_size;
        if data[name_field] == 0 {
            assert!(
                data[name_field + 1..name_field + 4] == [0, 0, 0],
                "long-name symbol reference at offset {name_field} has non-zero padding bytes"
            );
        }
        let Ok(name_bytes) = symbol.name(strings) else { continue };
        let Ok(mut name) = str::from_utf8(name_bytes).map(String::from) else { continue };

        if matches!(mode, MatchMode::Undecorated) {
            match header.machine() {
                pe::IMAGE_FILE_MACHINE_I386 => {
                    name = name.strip_prefix('_').unwrap_or(&name).to_string();
                }
                pe::IMAGE_FILE_MACHINE_ARM64EC => {
                    name = name.strip_prefix('#').unwrap_or(&name).to_string();
                }
                _ => {}
            }
        }

        if exported.contains(&name.as_str()) {
            assert!(
                !name.contains(".rs"),
                "exported symbol `{name}` should not contain .rs suffix"
            );
            found_exported.insert(name);
        } else {
            assert!(
                name.contains(".rs"),
                "internal symbol `{name}` should contain .rs suffix after rename"
            );
            *found_rs_suffix = true;
        }
    }
}
