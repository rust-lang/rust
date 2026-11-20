//@ only-apple
//@ ignore-cross-compile

use std::collections::HashSet;

use run_make_support::object::Endianness;
use run_make_support::object::macho::{MachHeader64, N_EXT, N_PEXT, N_SECT, N_STAB, N_TYPE};
use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::read::macho::{MachHeader as _, Nlist as _};
use run_make_support::path_helpers::source_root;
use run_make_support::{cc, extra_c_flags, object, rfs, run, rustc, static_lib_name};

type MachOFileHeader64 = MachHeader64<Endianness>;
type SymbolTable<'data> =
    run_make_support::object::read::macho::SymbolTable<'data, MachOFileHeader64>;

const EXPORTED: &[&str] = &["my_add", "my_hash_lookup", "call_internal", "my_safe_div"];

fn main() {
    let hide_sibling = source_root().join("tests/run-make/staticlib-hide-internal-symbols");
    let rename_sibling = source_root().join("tests/run-make/staticlib-rename-internal-symbols");
    rfs::copy(hide_sibling.join("lib.rs"), "lib.rs");
    rfs::copy(hide_sibling.join("main.c"), "main.c");
    rfs::copy(rename_sibling.join("liba.rs"), "liba.rs");
    rfs::copy(rename_sibling.join("libb.rs"), "libb.rs");
    rfs::copy(rename_sibling.join("dual_main.c"), "dual_main.c");

    test_basic_functionality();
    test_suffix_present();
    test_dual_staticlib_linking();
}

fn test_basic_functionality() {
    let lib_name = static_lib_name("lib");

    rustc()
        .input("lib.rs")
        .crate_type("staticlib")
        .arg("-Zstaticlib-rename-internal-symbols")
        .opt()
        .run();

    cc().input("main.c").input(&lib_name).out_exe("main").args(extra_c_flags()).run();
    run("main");

    rfs::remove_file(&lib_name);
}

fn test_suffix_present() {
    let lib_name = static_lib_name("lib");

    rustc()
        .input("lib.rs")
        .crate_type("staticlib")
        .arg("-Zstaticlib-rename-internal-symbols")
        .opt()
        .run();

    let data = rfs::read(&lib_name);
    check_rename_symbols(&data);

    rfs::remove_file(&lib_name);
}

fn test_dual_staticlib_linking() {
    let liba_name = static_lib_name("liba");
    let libb_name = static_lib_name("libb");

    rustc()
        .input("liba.rs")
        .crate_type("staticlib")
        .arg("-Zstaticlib-rename-internal-symbols")
        .opt()
        .run();

    rustc()
        .input("libb.rs")
        .crate_type("staticlib")
        .arg("-Zstaticlib-rename-internal-symbols")
        .opt()
        .run();

    cc().input("dual_main.c")
        .input(&liba_name)
        .input(&libb_name)
        .out_exe("dual_main")
        .args(extra_c_flags())
        .run();
    run("dual_main");
}

fn check_rename_symbols(archive_data: &[u8]) {
    let archive = ArchiveFile::parse(archive_data).unwrap();
    let mut found_exported = HashSet::new();
    let mut found_suffix = false;

    for member in archive.members() {
        let member = member.unwrap();
        if !member.name().ends_with(b".rcgu.o") {
            continue;
        }
        let data = member.data(archive_data).unwrap();

        let Ok(header) = MachOFileHeader64::parse(data, 0) else { continue };
        let Ok(endian) = header.endian() else { continue };

        let Some(symtab) = find_symtab(header, endian, data) else { continue };
        let strtab = symtab.strings();

        for nlist in symtab.iter() {
            let n_type = nlist.n_type();
            if n_type & N_STAB != 0 {
                continue;
            }
            if n_type & N_EXT == 0 {
                continue;
            }
            if n_type & N_TYPE != N_SECT {
                continue;
            }

            let Ok(name_bytes) = nlist.name(endian, strtab) else { continue };
            let Ok(name) = std::str::from_utf8(name_bytes) else { continue };
            let name = name.strip_prefix('_').unwrap_or(name);

            if EXPORTED.contains(&name) {
                assert!(
                    !name.contains(".rs"),
                    "exported symbol `{name}` should not contain .rs suffix"
                );
                found_exported.insert(name.to_string());
            } else {
                assert!(
                    name.contains(".rs"),
                    "internal symbol `{name}` should contain .rs suffix after rename"
                );
                found_suffix = true;
            }
        }
    }

    assert!(found_suffix, "expected to find at least one renamed symbol with .rs suffix");
    for expected in EXPORTED {
        assert!(
            found_exported.contains(*expected),
            "expected to find exported symbol `{expected}` in archive"
        );
    }
}

fn find_symtab<'data>(
    header: &MachOFileHeader64,
    endian: Endianness,
    data: &'data [u8],
) -> Option<SymbolTable<'data>> {
    let mut commands = header.load_commands(endian, data, 0).ok()?;
    while let Ok(Some(command)) = commands.next() {
        if let Ok(Some(symtab_cmd)) = command.symtab() {
            return symtab_cmd.symbols::<MachOFileHeader64, _>(endian, data).ok();
        }
    }
    None
}
