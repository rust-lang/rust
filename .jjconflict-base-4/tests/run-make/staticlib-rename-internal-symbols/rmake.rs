//@ only-elf
//@ ignore-cross-compile

use std::collections::HashSet;

use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::read::elf::{FileHeader as _, SectionHeader as _, Sym as _};
use run_make_support::object::{Endianness, elf};
use run_make_support::path_helpers::source_root;
use run_make_support::{cc, extra_c_flags, rfs, run, rustc, static_lib_name};

const EXPORTED: &[&str] = &["my_add", "my_hash_lookup", "call_internal", "my_safe_div"];

fn main() {
    let sibling = source_root().join("tests/run-make/staticlib-hide-internal-symbols");
    rfs::copy(sibling.join("lib.rs"), "lib.rs");
    rfs::copy(sibling.join("main.c"), "main.c");

    test_basic_functionality();
    test_rs_suffix_present();
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

fn test_rs_suffix_present() {
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
    let mut found_rs_suffix = false;

    for member in archive.members() {
        let member = member.unwrap();
        if !member.name().ends_with(b".rcgu.o") {
            continue;
        }
        let data = member.data(archive_data).unwrap();

        if let Ok(header) = elf::FileHeader64::<Endianness>::parse(data) {
            check_elf_symbols(header, data, &mut found_exported, &mut found_rs_suffix);
        } else if let Ok(header) = elf::FileHeader32::<Endianness>::parse(data) {
            check_elf_symbols(header, data, &mut found_exported, &mut found_rs_suffix);
        }
    }

    assert!(found_rs_suffix, "expected to find at least one renamed symbol with .rs suffix");
    for expected in EXPORTED {
        assert!(
            found_exported.contains(*expected),
            "expected to find exported symbol `{expected}` in archive"
        );
    }
}

fn check_elf_symbols<Elf: run_make_support::object::read::elf::FileHeader<Endian = Endianness>>(
    header: &Elf,
    data: &[u8],
    found_exported: &mut HashSet<String>,
    found_rs_suffix: &mut bool,
) {
    let Ok(endian) = header.endian() else { return };
    let Ok(sections) = header.sections(endian, data) else { return };

    for (si, section) in sections.enumerate() {
        if section.sh_type(endian) != elf::SHT_SYMTAB {
            continue;
        }
        let Ok(symbols) = run_make_support::object::read::elf::SymbolTable::parse(
            endian, data, &sections, si, section,
        ) else {
            continue;
        };
        let strtab = symbols.strings();

        for symbol in symbols.symbols() {
            let bind = symbol.st_bind();
            let shndx = symbol.st_shndx(endian);

            if shndx == elf::SHN_UNDEF {
                continue;
            }
            if bind != elf::STB_GLOBAL && bind != elf::STB_WEAK {
                continue;
            }

            let Ok(name_bytes) = symbol.name(endian, strtab) else { continue };
            let Ok(name) = str::from_utf8(name_bytes) else { continue };

            if EXPORTED.contains(&name) {
                assert!(
                    !name.contains(".rs"),
                    "exported symbol `{name}` should not contain .rs suffix"
                );
                assert_eq!(
                    symbol.st_visibility(),
                    elf::STV_DEFAULT,
                    "exported symbol `{name}` should be STV_DEFAULT"
                );
                found_exported.insert(name.to_string());
            } else {
                assert!(
                    name.contains(".rs"),
                    "internal symbol `{name}` should contain .rs suffix after rename"
                );
                *found_rs_suffix = true;
            }
        }
    }
}
