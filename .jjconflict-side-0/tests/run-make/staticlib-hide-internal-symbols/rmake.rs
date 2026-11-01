//@ only-elf
//@ ignore-cross-compile

use std::collections::HashSet;

use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::read::elf::{FileHeader as _, SectionHeader as _, Sym as _};
use run_make_support::object::{Endianness, elf};
use run_make_support::{cc, extra_c_flags, rfs, run, rustc, static_lib_name};

const EXPORTED: &[&str] = &["my_add", "my_hash_lookup", "call_internal", "my_safe_div"];

fn main() {
    let lib_name = static_lib_name("lib");

    rustc()
        .input("lib.rs")
        .crate_type("staticlib")
        .arg("-Zstaticlib-hide-internal-symbols")
        .opt()
        .run();

    cc().input("main.c").input(&lib_name).out_exe("main").args(extra_c_flags()).run();
    run("main");

    let data = rfs::read(&lib_name);
    check_symbols(&data, true);

    rfs::remove_file(&lib_name);
    rustc().input("lib.rs").crate_type("staticlib").opt().run();

    let data = rfs::read(&lib_name);
    check_symbols(&data, false);
}

fn check_symbols(archive_data: &[u8], with_flag: bool) {
    let archive = ArchiveFile::parse(archive_data).unwrap();
    let mut found_exported = HashSet::new();

    for member in archive.members() {
        let member = member.unwrap();
        if !member.name().ends_with(b".rcgu.o") {
            continue;
        }
        let data = member.data(archive_data).unwrap();

        if let Ok(header) = elf::FileHeader64::<Endianness>::parse(data) {
            check_elf_symbols(header, data, with_flag, &mut found_exported);
        } else if let Ok(header) = elf::FileHeader32::<Endianness>::parse(data) {
            check_elf_symbols(header, data, with_flag, &mut found_exported);
        }
    }

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
    with_flag: bool,
    found_exported: &mut HashSet<String>,
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
            let vis = symbol.st_visibility();
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

            let exported = EXPORTED.contains(&name);

            if with_flag {
                let expected = if exported { elf::STV_DEFAULT } else { elf::STV_HIDDEN };
                assert_eq!(
                    vis,
                    expected,
                    "with -Z hide: `{name}` should be {}, got {}",
                    visibility_name(expected),
                    visibility_name(vis)
                );
            } else if exported {
                assert_eq!(
                    vis,
                    elf::STV_DEFAULT,
                    "without -Z: `{name}` should be STV_DEFAULT, got {}",
                    visibility_name(vis)
                );
            }

            if exported {
                found_exported.insert(name.to_string());
            }
        }
    }
}

fn visibility_name(v: u8) -> &'static str {
    match v {
        elf::STV_DEFAULT => "STV_DEFAULT",
        elf::STV_HIDDEN => "STV_HIDDEN",
        elf::STV_INTERNAL => "STV_INTERNAL",
        elf::STV_PROTECTED => "STV_PROTECTED",
        _ => "UNKNOWN",
    }
}
