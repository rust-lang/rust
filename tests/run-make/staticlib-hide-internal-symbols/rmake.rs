//@ only-elf
//@ ignore-cross-compile

use std::collections::HashSet;

use run_make_support::object::Endianness;
use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::read::elf::{FileHeader as _, SectionHeader as _, Sym as _};
use run_make_support::{cc, extra_c_flags, object, rfs, run, rustc, static_lib_name};

type FileHeader64 = run_make_support::object::elf::FileHeader64<Endianness>;
type SymbolTable<'data> = run_make_support::object::read::elf::SymbolTable<'data, FileHeader64>;

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
        let member_name = std::str::from_utf8(member.name()).unwrap();
        if !member_name.ends_with(".rcgu.o") {
            continue;
        }
        let data = member.data(archive_data).unwrap();

        let Ok(header) = FileHeader64::parse(data) else { continue };
        let Ok(endian) = header.endian() else { continue };
        let Ok(sections) = header.sections(endian, data) else { continue };

        for (si, section) in sections.enumerate() {
            if section.sh_type(endian) != object::elf::SHT_SYMTAB {
                continue;
            }
            let Ok(symbols) = SymbolTable::parse(endian, data, &sections, si, section) else {
                continue;
            };
            let strtab = symbols.strings();

            for symbol in symbols.symbols() {
                let vis = symbol.st_visibility();
                let bind = symbol.st_bind();
                let shndx = symbol.st_shndx(endian);

                if shndx == object::elf::SHN_UNDEF as u16 {
                    continue;
                }
                if bind != object::elf::STB_GLOBAL && bind != object::elf::STB_WEAK {
                    continue;
                }

                let Some(name) = read_symbol_name(endian, symbol, &strtab) else { continue };

                let exported = EXPORTED.contains(&name);

                if with_flag {
                    let expected =
                        if exported { object::elf::STV_DEFAULT } else { object::elf::STV_HIDDEN };
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
                        object::elf::STV_DEFAULT,
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

    for expected in EXPORTED {
        assert!(
            found_exported.contains(*expected),
            "expected to find exported symbol `{expected}` in archive"
        );
    }
}

fn read_symbol_name<'data>(
    endian: Endianness,
    symbol: &run_make_support::object::elf::Sym64<Endianness>,
    strtab: &object::StringTable<'data>,
) -> Option<&'data str> {
    let bytes = strtab.get(symbol.st_name(endian)).ok()?;
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    std::str::from_utf8(&bytes[..end]).ok()
}

fn visibility_name(v: u8) -> &'static str {
    match v {
        v if v == object::elf::STV_DEFAULT => "STV_DEFAULT",
        v if v == object::elf::STV_HIDDEN => "STV_HIDDEN",
        v if v == object::elf::STV_INTERNAL => "STV_INTERNAL",
        v if v == object::elf::STV_PROTECTED => "STV_PROTECTED",
        _ => "UNKNOWN",
    }
}
