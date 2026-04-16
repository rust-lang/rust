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
    let archive = ArchiveFile::parse(&*data).unwrap();
    let mut found_exported = HashSet::new();
    let mut found_rs_suffix = false;

    for member in archive.members() {
        let member = member.unwrap();
        let member_name = std::str::from_utf8(member.name()).unwrap();
        if !member_name.ends_with(".rcgu.o") {
            continue;
        }
        let member_data = member.data(&*data).unwrap();

        let Ok(header) = FileHeader64::parse(member_data) else { continue };
        let Ok(endian) = header.endian() else { continue };
        let Ok(sections) = header.sections(endian, member_data) else { continue };

        for (si, section) in sections.enumerate() {
            if section.sh_type(endian) != object::elf::SHT_SYMTAB {
                continue;
            }
            let Ok(symbols) = SymbolTable::parse(endian, member_data, &sections, si, section)
            else {
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

                if EXPORTED.contains(&name) {
                    assert!(
                        !name.contains("_rs"),
                        "exported symbol `{name}` should not contain _rs suffix"
                    );
                    assert_eq!(
                        vis,
                        object::elf::STV_DEFAULT,
                        "exported symbol `{name}` should be STV_DEFAULT, got {}",
                        visibility_name(vis)
                    );
                    found_exported.insert(name.to_string());
                } else {
                    assert!(
                        name.contains("_rs"),
                        "internal symbol `{name}` should contain _rs suffix after rename"
                    );
                    assert_ne!(
                        vis,
                        object::elf::STV_DEFAULT,
                        "renamed internal symbol `{name}` should NOT be STV_DEFAULT"
                    );
                    found_rs_suffix = true;
                }
            }
        }
    }

    assert!(found_rs_suffix, "expected to find at least one renamed symbol with _rs suffix");
    for expected in EXPORTED {
        assert!(
            found_exported.contains(*expected),
            "expected to find exported symbol `{expected}` in archive"
        );
    }

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
