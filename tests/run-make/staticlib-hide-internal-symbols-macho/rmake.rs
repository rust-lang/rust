//@ only-apple
//@ ignore-cross-compile

use std::collections::HashSet;

use run_make_support::object::Endianness;
use run_make_support::object::macho::{MachHeader64, N_EXT, N_PEXT, N_SECT, N_STAB, N_TYPE};
use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::read::macho::{MachHeader as _, Nlist as _};
use run_make_support::path_helpers::source_root;
use run_make_support::{cc, extra_c_flags, rfs, run, rustc, static_lib_name};

type MachOFileHeader64 = MachHeader64<Endianness>;
type SymbolTable<'data> =
    run_make_support::object::read::macho::SymbolTable<'data, MachOFileHeader64>;

const EXPORTED: &[&str] = &["my_add", "my_hash_lookup", "call_internal", "my_safe_div"];

fn main() {
    let sibling = source_root().join("tests/run-make/staticlib-hide-internal-symbols");
    rfs::copy(sibling.join("lib.rs"), "lib.rs");
    rfs::copy(sibling.join("main.c"), "main.c");

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
            let Ok(name) = str::from_utf8(name_bytes) else { continue };
            let name = name.strip_prefix('_').unwrap_or(name);

            let exported = EXPORTED.contains(&name);
            let has_pext = n_type & N_PEXT != 0;

            if with_flag {
                if exported {
                    assert!(!has_pext, "with -Z hide: exported `{name}` should NOT have N_PEXT");
                } else {
                    assert!(has_pext, "with -Z hide: internal `{name}` should have N_PEXT");
                }
            } else if exported {
                assert!(!has_pext, "without -Z: exported `{name}` should NOT have N_PEXT");
            }

            if exported {
                found_exported.insert(name.to_string());
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
