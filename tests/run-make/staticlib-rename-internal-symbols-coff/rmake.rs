//@ only-windows
//@ ignore-cross-compile

use std::collections::HashSet;

use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::read::coff::ImageSymbol as _;
use run_make_support::object::{File, pe};
use run_make_support::path_helpers::source_root;
use run_make_support::{cc, extra_c_flags, rfs, run, rustc, static_lib_name};

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
        // Copy to an aligned buffer: COFF headers are parsed with an aligned
        // `read`, which fails on archive members that sit at an odd offset.
        let data = member.data(archive_data).unwrap().to_vec();
        match File::parse(&*data) {
            Ok(File::Coff(f)) => check_coff_symbols(
                f.coff_header(),
                &data,
                &mut found_exported,
                &mut found_rs_suffix,
            ),
            Ok(File::CoffBig(f)) => check_coff_symbols(
                f.coff_header(),
                &data,
                &mut found_exported,
                &mut found_rs_suffix,
            ),
            Ok(_) => panic!("unexpected object file format in archive member"),
            Err(e) => panic!("failed to parse archive member: {e}"),
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

fn check_coff_symbols<Coff: run_make_support::object::read::coff::CoffHeader>(
    header: &Coff,
    data: &[u8],
    found_exported: &mut HashSet<String>,
    found_rs_suffix: &mut bool,
) {
    // i686 decorates symbol names with a leading underscore.
    let strip_underscore = header.machine() == pe::IMAGE_FILE_MACHINE_I386;
    let Ok(symbols) = header.symbols(data) else { return };
    let strings = symbols.strings();

    for (_index, symbol) in symbols.iter() {
        let storage_class = symbol.storage_class();
        if storage_class != pe::IMAGE_SYM_CLASS_EXTERNAL
            && storage_class != pe::IMAGE_SYM_CLASS_WEAK_EXTERNAL
        {
            continue;
        }
        if symbol.section_number() <= 0 {
            continue;
        }
        let Ok(name_bytes) = symbol.name(strings) else { continue };
        let Ok(mut name) = str::from_utf8(name_bytes).map(String::from) else { continue };
        if strip_underscore {
            name = name.strip_prefix('_').unwrap_or(&name).to_string();
        }

        if EXPORTED.contains(&name.as_str()) {
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
