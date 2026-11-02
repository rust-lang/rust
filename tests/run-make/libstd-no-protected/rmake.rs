// If libstd was compiled to use protected symbols, then linking would fail if GNU ld < 2.40 were
// used. This might not be noticed, since usually we use LLD for linking, so we could end up
// distributing a version of libstd that would cause link errors for such users.

//@ only-x86_64-unknown-linux-gnu

use run_make_support::obj::BinFile;
use run_make_support::object::{Endianness, ObjectSymbol, SymbolFlags};
use run_make_support::{has_prefix, has_suffix, object, rustc, shallow_find_files, target};

type FileHeader = run_make_support::object::elf::FileHeader64<Endianness>;
type SymbolTable<'data> = run_make_support::object::read::elf::SymbolTable<'data, FileHeader>;

fn main() {
    // Find libstd-...rlib
    let sysroot_libs_dir = rustc().print("target-libdir").target(target()).run().stdout_utf8();
    let mut libs = shallow_find_files(sysroot_libs_dir.trim(), |path| {
        has_prefix(path, "libstd-") && has_suffix(path, ".rlib")
    });
    assert_eq!(libs.len(), 1);
    let libstd_path = libs.pop().unwrap();
    let archive = BinFile::read_path(libstd_path);

    // Parse all the object files within the libstd archive, checking defined symbols.
    let mut protected = Vec::new();
    let mut num_symbols = 0;

    archive.for_each_symbol(|symbol, _obj, obj_name| {
        if obj_name == "lib.rmeta" {
            return;
        }
        if symbol.flags().elf_visibility().expect("non-elf file") == object::elf::STV_PROTECTED {
            protected.push(String::from_utf8_lossy(symbol.name_bytes().unwrap()).into_owned());
        }
        num_symbols += 1;
    });

    // If there were no symbols at all, then something is wrong with the test.
    assert_ne!(num_symbols, 0);

    // The purpose of this test - check that no symbols have protected visibility.
    assert!(protected.is_empty(), "found protected symbols {protected:#?}");
}
