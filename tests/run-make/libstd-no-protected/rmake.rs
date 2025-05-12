// If libstd was compiled to use protected symbols, then linking would fail if GNU ld < 2.40 were
// used. This might not be noticed, since usually we use LLD for linking, so we could end up
// distributing a version of libstd that would cause link errors for such users.

//@ only-x86_64-unknown-linux-gnu

use run_make_support::object::Endianness;
use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::read::elf::{FileHeader as _, SectionHeader as _};
use run_make_support::rfs::read;
use run_make_support::{has_prefix, has_suffix, object, path, rustc, shallow_find_files, target};

type FileHeader = run_make_support::object::elf::FileHeader64<Endianness>;
type SymbolTable<'data> = run_make_support::object::read::elf::SymbolTable<'data, FileHeader>;

fn main() {
    // Find libstd-...rlib
    let sysroot = rustc().print("sysroot").run().stdout_utf8();
    let sysroot = sysroot.trim();
    let target_sysroot = path(sysroot).join("lib/rustlib").join(target()).join("lib");
    let mut libs = shallow_find_files(&target_sysroot, |path| {
        has_prefix(path, "libstd-") && has_suffix(path, ".rlib")
    });
    assert_eq!(libs.len(), 1);
    let libstd_path = libs.pop().unwrap();
    let archive_data = read(libstd_path);

    // Parse all the object files within the libstd archive, checking defined symbols.
    let mut num_protected = 0;
    let mut num_symbols = 0;

    let archive = ArchiveFile::parse(&*archive_data).unwrap();
    for member in archive.members() {
        let member = member.unwrap();
        if member.name() == b"lib.rmeta" {
            continue;
        }
        let data = member.data(&*archive_data).unwrap();

        let header = FileHeader::parse(data).unwrap();
        let endian = header.endian().unwrap();
        let sections = header.sections(endian, data).unwrap();

        for (section_index, section) in sections.enumerate() {
            if section.sh_type(endian) == object::elf::SHT_SYMTAB {
                let symbols =
                    SymbolTable::parse(endian, data, &sections, section_index, section).unwrap();
                for symbol in symbols.symbols() {
                    if symbol.st_visibility() == object::elf::STV_PROTECTED {
                        num_protected += 1;
                    }
                    num_symbols += 1;
                }
            }
        }
    }

    // If there were no symbols at all, then something is wrong with the test.
    assert_ne!(num_symbols, 0);

    // The purpose of this test - check that no symbols have protected visibility.
    assert_eq!(num_protected, 0);
}
