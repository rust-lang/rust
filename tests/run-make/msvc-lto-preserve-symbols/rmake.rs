//@ needs-llvm-components: x86
//@ ignore-backends: gcc

// Verify that ThinLTO preserves both ordinary LLVM IR names and explicitly
// decorated MSVC symbols. Compile for i686 without requiring the i686 std
// sysroot by using the shared minicore test fixture.

use std::collections::HashSet;

use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::{File, Object, ObjectSymbol};
use run_make_support::{path, rfs, rustc, rustc_minicore};

fn main() {
    rustc_minicore().target("i686-pc-windows-msvc").output("libminicore.rlib").run();

    rustc()
        .input("lib.rs")
        .crate_type("staticlib")
        .target("i686-pc-windows-msvc")
        .extern_("minicore", path("libminicore.rlib"))
        .arg("-Clto=thin")
        .codegen_units(16)
        .arg("-Cpanic=abort")
        .opt()
        .output("preserve.lib")
        .run();

    let data = rfs::read("preserve.lib");
    let archive = ArchiveFile::parse(&*data).unwrap();
    let mut defined = HashSet::new();

    for member in archive.members() {
        let member = member.unwrap();
        let Ok(bytes) = member.data(&*data) else {
            continue;
        };
        let Ok(file) = File::parse(bytes) else {
            continue;
        };

        for symbol in file.symbols() {
            if symbol.is_definition() && symbol.is_global() {
                if let Ok(name) = symbol.name() {
                    defined.insert(name.to_owned());
                }
            }
        }
    }

    for expected in
        ["___rust_lto_preserve_plain", "_rust_lto_preserve_stdcall@8", "_RUST_LTO_PRESERVE_DATA"]
    {
        assert!(
            defined.contains(expected),
            "ThinLTO did not preserve exported symbol `{expected}`"
        );
    }
}
