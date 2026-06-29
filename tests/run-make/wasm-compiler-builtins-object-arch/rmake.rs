//! Regression test for <https://github.com/rust-lang/rust/issues/132802>.
//!
//! The prebuilt `libcompiler_builtins` rlib bundled in the wasm sysroot must
//! contain wasm object files — never host ELF/Mach-O/COFF. Bootstrap could
//! previously pick the host C toolchain for compiler-rt fallbacks on wasm
//! targets and silently embed host objects into the wasm sysroot
//! (fixed in rust-lang/rust#137457).

//@ only-wasm32

use run_make_support::object::read::Object;
use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::{self, Architecture};
use run_make_support::{has_extension, has_prefix, rfs, rustc, shallow_find_files};

fn main() {
    let libdir = rustc().print("target-libdir").run().stdout_utf8();
    let libdir = libdir.trim();

    let rlibs = shallow_find_files(libdir, |path| {
        has_prefix(path, "libcompiler_builtins") && has_extension(path, "rlib")
    });
    assert!(!rlibs.is_empty(), "no libcompiler_builtins rlib found in {libdir}");

    let data = rfs::read(&rlibs[0]);
    let archive = ArchiveFile::parse(&*data).unwrap();

    let mut checked = 0usize;
    for member in archive.members() {
        let member = member.unwrap();
        let name = std::str::from_utf8(member.name()).unwrap_or("<invalid-utf8>");
        if name.ends_with(".rmeta") || name.ends_with(".rmeta-link") {
            continue;
        }
        let obj_data = member.data(&*data).unwrap();
        let obj = object::File::parse(obj_data).unwrap_or_else(|e| {
            panic!("failed to parse member `{name}` in compiler_builtins rlib: {e}")
        });
        let arch = obj.architecture();
        assert!(
            matches!(arch, Architecture::Wasm32 | Architecture::Wasm64),
            "object `{name}` in compiler_builtins rlib has architecture {arch:?}, \
             expected wasm — see rust-lang/rust#132802",
        );
        checked += 1;
    }

    assert!(
        checked > 0,
        "no object members found in compiler_builtins rlib at {} — \
         archive should always contain object files",
        rlibs[0].display(),
    );
}
