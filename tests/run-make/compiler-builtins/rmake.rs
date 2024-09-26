//! The compiler_builtins library is special. It can call functions in core, but it must not
//! require linkage against a build of core. If it ever does, building the standard library *may*
//! result in linker errors, depending on whether the linker in use applies optimizations first or
//! resolves symbols first. So the portable and safe approach is to forbid such a linkage
//! requirement entirely.
//!
//! In addition, whether compiler_builtins requires linkage against core can depend on optimization
//! settings. Turning off optimizations and enabling debug assertions tends to produce the most
//! dependence on core that is possible, so that is the configuration we test here.

// wasm and nvptx targets don't produce rlib files that object can parse.
//@ ignore-wasm
//@ ignore-nvptx64

#![deny(warnings)]

use std::collections::HashSet;

use run_make_support::object::read::Object;
use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::{ObjectSection, ObjectSymbol, RelocationTarget};
use run_make_support::rfs::{read, read_dir};
use run_make_support::{cargo, object, path, target};

fn main() {
    let target_dir = path("target");

    println!("Testing compiler_builtins for {}", target());

    cargo()
        .args(&[
            "build",
            "--manifest-path",
            "Cargo.toml",
            "-Zbuild-std=core",
            "--target",
            &target(),
        ])
        .env("RUSTFLAGS", "-Copt-level=0 -Cdebug-assertions=yes")
        .env("CARGO_TARGET_DIR", &target_dir)
        .env("RUSTC_BOOTSTRAP", "1")
        // Visual Studio 2022 requires that the LIB env var be set so it can
        // find the Windows SDK.
        .env("LIB", std::env::var("LIB").unwrap_or_default())
        .run();

    let rlibs_path = target_dir.join(target()).join("debug").join("deps");
    let compiler_builtins_rlib = read_dir(rlibs_path)
        .find_map(|e| {
            let path = e.unwrap().path();
            let file_name = path.file_name().unwrap().to_str().unwrap();
            if file_name.starts_with("libcompiler_builtins") && file_name.ends_with(".rlib") {
                Some(path)
            } else {
                None
            }
        })
        .unwrap();

    // rlib files are archives, where the archive members each a CGU, and we also have one called
    // lib.rmeta which is the encoded metadata. Each of the CGUs is an object file.
    let data = read(compiler_builtins_rlib);

    let mut defined_symbols = HashSet::new();
    let mut undefined_relocations = HashSet::new();

    let archive = ArchiveFile::parse(&*data).unwrap();
    for member in archive.members() {
        let member = member.unwrap();
        if member.name() == b"lib.rmeta" {
            continue;
        }
        let data = member.data(&*data).unwrap();
        let object = object::File::parse(&*data).unwrap();

        // Record all defined symbols in this CGU.
        for symbol in object.symbols() {
            if !symbol.is_undefined() {
                let name = symbol.name().unwrap();
                defined_symbols.insert(name);
            }
        }

        // Find any relocations against undefined symbols. Calls within this CGU are relocations
        // against a defined symbol.
        for (_offset, relocation) in object.sections().flat_map(|section| section.relocations()) {
            let RelocationTarget::Symbol(symbol_index) = relocation.target() else {
                continue;
            };
            let symbol = object.symbol_by_index(symbol_index).unwrap();
            if symbol.is_undefined() {
                let name = symbol.name().unwrap();
                undefined_relocations.insert(name);
            }
        }
    }

    // We can have symbols in the compiler_builtins rlib that are actually from core, if they were
    // monomorphized in the compiler_builtins crate. This is totally fine, because though the call
    // is to a function in core, it's resolved internally.
    //
    // It is normal to have relocations against symbols not defined in the rlib for things like
    // unwinding, or math functions provided the target's platform libraries. Finding these is not
    // a problem, we want to specifically ban relocations against core which are not resolved
    // internally.
    undefined_relocations
        .retain(|symbol| !defined_symbols.contains(symbol) && symbol.contains("core"));

    if !undefined_relocations.is_empty() {
        panic!(
            "compiler_builtins must not link against core, but it does. \n\
            These symbols may be undefined in a debug build of compiler_builtins:\n\
            {:?}",
            undefined_relocations
        );
    }
}
