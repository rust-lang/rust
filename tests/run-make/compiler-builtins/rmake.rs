//! The compiler_builtins library is special. It can call functions in core, but it must not
//! require linkage against a build of core. If it ever does, building the standard library *may*
//! result in linker errors, depending on whether the linker in use applies optimizations first or
//! resolves symbols first. So the portable and safe approach is to forbid such a linkage
//! requirement entirely.
//!
//! In addition, whether compiler_builtins requires linkage against core can depend on optimization
//! settings. Turning off optimizations and enabling debug assertions tends to produce the most
//! dependence on core that is possible, so that is the configuration we test here.

#![deny(warnings)]

extern crate run_make_support;

use run_make_support::object;
use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::read::Object;
use run_make_support::object::ObjectSection;
use run_make_support::object::ObjectSymbol;
use run_make_support::object::RelocationTarget;
use run_make_support::out_dir;
use std::collections::HashSet;

const MANIFEST: &str = r#"
[package]
name = "scratch"
version = "0.1.0"
edition = "2021"

[lib]
path = "lib.rs""#;

fn main() {
    let target_dir = out_dir().join("target");
    let target = std::env::var("TARGET").unwrap();
    if target.starts_with("wasm") || target.starts_with("nvptx") {
        // wasm and nvptx targets don't produce rlib files that object can parse.
        return;
    }

    println!("Testing compiler_builtins for {}", target);

    // Set up the tiniest Cargo project: An empty no_std library. Just enough to run -Zbuild-std.
    let manifest_path = out_dir().join("Cargo.toml");
    std::fs::write(&manifest_path, MANIFEST.as_bytes()).unwrap();
    std::fs::write(out_dir().join("lib.rs"), b"#![no_std]").unwrap();

    let path = std::env::var("PATH").unwrap();
    let rustc = std::env::var("RUSTC").unwrap();
    let bootstrap_cargo = std::env::var("BOOTSTRAP_CARGO").unwrap();
    let status = std::process::Command::new(bootstrap_cargo)
        .args([
            "build",
            "--manifest-path",
            manifest_path.to_str().unwrap(),
            "-Zbuild-std=core",
            "--target",
            &target,
        ])
        .env_clear()
        .env("PATH", path)
        .env("RUSTC", rustc)
        .env("RUSTFLAGS", "-Copt-level=0 -Cdebug-assertions=yes")
        .env("CARGO_TARGET_DIR", &target_dir)
        .env("RUSTC_BOOTSTRAP", "1")
        .status()
        .unwrap();

    assert!(status.success());

    let rlibs_path = target_dir.join(target).join("debug").join("deps");
    let compiler_builtins_rlib = std::fs::read_dir(rlibs_path)
        .unwrap()
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
    let data = std::fs::read(compiler_builtins_rlib).unwrap();

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
