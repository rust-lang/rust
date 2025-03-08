//! The compiler_builtins library is special. It exists to export a number of intrinsics which may
//! also be provided by libgcc or compiler-rt, and when an intrinsic is provided by another
//! library, we want that definition to override the one in compiler_builtins because we expect
//! that those implementations are more optimized than compiler_builtins. To make sure that an
//! attempt to override a compiler_builtins intrinsic does not result in a multiple definitions
//! linker error, the compiler has special CGU partitioning logic for compiler_builtins that
//! ensures every intrinsic gets its own CGU.
//!
//! This test is slightly overfit to the current compiler_builtins CGU naming strategy; it doesn't
//! distinguish between "multiple intrinsics are in one object file!" which would be very bad, and
//! "This object file has an intrinsic and also some of its helper functions!" which would be okay.
//!
//! This test ensures that the compiler_builtins rlib has only one intrinsic in each object file.

// wasm and nvptx targets don't produce rlib files that object can parse.
//@ ignore-wasm
//@ ignore-nvptx64

#![deny(warnings)]

use std::str;

use run_make_support::object::read::Object;
use run_make_support::object::read::archive::ArchiveFile;
use run_make_support::object::{ObjectSymbol, SymbolKind};
use run_make_support::rfs::{read, read_dir};
use run_make_support::{cargo, object, path, target};

fn main() {
    println!("Testing compiler_builtins CGU partitioning for {}", target());

    // CGU partitioning has some special cases for codegen-units=1, so we also test 2 CGUs.
    for cgus in [1, 2] {
        for profile in ["debug", "release"] {
            run_test(profile, cgus);
        }
    }
}

fn run_test(profile: &str, cgus: usize) {
    println!("Testing with profile {profile} and -Ccodegen-units={cgus}");

    let target_dir = path("target");

    let mut cmd = cargo();
    cmd.args(&[
        "build",
        "--manifest-path",
        "Cargo.toml",
        "-Zbuild-std=core",
        "--target",
        &target(),
    ])
    .env("RUSTFLAGS", &format!("-Ccodegen-units={cgus}"))
    .env("CARGO_TARGET_DIR", &target_dir)
    .env("RUSTC_BOOTSTRAP", "1")
    // Visual Studio 2022 requires that the LIB env var be set so it can
    // find the Windows SDK.
    .env("LIB", std::env::var("LIB").unwrap_or_default());
    if profile == "release" {
        cmd.arg("--release");
    }
    cmd.run();

    let rlibs_path = target_dir.join(target()).join(profile).join("deps");
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

    // rlib files are archives, where the archive members are our CGUs, and we also have one called
    // lib.rmeta which is the encoded metadata. Each of the CGUs is an object file.
    let data = read(compiler_builtins_rlib);

    let archive = ArchiveFile::parse(&*data).unwrap();
    for member in archive.members() {
        let member = member.unwrap();
        if member.name() == b"lib.rmeta" {
            continue;
        }
        let data = member.data(&*data).unwrap();
        let object = object::File::parse(&*data).unwrap();

        let mut global_text_symbols = 0;
        println!("Inspecting object {}", str::from_utf8(&member.name()).unwrap());
        for symbol in object
            .symbols()
            .filter(|symbol| matches!(symbol.kind(), SymbolKind::Text))
            .filter(|symbol| symbol.is_definition() && symbol.is_global())
        {
            println!("symbol: {:?}", symbol.name().unwrap());
            global_text_symbols += 1;
        }
        // Assert that this object/CGU does not define multiple global text symbols.
        // We permit the 0 case because some CGUs may only be assigned a static.
        assert!(global_text_symbols <= 1);
    }
}
