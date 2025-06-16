//@ needs-target-std
//
// This test checks that the object files we generate are actually
// LLVM bitcode files (as used by linker LTO plugins) when compiling with
// -Clinker-plugin-lto.
// See https://github.com/rust-lang/rust/pull/50000

use std::path::PathBuf;

use run_make_support::{
    cwd, has_extension, has_prefix, llvm_ar, llvm_bcanalyzer, path, rfs, rust_lib_name, rustc,
    shallow_find_files, static_lib_name,
};

fn main() {
    check_bitcode(LibBuild {
        source: path("lib.rs"),
        crate_type: Some("staticlib"),
        output: path(static_lib_name("liblib")),
        lto: None,
        emit_obj: false,
    });
    check_bitcode(LibBuild {
        source: path("lib.rs"),
        crate_type: Some("staticlib"),
        output: path(static_lib_name("liblib-fat-lto")),
        lto: Some("fat"),
        emit_obj: false,
    });
    check_bitcode(LibBuild {
        source: path("lib.rs"),
        crate_type: Some("staticlib"),
        output: path(static_lib_name("liblib-thin-lto")),
        lto: Some("thin"),
        emit_obj: false,
    });
    check_bitcode(LibBuild {
        source: path("lib.rs"),
        crate_type: Some("rlib"),
        output: path(rust_lib_name("liblib")),
        lto: None,
        emit_obj: false,
    });
    check_bitcode(LibBuild {
        source: path("lib.rs"),
        crate_type: Some("cdylib"),
        output: path("cdylib.o"),
        lto: None,
        emit_obj: true,
    });
    check_bitcode(LibBuild {
        source: path("lib.rs"),
        crate_type: Some("dylib"),
        output: path("rdylib.o"),
        lto: None,
        emit_obj: true,
    });
    check_bitcode(LibBuild {
        source: path("main.rs"),
        crate_type: None,
        output: path("exe.o"),
        lto: None,
        emit_obj: true,
    });
}

#[track_caller]
fn check_bitcode(instructions: LibBuild) {
    let mut rustc = rustc();
    rustc
        .input(instructions.source)
        .output(&instructions.output)
        .opt_level("2")
        .codegen_units(1)
        .arg("-Clinker-plugin-lto");
    if instructions.emit_obj {
        rustc.emit("obj");
    }
    if let Some(crate_type) = instructions.crate_type {
        rustc.crate_type(crate_type);
    }
    if let Some(lto) = instructions.lto {
        rustc.arg(format!("-Clto={lto}"));
    }
    rustc.run();

    if instructions.output.extension().unwrap() != "o" {
        // Remove all potential leftover object files, then turn the output into an object file.
        for object in shallow_find_files(cwd(), |path| has_extension(path, "o")) {
            rfs::remove_file(object);
        }
        llvm_ar().extract().arg(&instructions.output).run();
    }

    let objects = shallow_find_files(cwd(), |path| {
        let mut output_path = instructions.output.clone();
        output_path.set_extension("");
        has_prefix(path, output_path.file_name().unwrap().to_str().unwrap())
            && has_extension(path, "o")
    });
    assert!(!objects.is_empty());
    println!("objects: {:#?}", objects);

    for object in objects {
        println!("reading bitcode: {}", object.display());
        // All generated object files should be LLVM bitcode files - this will fail otherwise.
        llvm_bcanalyzer().input(object).run();
    }
}

struct LibBuild {
    source: PathBuf,
    crate_type: Option<&'static str>,
    output: PathBuf,
    lto: Option<&'static str>,
    emit_obj: bool,
}
