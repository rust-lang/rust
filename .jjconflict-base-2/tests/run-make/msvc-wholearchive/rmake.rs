//! This is a regression test for #129020
//! It ensures we can use `/WHOLEARCHIVE` to link a rust staticlib into DLL
//! using the MSVC linker

//@ only-msvc
// Reason: this is testing the MSVC linker

use std::path::PathBuf;

use run_make_support::{cc, cmd, env_var, extra_linker_flags, rustc};

fn main() {
    // Build the staticlib
    rustc().crate_type("staticlib").input("static.rs").output("static.lib").run();
    // Build an empty object to pass to the linker.
    cc().input("c.c").output("c.obj").args(["-c"]).run();

    // Find the C toolchain's linker.
    let mut linker = PathBuf::from(env_var("CC"));
    let linker_flavour = if linker.file_stem().is_some_and(|s| s == "cl") {
        linker.set_file_name("link.exe");
        "msvc"
    } else if linker.file_stem().is_some_and(|s| s == "clang-cl") {
        linker.set_file_name("lld-link.exe");
        "llvm"
    } else {
        panic!("unknown C toolchain");
    };

    // As a sanity check, make sure this works without /WHOLEARCHIVE.
    // Otherwise the actual test failure may be caused by something else.
    cmd(&linker)
        .args(["c.obj", "./static.lib", "-dll", "-def:dll.def", "-out:dll.dll"])
        .args(extra_linker_flags())
        .run();

    // FIXME(@ChrisDenton): this doesn't currently work with llvm's lld-link for other reasons.
    // May need LLVM patches.
    if linker_flavour == "msvc" {
        // Link in the staticlib using `/WHOLEARCHIVE` and produce a DLL.
        cmd(&linker)
            .args([
                "c.obj",
                "-WHOLEARCHIVE:./static.lib",
                "-dll",
                "-def:dll.def",
                "-out:dll_whole_archive.dll",
            ])
            .args(extra_linker_flags())
            .run();
    }
}
