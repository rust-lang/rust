//@ only-msvc
// Reason: this is testing the MSVC linker

// This is a regression test for #129020
// It ensures we can link a rust staticlib into DLL using the MSVC linker

use run_make_support::run::cmd;
use run_make_support::rustc;

fn main() {
    // Build the staticlib
    rustc().crate_type("staticlib").input("static.rs").output("static.lib").run();
    // Create an empty obj file (using the C compiler)
    // Then use it to link in the staticlib using `/WHOLEARCHIVE` and produce a DLL.
    // We call link.exe directly because this test is only for MSVC and not their LLVM equivalents
    cmd("cl.exe").args(&["c.c", "-nologo", "-c", "-MT"]).run();
    cmd("link.exe")
        .args(&["c", "/WHOLEARCHIVE:./static.lib", "/dll", "/def:dll.def", "/out:dll.dll"])
        // Import libs
        .args(&["libcmt.lib", "kernel32.lib", "ws2_32.lib", "ntdll.lib", "userenv.lib"])
        .run();
}
