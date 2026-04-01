//@ ignore-windows (Windows does not actually strip)
//@ ignore-cross-compile (relocations in generic ELF against `arm-unknown-linux-gnueabihf`)

// Test that -Cstrip correctly strips/preserves debuginfo and symbols.

use run_make_support::{bin_name, is_darwin, llvm_dwarfdump, llvm_nm, rustc};

fn main() {
    // We use DW_ (the start of any DWARF name) to check that some debuginfo is present.
    let dwarf_indicator = "DW_";

    let test_symbol = "hey_i_get_compiled";
    let binary = &bin_name("hello");

    // Avoid checking debuginfo on darwin, because it is not actually affected by strip.
    // Darwin *never* puts debuginfo in the main binary (-Csplit-debuginfo=off just removes it),
    // so we never actually have any debuginfo in there, so we can't check that it's present.
    let do_debuginfo_check = !is_darwin();

    // Additionally, use -Cdebuginfo=2 to make the test independent of the amount of debuginfo
    // for std.

    // -Cstrip=none should preserve symbols and debuginfo.
    rustc().arg("hello.rs").arg("-Cdebuginfo=2").arg("-Cstrip=none").run();
    llvm_nm().input(binary).run().assert_stdout_contains(test_symbol);
    if do_debuginfo_check {
        llvm_dwarfdump().input(binary).run().assert_stdout_contains(dwarf_indicator);
    }

    // -Cstrip=debuginfo should preserve symbols and strip debuginfo.
    rustc().arg("hello.rs").arg("-Cdebuginfo=2").arg("-Cstrip=debuginfo").run();
    llvm_nm().input(binary).run().assert_stdout_contains(test_symbol);
    if do_debuginfo_check {
        llvm_dwarfdump().input(binary).run().assert_stdout_not_contains(dwarf_indicator);
    }

    // -Cstrip=symbols should strip symbols and strip debuginfo.
    rustc().arg("hello.rs").arg("-Cdebuginfo=2").arg("-Cstrip=symbols").run();
    llvm_nm().input(binary).run().assert_stderr_not_contains(test_symbol);
    if do_debuginfo_check {
        llvm_dwarfdump().input(binary).run().assert_stdout_not_contains(dwarf_indicator);
    }
}
