// This test case makes sure that changing split-debuginfo commandline options triggers a full re-compilation.
// We only test on x86_64-unknown-linux-gnu because there all combinations split-debuginfo settings are valid
// and the test is platform-independent otherwise.

// ignore-tidy-linelength
// only-x86_64-unknown-linux-gnu
// revisions:rpass1 rpass2 rpass3 rpass4

// [rpass1]compile-flags: -Zquery-dep-graph -Csplit-debuginfo=unpacked -Zsplit-dwarf-kind=single -Zsplit-dwarf-inlining=on
// [rpass2]compile-flags: -Zquery-dep-graph -Csplit-debuginfo=packed -Zsplit-dwarf-kind=single -Zsplit-dwarf-inlining=on
// [rpass3]compile-flags: -Zquery-dep-graph -Csplit-debuginfo=packed -Zsplit-dwarf-kind=split -Zsplit-dwarf-inlining=on
// [rpass4]compile-flags: -Zquery-dep-graph -Csplit-debuginfo=packed -Zsplit-dwarf-kind=split -Zsplit-dwarf-inlining=off

#![feature(rustc_attrs)]
// For rpass2 we change -Csplit-debuginfo and thus expect every CGU to be recompiled
#![rustc_partition_codegened(module = "split_debuginfo_mode", cfg = "rpass2")]
#![rustc_partition_codegened(module = "split_debuginfo_mode-another_module", cfg = "rpass2")]
// For rpass3 we change -Zsplit-dwarf-kind and thus also expect every CGU to be recompiled
#![rustc_partition_codegened(module = "split_debuginfo_mode", cfg = "rpass3")]
#![rustc_partition_codegened(module = "split_debuginfo_mode-another_module", cfg = "rpass3")]
// For rpass4 we change -Zsplit-dwarf-inlining and thus also expect every CGU to be recompiled
#![rustc_partition_codegened(module = "split_debuginfo_mode", cfg = "rpass4")]
#![rustc_partition_codegened(module = "split_debuginfo_mode-another_module", cfg = "rpass4")]

mod another_module {
    pub fn foo() -> &'static str {
        "hello world"
    }
}

pub fn main() {
    println!("{}", another_module::foo());
}
