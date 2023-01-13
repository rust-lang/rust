// Check that compiling with packed Split DWARF twice succeeds. This should confirm that DWARF
// objects are cached as work products and available to the incremental compilation for `thorin` to
// pack into a DWARF package.

// ignore-tidy-linelength
// only-x86_64-unknown-linux-gnu
// revisions:rpass1 rpass2

// [rpass1]compile-flags: -g -Zquery-dep-graph -Csplit-debuginfo=packed -Zsplit-dwarf-kind=split
// [rpass2]compile-flags: -g -Zquery-dep-graph -Csplit-debuginfo=packed -Zsplit-dwarf-kind=split

#![feature(rustc_attrs)]
// For `rpass2`, nothing has changed so everything should re-used.
#![rustc_partition_reused(module = "split_debuginfo_cached", cfg = "rpass2")]
#![rustc_partition_reused(module = "split_debuginfo_cached-another_module", cfg = "rpass2")]

mod another_module {
    pub fn foo() -> &'static str {
        "hello world"
    }
}

pub fn main() {
    println!("{}", another_module::foo());
}
