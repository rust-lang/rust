// Test that changing a tracked commandline argument invalidates
// the cache while changing an untracked one doesn't.

// revisions:rpass1 rpass2 rpass3
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

#![rustc_partition_codegened(module="commandline_args", cfg="rpass2")]
#![rustc_partition_reused(module="commandline_args", cfg="rpass3")]

// Between revisions 1 and 2, we are changing the debuginfo-level, which should
// invalidate the cache. Between revisions 2 and 3, we are adding `--verbose`
// which should have no effect on the cache:
//[rpass1] compile-flags: -C debuginfo=0
//[rpass2] compile-flags: -C debuginfo=2
//[rpass3] compile-flags: -C debuginfo=2 --verbose

pub fn main() {
    // empty
}
