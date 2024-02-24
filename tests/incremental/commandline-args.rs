// Test that changing a tracked commandline argument invalidates
// the cache while changing an untracked one doesn't.

//@ revisions:rpass1 rpass2 rpass3 rpass4
//@ compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

#![rustc_partition_codegened(module="commandline_args", cfg="rpass2")]
#![rustc_partition_reused(module="commandline_args", cfg="rpass3")]
#![rustc_partition_codegened(module="commandline_args", cfg="rpass4")]

// Between revisions 1 and 2, we are changing the debuginfo-level, which should
// invalidate the cache. Between revisions 2 and 3, we are adding `--diagnostic-width`
// which should have no effect on the cache. Between revisions, we are adding
// `--remap-path-prefix` which should invalidate the cache:
//@[rpass1] compile-flags: -C debuginfo=0
//@[rpass2] compile-flags: -C debuginfo=2
//@[rpass3] compile-flags: -C debuginfo=2 --diagnostic-width=80
//@[rpass4] compile-flags: -C debuginfo=2 --diagnostic-width=80 --remap-path-prefix=/home/bors/r=src

pub fn main() {
    // empty
}
