// Regression test for #35593. Check that we can reuse this trivially
// equal example.

// revisions:rpass1 rpass2
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![rustc_partition_reused(module="issue_35593", cfg="rpass2")]

fn main() {
    println!("hello world");
}
