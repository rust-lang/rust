// Test dump-dep-graph requires query-dep-graph enabled

//@ incremental
//@ compile-flags: -Z dump-dep-graph

fn main() {}

//~? ERROR can't dump dependency graph without `-Z query-dep-graph`
