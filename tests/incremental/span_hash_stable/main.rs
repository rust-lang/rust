// This test makes sure that it doesn't make a difference in which order we are
// adding source files to the source_map. The order affects the BytePos values of
// the spans and this test makes sure that we handle them correctly by hashing
// file:line:column instead of raw byte offset.

//@ revisions:rpass1 rpass2
//@ compile-flags: -g -Z query-dep-graph

#![feature(rustc_attrs)]

mod auxiliary;

fn main() {
    let _ = auxiliary::sub1::SomeType {
        x: 0,
        y: 1,
    };

    let _ = auxiliary::sub2::SomeOtherType {
        a: 2,
        b: 3,
    };
}
