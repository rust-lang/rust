// This test makes sure that we detect changed feature gates.

// revisions:rpass1 cfail2
// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]
#![cfg_attr(rpass1, feature(nll))]

fn main() {
    let mut v = vec![1];
    v.push(v[0]);
    //[cfail2]~^ ERROR cannot borrow
}
