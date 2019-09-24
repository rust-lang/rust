// build-pass (FIXME(62277): could be check-pass?)
// compile-flags:--cfg set1

#![cfg_attr(set1, feature(rustc_attrs))]
#![rustc_dummy]

fn main() {}
