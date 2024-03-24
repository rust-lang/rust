#![feature(rustc_attrs)]

#[rustc_error(delayed_bug_from_inside_query)]
fn main() {}
