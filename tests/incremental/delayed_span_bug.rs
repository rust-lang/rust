//@ revisions: cfail1 cfail2
//@ should-ice
//@ error-pattern: delayed bug triggered by #[rustc_delayed_bug_from_inside_query]

#![feature(rustc_attrs)]

#[rustc_delayed_bug_from_inside_query]
fn main() {}
