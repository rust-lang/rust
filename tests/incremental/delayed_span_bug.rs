//@ revisions: cfail1 cfail2
//@ failure-status: 101

#![feature(rustc_attrs)]

#[rustc_delayed_bug_from_inside_query]
fn main() {} //~ ERROR delayed bug triggered by #[rustc_delayed_bug_from_inside_query]
