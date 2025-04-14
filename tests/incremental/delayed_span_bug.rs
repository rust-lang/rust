//@ revisions: cfail1 cfail2
//@ should-ice

#![feature(rustc_attrs)]

#[rustc_delayed_bug_from_inside_query]
fn main() {} //~ ERROR delayed bug triggered by #[rustc_delayed_bug_from_inside_query]
