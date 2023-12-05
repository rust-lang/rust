// revisions: cfail1 cfail2
// should-ice
// error-pattern: delayed span bug triggered by #[rustc_error(span_delayed_bug_from_inside_query)]

#![feature(rustc_attrs)]

#[rustc_error(span_delayed_bug_from_inside_query)]
fn main() {}
