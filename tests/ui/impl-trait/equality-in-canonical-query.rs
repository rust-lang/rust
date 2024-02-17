// issue: #116877
//@ revisions: sized clone
//@[sized] check-pass
//@[clone] known-bug: #108498
//@[clone] failure-status: 101
//@[clone] normalize-stderr-test: "DefId\(.*?\]::" -> "DefId("
//@[clone] normalize-stderr-test: "(?m)note: we would appreciate a bug report.*\n\n" -> ""
//@[clone] normalize-stderr-test: "(?m)note: rustc.*running on.*\n\n" -> ""
//@[clone] normalize-stderr-test: "(?m)note: compiler flags.*\n\n" -> ""
//@[clone] normalize-stderr-test: "(?m)note: delayed at.*$" -> ""
//@[clone] normalize-stderr-test: "(?m)^ *\d+: .*\n" -> ""
//@[clone] normalize-stderr-test: "(?m)^ *at .*\n" -> ""

#[cfg(sized)] fn rpit() -> impl Sized {}
#[cfg(clone)] fn rpit() -> impl Clone {}

fn same_output<Out>(_: impl Fn() -> Out, _: impl Fn() -> Out) {}

pub fn foo() -> impl Sized {
    same_output(rpit, foo);
    same_output(foo, rpit);
    rpit()
}

fn main () {}
