// issue: #116877
// revisions: sized clone
//[sized] check-pass

//[clone] known-bug: #108498
//[clone] failure-status: 101
//[clone] normalize-stderr-test: "DefId\(.*?\]::" -> "DefId("
//[clone] normalize-stderr-test: "(?m)note: .*$" -> ""
//[clone] normalize-stderr-test: "(?m)^ *\d+: .*\n" -> ""
//[clone] normalize-stderr-test: "(?m)^ *at .*\n" -> ""

#[cfg(sized)] fn rpit() -> impl Sized {}
#[cfg(clone)] fn rpit() -> impl Clone {}

fn same_output<Out>(_: impl Fn() -> Out, _: impl Fn() -> Out) {}

pub fn foo() -> impl Sized {
    same_output(rpit, foo);
    same_output(foo, rpit);
    rpit()
}

fn main () {}
