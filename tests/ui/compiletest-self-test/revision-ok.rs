// Meta test for compiletest: check that when we give the right error
// patterns, the test passes. See all `revision-bad.rs`.

//@ run-fail
//@ revisions: foo bar
//@[foo] error-pattern:foo
//@[bar] error-pattern:bar
//@ needs-subprocess

#[cfg(foo)]
fn die() {
    panic!("foo");
}
#[cfg(bar)]
fn die() {
    panic!("bar");
}

fn main() {
    die();
}
