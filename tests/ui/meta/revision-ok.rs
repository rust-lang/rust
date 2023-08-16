// Meta test for compiletest: check that when we give the right error
// patterns, the test passes. See all `revision-bad.rs`.

// run-fail
//@revisions: foo bar
//@[foo] error-in-other-file:foo
//@[bar] error-in-other-file:bar
//@ignore-target-emscripten no processes

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
