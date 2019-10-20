// Meta test for compiletest: check that when we give the right error
// patterns, the test passes. See all `meta-revision-bad.rs`.

// revisions: foo bar
// run-fail
//[foo] error-pattern:foo
//[bar] error-pattern:bar

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
