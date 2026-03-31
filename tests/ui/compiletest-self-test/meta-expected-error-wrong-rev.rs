//@ ignore-compare-mode-polonius

//@ revisions: a
//@ unused-revision-names: b
//@ should-fail

// This is a "meta-test" of the compilertest framework itself.  In
// particular, it includes the right error message, but the message
// targets the wrong revision, so we expect the execution to fail.
// See also `meta-expected-error-correct-rev.rs`.

#[cfg(a)]
fn foo() {
    let x: u32 = 22_usize; //[b]~ ERROR mismatched types
}

fn main() { }
