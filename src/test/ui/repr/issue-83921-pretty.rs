// Regression test for #83921. A `delay_span_bug()` call was issued, but the
// error was never reported because the pass responsible for detecting and
// reporting the error does not run in certain modes of pretty-printing.

// Make sure the error is reported if we do not just pretty-print:
// revisions: pretty normal
// [pretty]compile-flags: -Zunpretty=everybody_loops
// [pretty]check-pass

#[repr("C")]
//[normal]~^ ERROR: meta item in `repr` must be an identifier [E0565]
struct A {}

fn main() {}
