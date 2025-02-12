//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[next] failure-status: 101
//@[next] known-bug: unknown
//@[next] normalize-stderr: "note: .*\n\n" -> ""
//@[next] normalize-stderr: "thread 'rustc' panicked.*\n.*\n" -> ""
//@[next] normalize-stderr: "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
//@[next] normalize-stderr: "delayed at .*" -> ""
//@[next] rustc-env:RUST_BACKTRACE=0
//@ check-pass

trait Super {
    type Assoc;
}

trait Sub: Super {}

impl<T: ?Sized> Super for T {
    type Assoc = i32;
}

fn illegal(x: &dyn Sub<Assoc = i32>) -> &dyn Super<Assoc = impl Sized> {
    x
}

fn main() {}
