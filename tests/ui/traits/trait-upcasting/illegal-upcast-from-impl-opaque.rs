//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@[next] failure-status: 101
//@[next] known-bug: unknown
//@[next] normalize-stderr-test "note: .*\n\n" -> ""
//@[next] normalize-stderr-test "thread 'rustc' panicked.*\n.*\n" -> ""
//@[next] normalize-stderr-test "(error: internal compiler error: [^:]+):\d+:\d+: " -> "$1:LL:CC: "
//@[next] normalize-stderr-test "delayed at .*" -> ""
//@[next] rustc-env:RUST_BACKTRACE=0

#![feature(trait_upcasting, type_alias_impl_trait)]

trait Super {
    type Assoc;
}

trait Sub: Super {}

impl<T: ?Sized> Super for T {
    type Assoc = i32;
}

type Foo = impl Sized;

fn illegal(x: &dyn Sub<Assoc = Foo>) -> &dyn Super<Assoc = i32> {
    x //[current]~ mismatched types
}

fn main() {}
