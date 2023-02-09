// known-bug: unknown
// compile-flags: -Ztrait-solver=next
// failure-status: 101
// normalize-stderr-test "note: .*\n\n" -> ""
// normalize-stderr-test "thread 'rustc' panicked.*\n" -> ""
// rustc-env:RUST_BACKTRACE=0

// This tests checks that we update results in the provisional cache when
// we pop a goal from the stack.
#![feature(auto_traits)]
auto trait Coinductive {}
struct Foo<T>(T);
struct Bar<T>(T);

impl<T> Coinductive for Foo<T>
where
    Bar<T>: Coinductive
{}

impl<T> Coinductive for Bar<T>
where
    Foo<T>: Coinductive,
    Bar<T>: ConstrainInfer,
{}

trait ConstrainInfer {}
impl ConstrainInfer for Bar<u8> {}
impl ConstrainInfer for Foo<u16> {}

fn impls<T: Coinductive>() -> T { todo!() }

fn constrain<T: ConstrainInfer>(_: T) {}

fn main() {
    // This should constrain `_` to `u8`.
    impls::<Foo<_>>();
}
