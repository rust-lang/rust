// known-bug
// build-fail
// failure-status: 101
// compile-flags:--crate-type=lib -Zmir-opt-level=3
// rustc-env:RUST_BACKTRACE=0

// normalize-stderr-test "thread 'rustc' panicked.*" -> "thread 'rustc' panicked"
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""
// normalize-stderr-test "error: internal compiler error.*" -> "error: internal compiler error"
// normalize-stderr-test "encountered.*with incompatible types:" "encountered ... with incompatible types:"
// normalize-stderr-test "note:.*unexpectedly panicked.*\n\n" -> ""
// normalize-stderr-test "note: we would appreciate a bug report.*\n\n" -> ""
// normalize-stderr-test "note: compiler flags.*\n\n" -> ""
// normalize-stderr-test "note: rustc.*running on.*\n\n" -> ""
// normalize-stderr-test "query stack during panic:\n" -> ""
// normalize-stderr-test "we're just showing a limited slice of the query stack\n" -> ""
// normalize-stderr-test "end of query stack\n" -> ""
// normalize-stderr-test "#.*\n" -> ""

// This is a known bug that @compiler-errors tried to fix in #94238,
// but the solution was probably not correct.

pub trait Factory<T> {
    type Item;
}

pub struct IntFactory;

impl<T> Factory<T> for IntFactory {
    type Item = usize;
}

pub fn foo<T>()
where
    IntFactory: Factory<T>,
{
    let mut x: <IntFactory as Factory<T>>::Item = bar::<T>();
}

#[inline]
pub fn bar<T>() -> <IntFactory as Factory<T>>::Item {
    0usize
}
