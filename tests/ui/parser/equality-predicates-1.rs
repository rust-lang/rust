// Test that we syntactically reject *equality predicates*.
//
// It's a feature that was originally proposed as part of accepted RFC 135 (2014). It's never been
// fully implemented and in the meantime design & implementation concerns have been raised.
// Between Feb 2017 and Jun 2026 we accidentally accepted such predicates syntactically
// (indeed, without any pre-expansion feature gate).
//
// We *might* add a more restricted version of this feature to the language in the future.
// See discussions in tracking issue <https://github.com/rust-lang/rust/issues/20041> for details.

#[cfg(false)]
fn f<T: Iterator>(mut xs: T) -> Option<u8>
where
    <T as Iterator>::Item == u8
    //~^ ERROR general type equality constraints are not supported
{
    xs.next()
}

fn main() {}
