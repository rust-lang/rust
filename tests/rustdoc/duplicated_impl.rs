// This test ensures that the same implementation doesn't show more than once.
// It's a regression test for https://github.com/rust-lang/rust/issues/96036.

#![crate_name = "foo"]

// We check that there is only one "impl<T> Something<Whatever> for T" listed in the
// blanket implementations.

// @has 'foo/struct.Whatever.html'
// @count - '//*[@id="blanket-implementations-list"]/section[@class="impl"]' 1

pub trait Something<T> { }
pub struct Whatever;
impl<T> Something<Whatever> for T {}
