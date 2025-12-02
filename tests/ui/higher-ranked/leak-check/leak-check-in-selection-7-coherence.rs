//@ check-pass

// cc #119820
struct W<T, U>(T, U);

trait Trait<T> {}
// using this impl results in a higher-ranked region error.
impl<'a> Trait<W<&'a str, &'a str>> for () {}
impl<'a> Trait<W<&'a str, String>> for () {}

trait NotString {}
impl NotString for &str {}
impl NotString for u32 {}


trait Overlap<U> {}
impl<T: for<'a> Trait<W<&'a str, U>>, U> Overlap<U> for T {}
impl<U: NotString> Overlap<U> for () {}

fn main() {}
