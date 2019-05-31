struct Baz<U> where U: Eq(U); //This is parsed as the new Fn* style parenthesis syntax.
struct Baz<U> where U: Eq(U) -> R; // Notice this parses as well.
struct Baz<U>(U) where U: Eq; // This rightfully signals no error as well.
struct Foo<T> where T: Copy, (T); //~ ERROR expected one of `:`, `==`, or `=`, found `;`

fn main() {}
