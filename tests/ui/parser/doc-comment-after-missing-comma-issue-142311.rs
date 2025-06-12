//! Check that the parser does not suggest converting `///` to a regular comment
//! when it appears after a missing comma in an item list (e.g. `enum` variants).
//!
//! Related issue
//! - https://github.com/rust-lang/rust/issues/142311

enum Foo {
    /// Like the noise a sheep makes
    Bar
    /// Like where people drink
    //~^ ERROR found a documentation comment that doesn't document anything [E0585]
    Baa///xxxxxx
    //~^ ERROR found a documentation comment that doesn't document anything [E0585]
    Baz///xxxxxx
    //~^ ERROR found a documentation comment that doesn't document anything [E0585]
}

fn main() {}
