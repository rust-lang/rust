// @has 'glob_shadowing/index.html'
// @count - '//tr[@class="module-item"]' 4
// @!has - '//tr[@class="module-item"]' 'sub1::describe'
// @has - '//tr[@class="module-item"]' 'mod::prelude'
// @has - '//tr[@class="module-item"]' 'sub2::describe'
// @has - '//tr[@class="module-item"]' 'sub1::Foo (struct)'
// @has - '//tr[@class="module-item"]' 'mod::Foo (function)'

// @has 'glob_shadowing/fn.describe.html'
// @has - '//div[@class="docblock"]' 'sub2::describe'

mod sub1 {
    // this should be shadowed by sub2::describe
    /// sub1::describe
    pub fn describe() -> &'static str {
        "sub1::describe"
    }

    // this should be shadowed by mod::prelude
    /// sub1::prelude
    pub mod prelude {
        pub use super::describe;
    }

    // this should not be shadowed, because sub1::Foo and mod::Foo are in different namespace
    /// sub1::Foo (struct)
    pub struct Foo;
}

mod sub2 {
    /// sub2::describe
    pub fn describe() -> &'static str {
        "sub2::describe"
    }
}

/// mod::prelude
pub mod prelude {}

/// mod::Foo (function)
pub fn Foo() {}

#[doc(inline)]
pub use sub2::describe;

#[doc(inline)]
pub use sub1::*;
