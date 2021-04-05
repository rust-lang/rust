// @has 'glob_shadowing/index.html'
// @count - '//tr[@class="module-item"]' 2
// @has - '//tr[@class="module-item"]' 'mod::prelude'
// @has - '//tr[@class="module-item"]' 'sub2::describe'

mod sub1 {
    /// sub1::describe
    pub fn describe() -> &'static str {
        "sub1::describe"
    }

    /// sub1::prelude
    pub mod prelude {
        pub use super::describe;
    }
}

mod sub2 {
    /// sub2::describe
    pub fn describe() -> &'static str {
        "sub2::describe"
    }
}

/// mod::prelude
pub mod prelude {
    /// mod::prelude::describe
    pub fn describe() -> &'static str {
        "mod::describe"
    }
}

#[doc(inline)]
pub use sub2::describe;

#[doc(inline)]
pub use sub1::*;

