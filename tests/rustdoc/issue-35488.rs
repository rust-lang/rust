mod foo {
    pub enum Foo {
        Bar,
    }
    pub use self::Foo::*;
}

// @has 'issue_35488/index.html' '//code' 'pub use self::Foo::*;'
// @has 'issue_35488/enum.Foo.html'
pub use self::foo::*;

// @has 'issue_35488/index.html' '//code' 'pub use std::option::Option::None;'
pub use std::option::Option::None;
