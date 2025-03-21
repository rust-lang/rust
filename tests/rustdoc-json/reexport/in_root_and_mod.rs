//@ !has "$.index[*].name" '"foo"'
mod foo {
    //@ has "$.index[*].name" '"Foo"'
    pub struct Foo;
}

//@ has "$.index[*].inner.use.source" '"foo::Foo"'
pub use foo::Foo;

pub mod bar {
    //@ has "$.index[*].inner.use.source" '"crate::foo::Foo"'
    pub use crate::foo::Foo;
}
