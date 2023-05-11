pub trait Bar {}

#[doc(hidden)]
pub mod hidden {
    pub struct Foo;
}

// @has issue_33069/trait.Bar.html
// @!has - '//code' 'impl Bar for Foo'
impl Bar for hidden::Foo {}
