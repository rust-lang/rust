#![crate_name="bar"]

pub trait Bar {}
pub struct Foo;

impl<'a> Bar for &'a char {}
impl Bar for Foo {}
