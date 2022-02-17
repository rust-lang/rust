#![crate_name = "foo"]


pub trait Foo<T> {
    fn foo() {}
}

pub struct Bar;

// @has foo/struct.Bar.html
// @has - '//*[@class="sidebar-elems"]//section//a[@href="#impl-Foo%3Cunsafe%20extern%20%22C%22%20fn()%3E"]' 'Foo<unsafe extern "C" fn()>'
impl Foo<unsafe extern "C" fn()> for Bar {}
