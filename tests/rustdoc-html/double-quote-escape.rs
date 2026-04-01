#![crate_name = "foo"]

pub trait Foo<T> {
    fn foo() {}
}

pub struct Bar;

//@ has foo/struct.Bar.html
//@ has - '//*[@class="sidebar-elems"]//section//a[@href="#impl-Foo%3Cunsafe+extern+%22C%22+fn()%3E-for-Bar"]' 'Foo<unsafe extern "C" fn()>'
impl Foo<unsafe extern "C" fn()> for Bar {}
