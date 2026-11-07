#![crate_name = "foo"]

//@ has foo/struct.Foo.html
//@ has - '//*[@class="sidebar-elems"]//section//a' 'super_long_name'
//@ has - '//*[@class="sidebar-elems"]//section//a' 'Disp'
pub struct Foo(usize);

impl Foo {
    pub fn super_long_name() {}
}

pub trait Disp {
    fn disp_trait_method();
}

impl Disp for Foo {
    fn disp_trait_method() {}
}
