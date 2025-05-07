#![feature(associated_type_defaults)]
#![crate_name = "foo"]

//@ has foo/trait.Foo.html
//@ has - '//div[@class="sidebar-elems"]//h3/a[@href="#required-methods"]' 'Required Methods'
//@ has - '//*[@class="sidebar-elems"]//section//a' 'bar'
//@ has - '//div[@class="sidebar-elems"]//h3/a[@href="#provided-methods"]' 'Provided Methods'
//@ has - '//*[@class="sidebar-elems"]//section//a' 'foo'
//@ has - '//div[@class="sidebar-elems"]//h3/a[@href="#required-associated-consts"]' 'Required Associated Constants'
//@ has - '//*[@class="sidebar-elems"]//section//a' 'FOO'
//@ has - '//div[@class="sidebar-elems"]//h3/a[@href="#provided-associated-consts"]' 'Provided Associated Constants'
//@ has - '//*[@class="sidebar-elems"]//section//a' 'BAR'
//@ has - '//div[@class="sidebar-elems"]//h3/a[@href="#required-associated-types"]' 'Required Associated Types'
//@ has - '//*[@class="sidebar-elems"]//section//a' 'Output'
//@ has - '//div[@class="sidebar-elems"]//h3/a[@href="#provided-associated-types"]' 'Provided Associated Types'
//@ has - '//*[@class="sidebar-elems"]//section//a' 'Extra'
//@ has - '//div[@class="sidebar-elems"]//h3/a[@href="#dyn-compatibility"]' 'Dyn Compatibility'
pub trait Foo {
    const FOO: usize;
    const BAR: u32 = 0;
    type Extra: Copy = ();
    type Output: ?Sized;

    fn foo() {}
    fn bar() -> Self::Output;
}

//@ has foo/trait.DynCompatible.html
//@ !has - '//div[@class="sidebar-elems"]//h3/a[@href="#dyn-compatibility"]' ''
pub trait DynCompatible {
    fn access(&self);
}

//@ has foo/struct.Bar.html
//@ has - '//div[@class="sidebar-elems"]//h3/a[@href="#fields"]' 'Fields'
//@ has - '//*[@class="sidebar-elems"]//section//a[@href="#structfield.f"]' 'f'
//@ has - '//*[@class="sidebar-elems"]//section//a[@href="#structfield.u"]' 'u'
//@ !has - '//*[@class="sidebar-elems"]//section//a' 'waza'
pub struct Bar {
    pub f: u32,
    pub u: u32,
    waza: u32,
}

//@ has foo/enum.En.html
//@ has - '//div[@class="sidebar-elems"]//h3/a[@href="#variants"]' 'Variants'
//@ has - '//*[@class="sidebar-elems"]//section//a' 'Foo'
//@ has - '//*[@class="sidebar-elems"]//section//a' 'Bar'
pub enum En {
    Foo,
    Bar,
}

//@ has foo/union.MyUnion.html
//@ has - '//div[@class="sidebar-elems"]//h3/a[@href="#fields"]' 'Fields'
//@ has - '//*[@class="sidebar-elems"]//section//a[@href="#structfield.f1"]' 'f1'
//@ has - '//*[@class="sidebar-elems"]//section//a[@href="#structfield.f2"]' 'f2'
//@ !has - '//*[@class="sidebar-elems"]//section//a' 'waza'
pub union MyUnion {
    pub f1: u32,
    pub f2: f32,
    waza: u32,
}
