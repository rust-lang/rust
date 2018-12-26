#![crate_name = "foo"]

// @has foo/trait.Foo.html
// @has - '//*[@class="sidebar-title"][@href="#required-methods"]' 'Required Methods'
// @has - '//*[@class="sidebar-links"]/a' 'bar'
// @has - '//*[@class="sidebar-title"][@href="#provided-methods"]' 'Provided Methods'
// @has - '//*[@class="sidebar-links"]/a' 'foo'
// @has - '//*[@class="sidebar-title"][@href="#associated-const"]' 'Associated Constants'
// @has - '//*[@class="sidebar-links"]/a' 'BAR'
// @has - '//*[@class="sidebar-title"][@href="#associated-types"]' 'Associated Types'
// @has - '//*[@class="sidebar-links"]/a' 'Output'
pub trait Foo {
    const BAR: u32 = 0;
    type Output: ?Sized;

    fn foo() {}
    fn bar() -> Self::Output;
}

// @has foo/struct.Bar.html
// @has - '//*[@class="sidebar-title"][@href="#fields"]' 'Fields'
// @has - '//*[@class="sidebar-links"]/a[@href="#structfield.f"]' 'f'
// @has - '//*[@class="sidebar-links"]/a[@href="#structfield.u"]' 'u'
// @!has - '//*[@class="sidebar-links"]/a' 'waza'
pub struct Bar {
    pub f: u32,
    pub u: u32,
    waza: u32,
}

// @has foo/enum.En.html
// @has - '//*[@class="sidebar-title"][@href="#variants"]' 'Variants'
// @has - '//*[@class="sidebar-links"]/a' 'foo'
// @has - '//*[@class="sidebar-links"]/a' 'bar'
pub enum En {
    foo,
    bar,
}

// @has foo/union.MyUnion.html
// @has - '//*[@class="sidebar-title"][@href="#fields"]' 'Fields'
// @has - '//*[@class="sidebar-links"]/a[@href="#structfield.f1"]' 'f1'
// @has - '//*[@class="sidebar-links"]/a[@href="#structfield.f2"]' 'f2'
// @!has - '//*[@class="sidebar-links"]/a' 'waza'
pub union MyUnion {
    pub f1: u32,
    pub f2: f32,
    waza: u32,
}
