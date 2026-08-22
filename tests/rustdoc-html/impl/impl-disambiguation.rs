#![crate_name = "foo"]

pub trait Foo {}

pub struct Bar<T> { field: T }

//@ has foo/trait.Foo.html '//*[@id="implementors-list"]//h3[@class="code-header"]' \
//     "impl Foo for Bar<u8>"
impl Foo for Bar<u8> {}
//@ has foo/trait.Foo.html '//*[@id="implementors-list"]//h3[@class="code-header"]' \
//     "impl Foo for Bar<u16>"
impl Foo for Bar<u16> {}
//@ has foo/trait.Foo.html '//*[@id="implementors-list"]//h3[@class="code-header"]' \
//     "impl<'a> Foo for &'a Bar<u8>"
impl<'a> Foo for &'a Bar<u8> {}

pub mod mod1 {
    pub struct Baz {}
}

pub mod mod2 {
    pub enum Baz {}
}

//@ has foo/trait.Foo.html '//*[@id="implementors-list"]//h3[@class="code-header"]' \
//     "impl Foo for foo::mod1::Baz"
impl Foo for mod1::Baz {}
//@ has foo/trait.Foo.html '//*[@id="implementors-list"]//h3[@class="code-header"]' \
//     "impl<'a> Foo for &'a foo::mod2::Baz"
impl<'a> Foo for &'a mod2::Baz {}

pub mod ops {
    pub struct Range<T>(pub T);
}

pub mod range {
    pub struct Range<T>(pub T);
}

// Regression test for https://github.com/rust-lang/rust/issues/154960.
//@ has foo/range/struct.Range.html '//section[@class="impl"]//h3[@class="code-header"]' \
//     "impl<T> From<foo::ops::Range<T>> for foo::range::Range<T>"
impl<T> From<ops::Range<T>> for range::Range<T> {
    fn from(range: ops::Range<T>) -> Self {
        Self(range.0)
    }
}

pub trait HasAssoc {
    type Assoc;
}

pub trait ProjectionBound {}

pub struct Projected<T>(pub T);

// Regression test for https://github.com/rust-lang/rust/issues/154960.
//@ has foo/struct.Projected.html '//section[@class="impl"]//h3[@class="code-header"]' \
//     "impl<T> From<T> for Projected<T>where T: HasAssoc, T::Assoc: ProjectionBound,"
impl<T> From<T> for Projected<T>
where
    T: HasAssoc,
    T::Assoc: ProjectionBound,
{
    fn from(value: T) -> Self {
        Self(value)
    }
}
