// aux-build: reexports.rs

#![crate_name = "foo"]

extern crate reexports;

// @has 'foo/macro.addr_of.html' '//*[@class="docblock item-decl"]' 'pub macro addr_of($place : expr) {'
pub use reexports::addr_of;
// @!has 'foo/macro.addr_of_crate.html'
pub(crate) use reexports::addr_of_crate;
// @!has 'foo/macro.addr_of_self.html'
pub(self) use reexports::addr_of_self;

// @has 'foo/struct.Foo.html' '//*[@class="docblock item-decl"]' 'pub struct Foo;'
pub use reexports::Foo;
// @!has 'foo/struct.FooCrate.html'
pub(crate) use reexports::FooCrate;
// @!has 'foo/struct.FooSelf.html'
pub(self) use reexports::FooSelf;

// @has 'foo/enum.Bar.html' '//*[@class="docblock item-decl"]' 'pub enum Bar {'
pub use reexports::Bar;
// @!has 'foo/enum.BarCrate.html'
pub(crate) use reexports::BarCrate;
// @!has 'foo/enum.BarSelf.html'
pub(self) use reexports::BarSelf;

// @has 'foo/fn.foo.html' '//*[@class="rust fn"]' 'pub fn foo()'
pub use reexports::foo;
// @!has 'foo/fn.foo_crate.html'
pub(crate) use reexports::foo_crate;
// @!has 'foo/fn.foo_self.html'
pub(self) use reexports::foo_self;

// @has 'foo/type.Type.html' '//*[@class="rust typedef"]' 'pub type Type ='
pub use reexports::Type;
// @!has 'foo/type.TypeCrate.html'
pub(crate) use reexports::TypeCrate;
// @!has 'foo/type.TypeSelf.html'
pub(self) use reexports::TypeSelf;

// @has 'foo/union.Union.html' '//*[@class="docblock item-decl"]' 'pub union Union {'
pub use reexports::Union;
// @!has 'foo/union.UnionCrate.html'
pub(crate) use reexports::UnionCrate;
// @!has 'foo/union.UnionSelf.html'
pub(self) use reexports::UnionSelf;
