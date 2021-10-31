// aux-build: reexports.rs
// compile-flags: --document-private-items

#![crate_name = "foo"]

extern crate reexports;

// @has 'foo/macro.addr_of.html' '//*[@class="docblock item-decl"]' 'pub macro addr_of($place : expr) {'
pub use reexports::addr_of;
// @has 'foo/macro.addr_of_crate.html' '//*[@class="docblock item-decl"]' 'pub(crate) macro addr_of_crate($place : expr) {'
pub(crate) use reexports::addr_of_crate;
// @has 'foo/macro.addr_of_self.html' '//*[@class="docblock item-decl"]' 'pub(crate) macro addr_of_self($place : expr) {'
pub(self) use reexports::addr_of_self;

// @has 'foo/struct.Foo.html' '//*[@class="docblock item-decl"]' 'pub struct Foo;'
pub use reexports::Foo;
// @has 'foo/struct.FooCrate.html' '//*[@class="docblock item-decl"]' 'pub(crate) struct FooCrate;'
pub(crate) use reexports::FooCrate;
// @has 'foo/struct.FooSelf.html' '//*[@class="docblock item-decl"]' 'pub(crate) struct FooSelf;'
pub(self) use reexports::FooSelf;

// @has 'foo/enum.Bar.html' '//*[@class="docblock item-decl"]' 'pub enum Bar {'
pub use reexports::Bar;
// @has 'foo/enum.BarCrate.html' '//*[@class="docblock item-decl"]' 'pub(crate) enum BarCrate {'
pub(crate) use reexports::BarCrate;
// @has 'foo/enum.BarSelf.html' '//*[@class="docblock item-decl"]' 'pub(crate) enum BarSelf {'
pub(self) use reexports::BarSelf;

// @has 'foo/fn.foo.html' '//*[@class="rust fn"]' 'pub fn foo()'
pub use reexports::foo;
// @has 'foo/fn.foo_crate.html' '//*[@class="rust fn"]' 'pub(crate) fn foo_crate()'
pub(crate) use reexports::foo_crate;
// @has 'foo/fn.foo_self.html' '//*[@class="rust fn"]' 'pub(crate) fn foo_self()'
pub(self) use reexports::foo_self;

// @has 'foo/type.Type.html' '//*[@class="rust typedef"]' 'pub type Type ='
pub use reexports::Type;
// @has 'foo/type.TypeCrate.html' '//*[@class="rust typedef"]' 'pub(crate) type TypeCrate ='
pub(crate) use reexports::TypeCrate;
// @has 'foo/type.TypeSelf.html' '//*[@class="rust typedef"]' 'pub(crate) type TypeSelf ='
pub(self) use reexports::TypeSelf;

// @has 'foo/union.Union.html' '//*[@class="docblock item-decl"]' 'pub union Union {'
pub use reexports::Union;
// @has 'foo/union.UnionCrate.html' '//*[@class="docblock item-decl"]' 'pub(crate) union UnionCrate {'
pub(crate) use reexports::UnionCrate;
// @has 'foo/union.UnionSelf.html' '//*[@class="docblock item-decl"]' 'pub(crate) union UnionSelf {'
pub(self) use reexports::UnionSelf;

pub mod foo {
    // @!has 'foo/foo/union.Union.html'
    use crate::reexports::Union;
}
