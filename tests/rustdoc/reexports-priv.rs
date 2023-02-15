// aux-build: reexports.rs
// compile-flags: --document-private-items

#![crate_name = "foo"]

extern crate reexports;

// @has 'foo/macro.addr_of.html' '//*[@class="rust item-decl"]' 'pub macro addr_of($place:expr) {'
pub use reexports::addr_of;
// @!has 'foo/macro.addr_of_crate.html'
pub(crate) use reexports::addr_of_crate;
// @!has 'foo/macro.addr_of_self.html'
pub(self) use reexports::addr_of_self;
// @!has 'foo/macro.addr_of_local.html'
use reexports::addr_of_local;

// @has 'foo/struct.Foo.html' '//*[@class="rust item-decl"]' 'pub struct Foo;'
pub use reexports::Foo;
// @!has 'foo/struct.FooCrate.html'
pub(crate) use reexports::FooCrate;
// @!has 'foo/struct.FooSelf.html'
pub(self) use reexports::FooSelf;
// @!has 'foo/struct.FooLocal.html'
use reexports::FooLocal;

// @has 'foo/enum.Bar.html' '//*[@class="rust item-decl"]' 'pub enum Bar {'
pub use reexports::Bar;
// @!has 'foo/enum.BarCrate.html'
pub(crate) use reexports::BarCrate;
// @!has 'foo/enum.BarSelf.html'
pub(self) use reexports::BarSelf;
// @!has 'foo/enum.BarLocal.html'
use reexports::BarLocal;

// @has 'foo/fn.foo.html' '//pre[@class="rust item-decl"]' 'pub fn foo()'
pub use reexports::foo;
// @!has 'foo/fn.foo_crate.html'
pub(crate) use reexports::foo_crate;
// @!has 'foo/fn.foo_self.html'
pub(self) use reexports::foo_self;
// @!has 'foo/fn.foo_local.html'
use reexports::foo_local;

// @has 'foo/type.Type.html' '//pre[@class="rust item-decl"]' 'pub type Type ='
pub use reexports::Type;
// @!has 'foo/type.TypeCrate.html'
pub(crate) use reexports::TypeCrate;
// @!has 'foo/type.TypeSelf.html'
pub(self) use reexports::TypeSelf;
// @!has 'foo/type.TypeLocal.html'
use reexports::TypeLocal;

// @has 'foo/union.Union.html' '//*[@class="rust item-decl"]' 'pub union Union {'
pub use reexports::Union;
// @!has 'foo/union.UnionCrate.html'
pub(crate) use reexports::UnionCrate;
// @!has 'foo/union.UnionSelf.html'
pub(self) use reexports::UnionSelf;
// @!has 'foo/union.UnionLocal.html'
use reexports::UnionLocal;

pub mod outer {
    pub mod inner {
        // @has 'foo/outer/inner/macro.addr_of.html' '//*[@class="rust item-decl"]' 'pub macro addr_of($place:expr) {'
        pub use reexports::addr_of;
        // @has 'foo/outer/inner/macro.addr_of_crate.html' '//*[@class="rust item-decl"]' 'pub(crate) macro addr_of_crate($place:expr) {'
        pub(crate) use reexports::addr_of_crate;
        // @has 'foo/outer/inner/macro.addr_of_super.html' '//*[@class="rust item-decl"]' 'pub(in outer) macro addr_of_super($place:expr) {'
        pub(super) use reexports::addr_of_super;
        // @!has 'foo/outer/inner/macro.addr_of_self.html'
        pub(self) use reexports::addr_of_self;
        // @!has 'foo/outer/inner/macro.addr_of_local.html'
        use reexports::addr_of_local;

        // @has 'foo/outer/inner/struct.Foo.html' '//*[@class="rust item-decl"]' 'pub struct Foo;'
        pub use reexports::Foo;
        // @has 'foo/outer/inner/struct.FooCrate.html' '//*[@class="rust item-decl"]' 'pub(crate) struct FooCrate;'
        pub(crate) use reexports::FooCrate;
        // @has 'foo/outer/inner/struct.FooSuper.html' '//*[@class="rust item-decl"]' 'pub(in outer) struct FooSuper;'
        pub(super) use reexports::FooSuper;
        // @!has 'foo/outer/inner/struct.FooSelf.html'
        pub(self) use reexports::FooSelf;
        // @!has 'foo/outer/inner/struct.FooLocal.html'
        use reexports::FooLocal;

        // @has 'foo/outer/inner/enum.Bar.html' '//*[@class="rust item-decl"]' 'pub enum Bar {'
        pub use reexports::Bar;
        // @has 'foo/outer/inner/enum.BarCrate.html' '//*[@class="rust item-decl"]' 'pub(crate) enum BarCrate {'
        pub(crate) use reexports::BarCrate;
        // @has 'foo/outer/inner/enum.BarSuper.html' '//*[@class="rust item-decl"]' 'pub(in outer) enum BarSuper {'
        pub(super) use reexports::BarSuper;
        // @!has 'foo/outer/inner/enum.BarSelf.html'
        pub(self) use reexports::BarSelf;
        // @!has 'foo/outer/inner/enum.BarLocal.html'
        use reexports::BarLocal;

        // @has 'foo/outer/inner/fn.foo.html' '//pre[@class="rust item-decl"]' 'pub fn foo()'
        pub use reexports::foo;
        // @has 'foo/outer/inner/fn.foo_crate.html' '//pre[@class="rust item-decl"]' 'pub(crate) fn foo_crate()'
        pub(crate) use reexports::foo_crate;
        // @has 'foo/outer/inner/fn.foo_super.html' '//pre[@class="rust item-decl"]' 'pub(in outer) fn foo_super( )'
        pub(super) use::reexports::foo_super;
        // @!has 'foo/outer/inner/fn.foo_self.html'
        pub(self) use reexports::foo_self;
        // @!has 'foo/outer/inner/fn.foo_local.html'
        use reexports::foo_local;

        // @has 'foo/outer/inner/type.Type.html' '//pre[@class="rust item-decl"]' 'pub type Type ='
        pub use reexports::Type;
        // @has 'foo/outer/inner/type.TypeCrate.html' '//pre[@class="rust item-decl"]' 'pub(crate) type TypeCrate ='
        pub(crate) use reexports::TypeCrate;
        // @has 'foo/outer/inner/type.TypeSuper.html' '//pre[@class="rust item-decl"]' 'pub(in outer) type TypeSuper ='
        pub(super) use reexports::TypeSuper;
        // @!has 'foo/outer/inner/type.TypeSelf.html'
        pub(self) use reexports::TypeSelf;
        // @!has 'foo/outer/inner/type.TypeLocal.html'
        use reexports::TypeLocal;

        // @has 'foo/outer/inner/union.Union.html' '//*[@class="rust item-decl"]' 'pub union Union {'
        pub use reexports::Union;
        // @has 'foo/outer/inner/union.UnionCrate.html' '//*[@class="rust item-decl"]' 'pub(crate) union UnionCrate {'
        pub(crate) use reexports::UnionCrate;
        // @has 'foo/outer/inner/union.UnionSuper.html' '//*[@class="rust item-decl"]' 'pub(in outer) union UnionSuper {'
        pub(super) use reexports::UnionSuper;
        // @!has 'foo/outer/inner/union.UnionSelf.html'
        pub(self) use reexports::UnionSelf;
        // @!has 'foo/outer/inner/union.UnionLocal.html'
        use reexports::UnionLocal;
    }
}

mod re_re_exports {
        // @!has 'foo/re_re_exports/union.Union.html'
        use crate::reexports::Union;
}
