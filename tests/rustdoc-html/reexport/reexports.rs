//@ aux-build: reexports.rs

#![crate_name = "foo"]

extern crate reexports;

//@ has 'foo/macro.addr_of.html' '//*[@class="rust item-decl"]' 'pub macro addr_of($place:expr) {'
pub use reexports::addr_of;
//@ !has 'foo/macro.addr_of_crate.html'
pub(crate) use reexports::addr_of_crate;
//@ !has 'foo/macro.addr_of_self.html'
pub(self) use reexports::addr_of_self;
//@ !has 'foo/macro.addr_of_local.html'
use reexports::addr_of_local;

//@ has 'foo/type.Foo.html' '//*[@class="rust item-decl"]' 'pub struct Foo;'
pub use reexports::Foo;
//@ !has 'foo/type.FooCrate.html'
pub(crate) use reexports::FooCrate;
//@ !has 'foo/type.FooSelf.html'
pub(self) use reexports::FooSelf;
//@ !has 'foo/type.FooLocal.html'
use reexports::FooLocal;

//@ has 'foo/type.Bar.html' '//*[@class="rust item-decl"]' 'pub enum Bar {'
pub use reexports::Bar;
//@ !has 'foo/type.BarCrate.html'
pub(crate) use reexports::BarCrate;
//@ !has 'foo/type.BarSelf.html'
pub(self) use reexports::BarSelf;
//@ !has 'foo/type.BarLocal.html'
use reexports::BarLocal;

//@ has 'foo/fn.foo.html' '//pre[@class="rust item-decl"]' 'pub fn foo()'
pub use reexports::foo;
//@ !has 'foo/fn.foo_crate.html'
pub(crate) use reexports::foo_crate;
//@ !has 'foo/fn.foo_self.html'
pub(self) use reexports::foo_self;
//@ !has 'foo/fn.foo_local.html'
use reexports::foo_local;

//@ has 'foo/type.Type.html' '//pre[@class="rust item-decl"]' 'pub type Type ='
pub use reexports::Type;
//@ !has 'foo/type.TypeCrate.html'
pub(crate) use reexports::TypeCrate;
//@ !has 'foo/type.TypeSelf.html'
pub(self) use reexports::TypeSelf;
//@ !has 'foo/type.TypeLocal.html'
use reexports::TypeLocal;

//@ has 'foo/type.Union.html' '//*[@class="rust item-decl"]' 'pub union Union {'
pub use reexports::Union;
//@ !has 'foo/type.UnionCrate.html'
pub(crate) use reexports::UnionCrate;
//@ !has 'foo/type.UnionSelf.html'
pub(self) use reexports::UnionSelf;
//@ !has 'foo/type.UnionLocal.html'
use reexports::UnionLocal;

pub mod outer {
    pub mod inner {
        //@ has 'foo/outer/inner/macro.addr_of.html' '//*[@class="rust item-decl"]' 'pub macro addr_of($place:expr) {'
        pub use reexports::addr_of;
        //@ !has 'foo/outer/inner/macro.addr_of_crate.html'
        pub(crate) use reexports::addr_of_crate;
        //@ !has 'foo/outer/inner/macro.addr_of_super.html'
        pub(super) use reexports::addr_of_super;
        //@ !has 'foo/outer/inner/macro.addr_of_self.html'
        pub(self) use reexports::addr_of_self;
        //@ !has 'foo/outer/inner/macro.addr_of_local.html'
        use reexports::addr_of_local;

        //@ has 'foo/outer/inner/type.Foo.html' '//*[@class="rust item-decl"]' 'pub struct Foo;'
        pub use reexports::Foo;
        //@ !has 'foo/outer/inner/type.FooCrate.html'
        pub(crate) use reexports::FooCrate;
        //@ !has 'foo/outer/inner/type.FooSuper.html'
        pub(super) use reexports::FooSuper;
        //@ !has 'foo/outer/inner/type.FooSelf.html'
        pub(self) use reexports::FooSelf;
        //@ !has 'foo/outer/inner/type.FooLocal.html'
        use reexports::FooLocal;

        //@ has 'foo/outer/inner/type.Bar.html' '//*[@class="rust item-decl"]' 'pub enum Bar {'
        pub use reexports::Bar;
        //@ !has 'foo/outer/inner/type.BarCrate.html'
        pub(crate) use reexports::BarCrate;
        //@ !has 'foo/outer/inner/type.BarSuper.html'
        pub(super) use reexports::BarSuper;
        //@ !has 'foo/outer/inner/type.BarSelf.html'
        pub(self) use reexports::BarSelf;
        //@ !has 'foo/outer/inner/type.BarLocal.html'
        use reexports::BarLocal;

        //@ has 'foo/outer/inner/fn.foo.html' '//pre[@class="rust item-decl"]' 'pub fn foo()'
        pub use reexports::foo;
        //@ !has 'foo/outer/inner/fn.foo_crate.html'
        pub(crate) use reexports::foo_crate;
        //@ !has 'foo/outer/inner/fn.foo_super.html'
        pub(super) use::reexports::foo_super;
        //@ !has 'foo/outer/inner/fn.foo_self.html'
        pub(self) use reexports::foo_self;
        //@ !has 'foo/outer/inner/fn.foo_local.html'
        use reexports::foo_local;

        //@ has 'foo/outer/inner/type.Type.html' '//pre[@class="rust item-decl"]' 'pub type Type ='
        pub use reexports::Type;
        //@ !has 'foo/outer/inner/type.TypeCrate.html'
        pub(crate) use reexports::TypeCrate;
        //@ !has 'foo/outer/inner/type.TypeSuper.html'
        pub(super) use reexports::TypeSuper;
        //@ !has 'foo/outer/inner/type.TypeSelf.html'
        pub(self) use reexports::TypeSelf;
        //@ !has 'foo/outer/inner/type.TypeLocal.html'
        use reexports::TypeLocal;

        //@ has 'foo/outer/inner/type.Union.html' '//*[@class="rust item-decl"]' 'pub union Union {'
        pub use reexports::Union;
        //@ !has 'foo/outer/inner/type.UnionCrate.html'
        pub(crate) use reexports::UnionCrate;
        //@ !has 'foo/outer/inner/type.UnionSuper.html'
        pub(super) use reexports::UnionSuper;
        //@ !has 'foo/outer/inner/type.UnionSelf.html'
        pub(self) use reexports::UnionSelf;
        //@ !has 'foo/outer/inner/type.UnionLocal.html'
        use reexports::UnionLocal;
    }
}
