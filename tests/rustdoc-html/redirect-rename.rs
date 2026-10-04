#![crate_name = "foo"]

mod hidden {
    //@ has foo/hidden/type.Foo.html
    //@ has - '//p/a' '../../foo/type.FooBar.html'
    pub struct Foo {}
    pub union U { a: usize }
    pub enum Empty {}
    pub const C: usize = 1;
    pub static S: usize = 1;

    //@ has foo/hidden/bar/index.html
    //@ has - '//p/a' '../../foo/baz/index.html'
    pub mod bar {
        //@ has foo/hidden/bar/type.Thing.html
        //@ has - '//p/a' '../../foo/baz/type.Thing.html'
        pub struct Thing {}
    }
}

//@ has foo/type.FooBar.html
pub use hidden::Foo as FooBar;
//@ has foo/type.FooU.html
pub use hidden::U as FooU;
//@ has foo/type.FooEmpty.html
pub use hidden::Empty as FooEmpty;
//@ has foo/constant.FooC.html
pub use hidden::C as FooC;
//@ has foo/static.FooS.html
pub use hidden::S as FooS;

//@ has foo/baz/index.html
//@ has foo/baz/type.Thing.html
pub use hidden::bar as baz;
