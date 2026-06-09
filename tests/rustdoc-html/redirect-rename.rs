#![crate_name = "foo"]

mod hidden {
    //@ has foo/hidden/struct.Foo.html
    //@ has - '//p/a' '../../foo/struct.FooBar.html'
    pub struct Foo {}
    pub union U { a: usize }
    pub enum Empty {}
    pub const C: usize = 1;
    pub static S: usize = 1;

    //@ has foo/hidden/bar/index.html
    //@ has - '//p/a' '../../foo/baz/index.html'
    pub mod bar {
        //@ has foo/hidden/bar/struct.Thing.html
        //@ has - '//p/a' '../../foo/baz/struct.Thing.html'
        pub struct Thing {}
    }
}

//@ has foo/struct.FooBar.html
pub use hidden::Foo as FooBar;
//@ has foo/union.FooU.html
pub use hidden::U as FooU;
//@ has foo/enum.FooEmpty.html
pub use hidden::Empty as FooEmpty;
//@ has foo/constant.FooC.html
pub use hidden::C as FooC;
//@ has foo/static.FooS.html
pub use hidden::S as FooS;

//@ has foo/baz/index.html
//@ has foo/baz/struct.Thing.html
pub use hidden::bar as baz;
