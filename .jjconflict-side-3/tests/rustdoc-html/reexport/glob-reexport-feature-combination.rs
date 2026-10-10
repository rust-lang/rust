// This test ensures that `cfg`s are correctly propagated for re-export chains
// where the outermost is a glob re-export.

#![crate_name = "foo"]
#![feature(doc_cfg)]

//@ has 'foo/index.html'
//@ count - '//*[@class="item-table"]/dt' 3

//@ has 'foo/index.html'
//@ has - '//dt/a[@title="struct foo::A"]/../*[@class="stab portability"]' 'Non-bar and non-foo'

//@ has 'foo/struct.A.html'
//@ has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//    'Available on non-crate feature bar and non-crate feature foo only.'

mod a {
    mod inner {
        pub struct A {}
    }
    #[cfg(not(feature = "bar"))]
    pub use self::inner::A;
}
#[cfg(not(feature = "foo"))]
pub use a::*;

//@ has 'foo/index.html'
//@ has - '//dt/a[@title="struct foo::B"]/../*[@class="stab portability"]' 'Non-bar and non-baz and non-foo'

//@ has 'foo/struct.B.html'
//@ has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//    'Available on non-crate feature bar and non-crate feature baz and non-crate feature foo only.'

mod b {
    mod inner {
        mod innermost {
            pub struct B {}
        }
        #[cfg(not(feature = "baz"))]
        pub use self::innermost::B;
    }
    #[cfg(not(feature = "bar"))]
    pub use self::inner::*;
}
#[cfg(not(feature = "foo"))]
pub use b::*;

//@ has 'foo/index.html'
//@ has - '//dt/a[@title="struct foo::C"]/../*[@class="stab portability"]' 'Non-bar and non-baz and non-foo'

//@ has 'foo/struct.C.html'
//@ has - '//*[@id="main-content"]/*[@class="item-info"]/*[@class="stab portability"]' \
//    'Available on non-crate feature bar and non-crate feature baz and non-crate feature foo only.'

mod c {
    mod inner {
        mod innermost {
            #[cfg(not(feature = "baz"))]
            pub struct C {}
        }
        pub use self::innermost::*;
    }
    #[cfg(not(feature = "bar"))]
    pub use self::inner::*;
}
#[cfg(not(feature = "foo"))]
pub use c::*;
