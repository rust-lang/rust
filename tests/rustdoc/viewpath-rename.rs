#![crate_name = "foo"]

pub mod io {
    pub trait Reader { fn dummy(&self) { } }
}

pub enum Maybe<A> {
    Just(A),
    Nothing
}

// @has foo/prelude/index.html
pub mod prelude {
    // @has foo/prelude/index.html '//code' 'pub use io as FooIo;'
    // @has foo/prelude/index.html '//code' 'pub use io::Reader as FooReader;'
    #[doc(no_inline)] pub use io::{self as FooIo, Reader as FooReader};
    // @has foo/prelude/index.html '//code' 'pub use Maybe;'
    // @has foo/prelude/index.html '//code' 'pub use Maybe::Just as MaybeJust;'
    // @has foo/prelude/index.html '//code' 'pub use Maybe::Nothing;'
    #[doc(no_inline)] pub use Maybe::{self, Just as MaybeJust, Nothing};
}
