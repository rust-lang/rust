// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
