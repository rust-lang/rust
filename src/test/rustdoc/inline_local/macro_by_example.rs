// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


/// docs for foo
#[deprecated(since = "1.2.3", note = "text")]
#[macro_export]
macro_rules! foo {
    ($($tt:tt)*) => {}
}

// @has macro_by_example/macros/index.html
pub mod macros {
    // @!has - 'pub use foo as bar;'
    // @has macro_by_example/macros/macro.bar.html
    // @has - '//*[@class="docblock"]' 'docs for foo'
    // @has - '//*[@class="stab deprecated"]' 'Deprecated since 1.2.3: text'
    // @has - '//a/@href' 'macro_by_example.rs.html#15-17'
    #[doc(inline)]
    pub use foo as bar;
}
