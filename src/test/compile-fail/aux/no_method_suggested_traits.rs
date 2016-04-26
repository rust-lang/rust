// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub use reexport::Reexported;

pub struct Foo;
pub enum Bar { X }

pub mod foo {
    pub trait PubPub {
        fn method(&self) {}

        fn method3(&self) {}
    }

    impl PubPub for u32 {}
    impl PubPub for i32 {}
}
pub mod bar {
    trait PubPriv {
        fn method(&self);
    }
}
mod qux {
    pub trait PrivPub {
        fn method(&self);
    }
}
mod quz {
    trait PrivPriv {
        fn method(&self);
    }
}

mod reexport {
    pub trait Reexported {
        fn method(&self);
    }
}
