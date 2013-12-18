// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!
The kind traits

Rust types can be classified in various useful ways according to
intrinsic properties of the type. These classifications, often called
'kinds', are represented as traits.

They cannot be implemented by user code, but are instead implemented
by the compiler automatically for the types to which they apply.

*/

/// Types able to be transferred across task boundaries.
#[lang="send"]
pub trait Send {
    // empty.
}

/// Types that are either immutable or have inherited mutability.
#[lang="freeze"]
pub trait Freeze {
    // empty.
}

/// Types with a constant size known at compile-time.
#[lang="sized"]
pub trait Sized {
    // Empty.
}

/// Types that can be copied by simply copying bits (i.e. `memcpy`).
///
/// The name "POD" stands for "Plain Old Data" and is borrowed from C++.
#[lang="pod"]
pub trait Pod {
    // Empty.
}

