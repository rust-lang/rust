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

The 4 kinds are

* Copy - types that may be copied without allocation. This includes
  scalar types and managed pointers, and exludes owned pointers. It
  also excludes types that implement `Drop`.

* Send - owned types and types containing owned types.  These types
  may be transferred across task boundaries.

* Const - types that are deeply immutable. Const types are used for
  freezable data structures.

* Durable - types that do not contain borrowed pointers.

`Copy` types include both implicitly copyable types that the compiler
will copy automatically and non-implicitly copyable types that require
the `copy` keyword to copy. Types that do not implement `Copy` may
instead implement `Clone`.

*/

#[lang="copy"]
pub trait Copy {
    // Empty.
}

#[lang="send"]
pub trait Send {
    // Empty.
}

#[lang="const"]
pub trait Const {
    // Empty.
}

#[cfg(stage0)]
#[lang="owned"]
pub trait Durable {
    // Empty.
}

#[cfg(stage1)]
#[cfg(stage2)]
#[cfg(stage3)]
#[lang="durable"]
pub trait Durable {
    // Empty.
}
