// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Operations on unique pointer types

// FIXME: this module should not exist in libcore. It must currently because the
//        Box implementation is quite ad-hoc in the compiler. Once there is
//        proper support in the compiler this type will be able to be defined in
//        its own module.

/// A value that represents the global exchange heap. This is the default
/// place that the `box` keyword allocates into when no place is supplied.
///
/// The following two examples are equivalent:
///
///     let foo = box(HEAP) Bar::new(...);
///     let foo = box Bar::new(...);
#[lang="exchange_heap"]
#[cfg(not(test))]
pub static HEAP: () = ();

#[cfg(test)]
pub static HEAP: () = ();

/// A type that represents a uniquely-owned value.
#[lang="owned_box"]
#[cfg(not(test))]
pub struct Box<T>(*T);

#[cfg(test)]
pub struct Box<T>(*T);
