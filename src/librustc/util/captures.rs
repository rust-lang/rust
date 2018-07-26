// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// "Signaling" trait used in impl trait to tag lifetimes that you may
/// need to capture but don't really need for other reasons.
/// Basically a workaround; see [this comment] for details.
///
/// [this comment]: https://github.com/rust-lang/rust/issues/34511#issuecomment-373423999
pub trait Captures<'a> { }

impl<'a, T: ?Sized> Captures<'a> for T { }
