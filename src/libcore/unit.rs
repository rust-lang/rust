// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use iter::FromIterator;

/// Collapses all unit items from an iterator into one.
///
/// This is more useful when combined with higher-level abstractions, like
/// collecting to a `Result<(), E>` where you only care about errors:
///
/// ```
/// use std::io::*;
/// let data = vec![1, 2, 3, 4, 5];
/// let res: Result<()> = data.iter()
///     .map(|x| writeln!(stdout(), "{}", x))
///     .collect();
/// assert!(res.is_ok());
/// ```
#[stable(feature = "unit_from_iter", since = "1.23.0")]
impl FromIterator<()> for () {
    fn from_iter<I: IntoIterator<Item=()>>(iter: I) -> Self {
        iter.into_iter().for_each(|()| {})
    }
}
