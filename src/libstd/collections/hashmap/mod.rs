// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Unordered containers, implemented as hash-tables

pub use self::map::HashMap;
pub use self::map::Entries;
pub use self::map::MutEntries;
pub use self::map::MoveEntries;
pub use self::map::Keys;
pub use self::map::Values;
pub use self::map::INITIAL_CAPACITY;
pub use self::set::HashSet;
pub use self::set::SetItems;
pub use self::set::SetMoveItems;
pub use self::set::SetAlgebraItems;

mod bench;
mod map;
mod set;
mod table;
