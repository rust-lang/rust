// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

/// Tracks undoable actions.
pub trait UndoableTracker {
    type Undoable: ?Sized + Undoable;
    fn push_action(&mut self, id: usize, undoer: <Self::Undoable as Undoable>::Undoer);
}

/// The 'undo' part of an undoable action that may be tracked by an UndoableTracker.
pub trait Undoer {
    type Undoable: ?Sized + Undoable;
    fn undo(self, item: &mut Self::Undoable);
}

/// Type that is contractually obligated to push undoable actions onto any (single) registered
/// tracker.
pub trait Undoable {
    type Undoer: Undoer<Undoable=Self> + Debug;
    type Tracker: UndoableTracker<Undoable=Self>;
    // We use an Rc here due to it being difficult to represent a reference-to-owner safely in the
    // type system.
    fn register_tracker(&mut self, tracker: Rc<RefCell<Self::Tracker>>, id: usize);
}
