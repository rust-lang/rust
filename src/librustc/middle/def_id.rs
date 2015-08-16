// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast::{CrateNum, NodeId};
use std::cell::Cell;
use std::fmt;

#[derive(Clone, Eq, Ord, PartialOrd, PartialEq, RustcEncodable,
           RustcDecodable, Hash, Copy)]
pub struct DefId {
    pub krate: CrateNum,
    pub node: NodeId,
}

fn default_def_id_debug(_: DefId, _: &mut fmt::Formatter) -> fmt::Result { Ok(()) }

thread_local!(pub static DEF_ID_DEBUG: Cell<fn(DefId, &mut fmt::Formatter) -> fmt::Result> =
                Cell::new(default_def_id_debug));

impl fmt::Debug for DefId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "DefId {{ krate: {}, node: {} }}",
                    self.krate, self.node));
        DEF_ID_DEBUG.with(|def_id_debug| def_id_debug.get()(*self, f))
    }
}

impl DefId {
    pub fn local(id: NodeId) -> DefId {
        DefId { krate: LOCAL_CRATE, node: id }
    }

    /// Read the node id, asserting that this def-id is krate-local.
    pub fn local_id(&self) -> NodeId {
        assert_eq!(self.krate, LOCAL_CRATE);
        self.node
    }

    pub fn is_local(&self) -> bool {
        self.krate == LOCAL_CRATE
    }
}


/// Item definitions in the currently-compiled crate would have the CrateNum
/// LOCAL_CRATE in their DefId.
pub const LOCAL_CRATE: CrateNum = 0;

