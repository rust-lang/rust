// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::ty;
use syntax::ast::{CrateNum, NodeId};
use std::fmt;

#[derive(Clone, Eq, Ord, PartialOrd, PartialEq, RustcEncodable,
           RustcDecodable, Hash, Copy)]
pub struct DefId {
    pub krate: CrateNum,
    pub node: NodeId,
}

impl fmt::Debug for DefId {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        try!(write!(f, "DefId {{ krate: {}, node: {}",
                    self.krate, self.node));

        // Unfortunately, there seems to be no way to attempt to print
        // a path for a def-id, so I'll just make a best effort for now
        // and otherwise fallback to just printing the crate/node pair
        try!(ty::tls::with_opt(|opt_tcx| {
            if let Some(tcx) = opt_tcx {
                try!(write!(f, " => {}", tcx.item_path_str(*self)));
            }
            Ok(())
        }));

        write!(f, " }}")
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

