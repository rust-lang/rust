// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use driver::session::Session;

use syntax::ast;
use syntax::ast_map;

struct NodeIdAssigner {
    sess: Session
}

impl ast_map::FoldOps for NodeIdAssigner {
    fn new_id(&self, old_id: ast::NodeId) -> ast::NodeId {
        assert_eq!(old_id, ast::DUMMY_NODE_ID);
        self.sess.next_node_id()
    }
}

pub fn assign_node_ids_and_map(sess: Session, crate: ast::Crate) -> (ast::Crate, ast_map::Map) {
    ast_map::map_crate(sess.diagnostic(), crate, NodeIdAssigner { sess: sess })
}
