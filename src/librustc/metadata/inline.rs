// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use middle::def_id::DefId;
use syntax::ast;
use syntax::ast_util::{IdRange, IdRangeComputingVisitor,
                       IdVisitor, IdVisitingOperation};
use syntax::ptr::P;
use syntax::visit::Visitor;
use self::InlinedItem::*;

/// The data we save and restore about an inlined item or method.  This is not
/// part of the AST that we parse from a file, but it becomes part of the tree
/// that we trans.
#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Debug)]
pub enum InlinedItem {
    Item(P<ast::Item>),
    TraitItem(DefId /* impl id */, P<ast::TraitItem>),
    ImplItem(DefId /* impl id */, P<ast::ImplItem>),
    Foreign(P<ast::ForeignItem>),
}

/// A borrowed version of `ast::InlinedItem`.
pub enum InlinedItemRef<'a> {
    Item(&'a ast::Item),
    TraitItem(DefId, &'a ast::TraitItem),
    ImplItem(DefId, &'a ast::ImplItem),
    Foreign(&'a ast::ForeignItem)
}

impl InlinedItem {
    pub fn visit<'ast,V>(&'ast self, visitor: &mut V)
        where V: Visitor<'ast>
    {
        match *self {
            Item(ref i) => visitor.visit_item(&**i),
            Foreign(ref i) => visitor.visit_foreign_item(&**i),
            TraitItem(_, ref ti) => visitor.visit_trait_item(ti),
            ImplItem(_, ref ii) => visitor.visit_impl_item(ii),
        }
    }

    pub fn visit_ids<O: IdVisitingOperation>(&self, operation: &mut O) {
        let mut id_visitor = IdVisitor {
            operation: operation,
            pass_through_items: true,
            visited_outermost: false,
        };
        self.visit(&mut id_visitor);
    }

    pub fn compute_id_range(&self) -> IdRange {
        let mut visitor = IdRangeComputingVisitor::new();
        self.visit_ids(&mut visitor);
        visitor.result()
    }
}

