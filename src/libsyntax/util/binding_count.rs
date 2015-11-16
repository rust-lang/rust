// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Count the number of immutable and mutable bindings in an AST.

use ast;
use visit;
use visit::Visitor;

#[derive(Copy, Clone, Default)]
pub struct BindingCount {
    pub immutable_local: usize,
    pub mutable_local: usize,
}

pub struct BindingCounter {
    count: BindingCount,
    in_local: bool,
}

impl BindingCounter {
    pub fn new() -> BindingCounter {
        BindingCounter {
            count: Default::default(),
            in_local: false,
        }
    }

    pub fn get(&self) -> BindingCount {
        self.count
    }
}

impl<'v> Visitor<'v> for BindingCounter {
    fn visit_local(&mut self, l: &ast::Local) {
        self.in_local = true;
        self.visit_pat(&l.pat);
        self.in_local = false;
    }

    fn visit_pat(&mut self, p: &ast::Pat) {
        if !self.in_local {
            return;
        }
        if let ast::PatIdent(binding_mode, _, _) = p.node {
            match binding_mode {
                ast::BindByValue(ast::MutImmutable) |
                ast::BindByRef(ast::MutImmutable) => {
                    self.count.immutable_local += 1;
                }
                ast::BindByValue(ast::MutMutable) |
                ast::BindByRef(ast::MutMutable) => {
                    self.count.mutable_local += 1;
                }
            }
        }
        visit::walk_pat(self, p);
    }

    fn visit_mac(&mut self, _: &ast::Mac) {}
}
