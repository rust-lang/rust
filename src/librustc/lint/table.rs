// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use syntax::ast;
use syntax_pos::MultiSpan;
use util::nodemap::NodeMap;

use super::{Lint, LintId, EarlyLint, IntoEarlyLint};

#[derive(RustcEncodable, RustcDecodable)]
pub struct LintTable {
    map: NodeMap<Vec<EarlyLint>>
}

impl LintTable {
    pub fn new() -> Self {
        LintTable { map: NodeMap() }
    }

    pub fn add_lint<S: Into<MultiSpan>>(&mut self,
                                        lint: &'static Lint,
                                        id: ast::NodeId,
                                        sp: S,
                                        msg: String)
    {
        self.add_lint_diagnostic(lint, id, (sp, &msg[..]))
    }

    pub fn add_lint_diagnostic<M>(&mut self,
                                  lint: &'static Lint,
                                  id: ast::NodeId,
                                  msg: M)
        where M: IntoEarlyLint,
    {
        let lint_id = LintId::of(lint);
        let early_lint = msg.into_early_lint(lint_id);
        let arr = self.map.entry(id).or_insert(vec![]);
        if !arr.contains(&early_lint) {
            arr.push(early_lint);
        }
    }

    pub fn get(&self, id: ast::NodeId) -> &[EarlyLint] {
        self.map.get(&id).map(|v| &v[..]).unwrap_or(&[])
    }

    pub fn take(&mut self, id: ast::NodeId) -> Vec<EarlyLint> {
        self.map.remove(&id).unwrap_or(vec![])
    }

    pub fn transfer(&mut self, into: &mut LintTable) {
        into.map.extend(self.map.drain());
    }

    /// Returns the first (id, lint) pair that is non-empty. Used to
    /// implement a sanity check in lints that all node-ids are
    /// visited.
    pub fn get_any(&self) -> Option<(&ast::NodeId, &Vec<EarlyLint>)> {
        self.map.iter()
                .filter(|&(_, v)| !v.is_empty())
                .next()
    }
}

