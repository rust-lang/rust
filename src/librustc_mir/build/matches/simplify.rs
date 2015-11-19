// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simplifying Candidates
//!
//! *Simplifying* a match pair `lvalue @ pattern` means breaking it down
//! into bindings or other, simpler match pairs. For example:
//!
//! - `lvalue @ (P1, P2)` can be simplified to `[lvalue.0 @ P1, lvalue.1 @ P2]`
//! - `lvalue @ x` can be simplified to `[]` by binding `x` to `lvalue`
//!
//! The `simplify_candidate` routine just repeatedly applies these
//! sort of simplifications until there is nothing left to
//! simplify. Match pairs cannot be simplified if they require some
//! sort of test: for example, testing which variant an enum is, or
//! testing a value against a constant.

use build::{BlockAnd, BlockAndExtension, Builder};
use build::matches::{Binding, MatchPair, Candidate};
use hair::*;
use rustc::mir::repr::*;

use std::mem;

impl<'a,'tcx> Builder<'a,'tcx> {
    pub fn simplify_candidate<'pat>(&mut self,
                                    mut block: BasicBlock,
                                    candidate: &mut Candidate<'pat, 'tcx>)
                                    -> BlockAnd<()> {
        // repeatedly simplify match pairs until fixed point is reached
        loop {
            let match_pairs = mem::replace(&mut candidate.match_pairs, vec![]);
            let mut progress = match_pairs.len(); // count how many were simplified
            for match_pair in match_pairs {
                match self.simplify_match_pair(block, match_pair, candidate) {
                    Ok(b) => {
                        block = b;
                    }
                    Err(match_pair) => {
                        candidate.match_pairs.push(match_pair);
                        progress -= 1; // this one was not simplified
                    }
                }
            }
            if progress == 0 {
                return block.unit(); // if we were not able to simplify any, done.
            }
        }
    }

    /// Tries to simplify `match_pair`, returning true if
    /// successful. If successful, new match pairs and bindings will
    /// have been pushed into the candidate. If no simplification is
    /// possible, Err is returned and no changes are made to
    /// candidate.
    fn simplify_match_pair<'pat>(&mut self,
                                 mut block: BasicBlock,
                                 match_pair: MatchPair<'pat, 'tcx>,
                                 candidate: &mut Candidate<'pat, 'tcx>)
                                 -> Result<BasicBlock, MatchPair<'pat, 'tcx>> {
        match *match_pair.pattern.kind {
            PatternKind::Wild => {
                // nothing left to do
                Ok(block)
            }

            PatternKind::Binding { name, mutability, mode, var, ty, ref subpattern } => {
                candidate.bindings.push(Binding {
                    name: name,
                    mutability: mutability,
                    span: match_pair.pattern.span,
                    source: match_pair.lvalue.clone(),
                    var_id: var,
                    var_ty: ty,
                    binding_mode: mode,
                });

                if let Some(subpattern) = subpattern.as_ref() {
                    // this is the `x @ P` case; have to keep matching against `P` now
                    candidate.match_pairs.push(MatchPair::new(match_pair.lvalue, subpattern));
                }

                Ok(block)
            }

            PatternKind::Constant { .. } => {
                // FIXME normalize patterns when possible
                Err(match_pair)
            }

            PatternKind::Array { ref prefix, ref slice, ref suffix } => {
                unpack!(block = self.prefix_suffix_slice(&mut candidate.match_pairs,
                                                         block,
                                                         match_pair.lvalue.clone(),
                                                         prefix,
                                                         slice.as_ref(),
                                                         suffix));
                Ok(block)
            }

            PatternKind::Slice { .. } |
            PatternKind::Range { .. } |
            PatternKind::Variant { .. } => {
                // cannot simplify, test is required
                Err(match_pair)
            }

            PatternKind::Leaf { ref subpatterns } => {
                // tuple struct, match subpats (if any)
                candidate.match_pairs
                         .extend(self.field_match_pairs(match_pair.lvalue, subpatterns));
                Ok(block)
            }

            PatternKind::Deref { ref subpattern } => {
                let lvalue = match_pair.lvalue.deref();
                candidate.match_pairs.push(MatchPair::new(lvalue, subpattern));
                Ok(block)
            }
        }
    }
}
