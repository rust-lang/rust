// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use build::{BlockAnd, Builder};
use build::matches::MatchPair;
use hair::*;
use repr::*;
use std::u32;

impl<'a,'tcx> Builder<'a,'tcx> {
    pub fn field_match_pairs(&mut self,
                             lvalue: Lvalue<'tcx>,
                             subpatterns: Vec<FieldPatternRef<'tcx>>)
                             -> Vec<MatchPair<'tcx>> {
        subpatterns.into_iter()
                   .map(|fieldpat| {
                       let lvalue = lvalue.clone().field(fieldpat.field);
                       self.match_pair(lvalue, fieldpat.pattern)
                   })
                   .collect()
    }

    pub fn match_pair(&mut self,
                      lvalue: Lvalue<'tcx>,
                      pattern: PatternRef<'tcx>)
                      -> MatchPair<'tcx> {
        let pattern = self.hir.mirror(pattern);
        MatchPair::new(lvalue, pattern)
    }

    /// When processing an array/slice pattern like `lv @ [x, y, ..s, z]`,
    /// this function converts the prefix (`x`, `y`) and suffix (`z`) into
    /// distinct match pairs:
    ///
    ///     lv[0 of 3] @ x  // see ProjectionElem::ConstantIndex (and its Debug impl)
    ///     lv[1 of 3] @ y  // to explain the `[x of y]` notation
    ///     lv[-1 of 3] @ z
    ///
    /// If a slice like `s` is present, then the function also creates
    /// a temporary like:
    ///
    ///     tmp0 = lv[2..-1] // using the special Rvalue::Slice
    ///
    /// and creates a match pair `tmp0 @ s`
    pub fn prefix_suffix_slice(&mut self,
                               match_pairs: &mut Vec<MatchPair<'tcx>>,
                               block: BasicBlock,
                               lvalue: Lvalue<'tcx>,
                               prefix: Vec<PatternRef<'tcx>>,
                               opt_slice: Option<PatternRef<'tcx>>,
                               suffix: Vec<PatternRef<'tcx>>)
                               -> BlockAnd<()> {
        // If there is a `..P` pattern, create a temporary `t0` for
        // the slice and then a match pair `t0 @ P`:
        if let Some(slice) = opt_slice {
            let slice = self.hir.mirror(slice);
            let prefix_len = prefix.len();
            let suffix_len = suffix.len();
            let rvalue = Rvalue::Slice {
                input: lvalue.clone(),
                from_start: prefix_len,
                from_end: suffix_len,
            };
            let temp = self.temp(slice.ty.clone()); // no need to schedule drop, temp is always copy
            self.cfg.push_assign(block, slice.span, &temp, rvalue);
            match_pairs.push(MatchPair::new(temp, slice));
        }

        self.prefix_suffix(match_pairs, lvalue, prefix, suffix);

        block.unit()
    }

    /// Helper for `prefix_suffix_slice` which just processes the prefix and suffix.
    fn prefix_suffix(&mut self,
                     match_pairs: &mut Vec<MatchPair<'tcx>>,
                     lvalue: Lvalue<'tcx>,
                     prefix: Vec<PatternRef<'tcx>>,
                     suffix: Vec<PatternRef<'tcx>>) {
        let min_length = prefix.len() + suffix.len();
        assert!(min_length < u32::MAX as usize);
        let min_length = min_length as u32;

        let prefix_pairs: Vec<_> =
            prefix.into_iter()
                  .enumerate()
                  .map(|(idx, subpattern)| {
                      let elem = ProjectionElem::ConstantIndex {
                          offset: idx as u32,
                          min_length: min_length,
                          from_end: false,
                      };
                      let lvalue = lvalue.clone().elem(elem);
                      self.match_pair(lvalue, subpattern)
                  })
                  .collect();

        let suffix_pairs: Vec<_> =
            suffix.into_iter()
                  .rev()
                  .enumerate()
                  .map(|(idx, subpattern)| {
                      let elem = ProjectionElem::ConstantIndex {
                          offset: (idx+1) as u32,
                          min_length: min_length,
                          from_end: true,
                      };
                      let lvalue = lvalue.clone().elem(elem);
                      self.match_pair(lvalue, subpattern)
                  })
                  .collect();

        match_pairs.extend(prefix_pairs.into_iter().chain(suffix_pairs));
    }
}

impl<'tcx> MatchPair<'tcx> {
    pub fn new(lvalue: Lvalue<'tcx>, pattern: Pattern<'tcx>) -> MatchPair<'tcx> {
        MatchPair {
            lvalue: lvalue,
            pattern: pattern,
        }
    }
}
