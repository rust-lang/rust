// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use build::Builder;
use build::matches::MatchPair;
use hair::*;
use repr::*;
use std::u32;

impl<H:Hair> Builder<H> {
    pub fn field_match_pairs(&mut self,
                             lvalue: Lvalue<H>,
                             subpatterns: Vec<FieldPatternRef<H>>)
                             -> Vec<MatchPair<H>> {
        subpatterns.into_iter()
                   .map(|fieldpat| {
                       let lvalue = lvalue.clone().field(fieldpat.field);
                       self.match_pair(lvalue, fieldpat.pattern)
                   })
                   .collect()
    }

    pub fn match_pair(&mut self, lvalue: Lvalue<H>, pattern: PatternRef<H>) -> MatchPair<H> {
        let pattern = self.hir.mirror(pattern);
        MatchPair::new(lvalue, pattern)
    }

    pub fn append_prefix_suffix_pairs(&mut self,
                                      match_pairs: &mut Vec<MatchPair<H>>,
                                      lvalue: Lvalue<H>,
                                      prefix: Vec<PatternRef<H>>,
                                      suffix: Vec<PatternRef<H>>)
    {
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

impl<H:Hair> MatchPair<H> {
    pub fn new(lvalue: Lvalue<H>, pattern: Pattern<H>) -> MatchPair<H> {
        MatchPair { lvalue: lvalue, pattern: pattern }
    }
}
