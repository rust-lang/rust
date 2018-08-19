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
use rustc::mir::*;
use std::u32;

impl<'a, 'gcx, 'tcx> Builder<'a, 'gcx, 'tcx> {
    pub fn field_match_pairs<'pat>(&mut self,
                                   place: Place<'tcx>,
                                   subpatterns: &'pat [FieldPattern<'tcx>])
                                   -> Vec<MatchPair<'pat, 'tcx>> {
        subpatterns.iter()
                   .map(|fieldpat| {
                       let place = place.clone().field(
                           self.hir.tcx(),
                           fieldpat.field,
                           fieldpat.pattern.ty
                       );
                       MatchPair::new(place, &fieldpat.pattern)
                   })
                   .collect()
    }

    pub fn prefix_slice_suffix<'pat>(&mut self,
                                     match_pairs: &mut Vec<MatchPair<'pat, 'tcx>>,
                                     place: &Place<'tcx>,
                                     prefix: &'pat [Pattern<'tcx>],
                                     opt_slice: Option<&'pat Pattern<'tcx>>,
                                     suffix: &'pat [Pattern<'tcx>]) {
        let min_length = prefix.len() + suffix.len();
        assert!(min_length < u32::MAX as usize);
        let min_length = min_length as u32;

        match_pairs.extend(
            prefix.iter()
                  .enumerate()
                  .map(|(idx, subpattern)| {
                      let place = place.clone().constant_index(
                          self.hir.tcx(),
                          idx as u32,
                          min_length,
                          false,
                      );
                      MatchPair::new(place, subpattern)
                  })
        );

        if let Some(subslice_pat) = opt_slice {
            let place = place.clone().subslice(
                self.hir.tcx(),
                prefix.len() as u32,
                suffix.len() as u32,
            );
            match_pairs.push(MatchPair::new(place, subslice_pat));
        }

        match_pairs.extend(
            suffix.iter()
                  .rev()
                  .enumerate()
                  .map(|(idx, subpattern)| {
                      let place = place.clone().constant_index(
                          self.hir.tcx(),
                          (idx + 1) as u32,
                          min_length,
                          true,
                      );
                      MatchPair::new(place, subpattern)
                  })
        );
    }
}

impl<'pat, 'tcx> MatchPair<'pat, 'tcx> {
    pub fn new(place: Place<'tcx>, pattern: &'pat Pattern<'tcx>) -> MatchPair<'pat, 'tcx> {
        MatchPair {
            place,
            pattern,
            slice_len_checked: false,
        }
    }
}
