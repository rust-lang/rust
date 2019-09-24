use crate::build::Builder;
use crate::build::matches::MatchPair;
use crate::hair::*;
use rustc::mir::*;
use std::u32;
use std::convert::TryInto;

impl<'a, 'tcx> Builder<'a, 'tcx> {
    pub fn field_match_pairs<'pat>(&mut self,
                                   place: Place<'tcx>,
                                   subpatterns: &'pat [FieldPattern<'tcx>])
                                   -> Vec<MatchPair<'pat, 'tcx>> {
        subpatterns.iter()
                   .map(|fieldpat| {
                       let place = place.clone().field(fieldpat.field,
                                                       fieldpat.pattern.ty);
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
        let min_length = min_length.try_into().unwrap();

        match_pairs.extend(
            prefix.iter()
                  .enumerate()
                  .map(|(idx, subpattern)| {
                      let elem = ProjectionElem::ConstantIndex {
                          offset: idx as u32,
                          min_length,
                          from_end: false,
                      };
                      let place = place.clone().elem(elem);
                      MatchPair::new(place, subpattern)
                  })
        );

        if let Some(subslice_pat) = opt_slice {
            let subslice = place.clone().elem(ProjectionElem::Subslice {
                from: prefix.len() as u32,
                to: suffix.len() as u32
            });
            match_pairs.push(MatchPair::new(subslice, subslice_pat));
        }

        match_pairs.extend(
            suffix.iter()
                  .rev()
                  .enumerate()
                  .map(|(idx, subpattern)| {
                      let elem = ProjectionElem::ConstantIndex {
                          offset: (idx+1) as u32,
                          min_length,
                          from_end: true,
                      };
                      let place = place.clone().elem(elem);
                      MatchPair::new(place, subpattern)
                  })
        );
    }

    /// Creates a false edge to `imaginary_target` and a real edge to
    /// real_target. If `imaginary_target` is none, or is the same as the real
    /// target, a Goto is generated instead to simplify the generated MIR.
    pub fn false_edges(
        &mut self,
        from_block: BasicBlock,
        real_target: BasicBlock,
        imaginary_target: Option<BasicBlock>,
        source_info: SourceInfo,
    )  {
        match imaginary_target {
            Some(target) if target != real_target => {
                self.cfg.terminate(
                    from_block,
                    source_info,
                    TerminatorKind::FalseEdges {
                        real_target,
                        imaginary_target: target,
                    },
                );
            }
            _ => {
                self.cfg.terminate(
                    from_block,
                    source_info,
                    TerminatorKind::Goto {
                        target: real_target
                    }
                );
            }
        }
    }
}

impl<'pat, 'tcx> MatchPair<'pat, 'tcx> {
    pub fn new(place: Place<'tcx>, pattern: &'pat Pattern<'tcx>) -> MatchPair<'pat, 'tcx> {
        MatchPair {
            place,
            pattern,
        }
    }
}
