//! Contains utilities for generating suggestions for borrowck errors related to unsatisfied
//! outlives constraints.

#![allow(rustc::diagnostic_outside_of_impl)]
#![allow(rustc::untranslatable_diagnostic)]

use std::collections::BTreeMap;

use rustc_data_structures::fx::FxIndexSet;
use rustc_errors::Diag;
use rustc_middle::ty::RegionVid;
use smallvec::SmallVec;
use tracing::debug;

use super::{ErrorConstraintInfo, RegionName, RegionNameSource};
use crate::MirBorrowckCtxt;

/// The different things we could suggest.
enum SuggestedConstraint {
    /// Outlives(a, [b, c, d, ...]) => 'a: 'b + 'c + 'd + ...
    Outlives(RegionName, SmallVec<[RegionName; 2]>),

    /// 'a = 'b
    Equal(RegionName, RegionName),

    /// 'a: 'static i.e. 'a = 'static and the user should just use 'static
    Static(RegionName),
}

/// Collects information about outlives constraints that needed to be added for a given MIR node
/// corresponding to a function definition.
///
/// Adds a help note suggesting adding a where clause with the needed constraints.
#[derive(Default)]
pub(crate) struct OutlivesSuggestionBuilder {
    /// The list of outlives constraints that need to be added. Specifically, we map each free
    /// region to all other regions that it must outlive. I will use the shorthand `fr:
    /// outlived_frs`. Not all of these regions will already have names necessarily. Some could be
    /// implicit free regions that we inferred. These will need to be given names in the final
    /// suggestion message.
    constraints_to_add: BTreeMap<RegionVid, Vec<RegionVid>>,
}

impl OutlivesSuggestionBuilder {
    /// Returns `true` iff the `RegionNameSource` is a valid source for an outlives
    /// suggestion.
    //
    // FIXME: Currently, we only report suggestions if the `RegionNameSource` is an early-bound
    // region or a named region, avoiding using regions with synthetic names altogether. This
    // allows us to avoid giving impossible suggestions (e.g. adding bounds to closure args).
    // We can probably be less conservative, since some inferred free regions are namable (e.g.
    // the user can explicitly name them. To do this, we would allow some regions whose names
    // come from `MatchedAdtAndSegment`, being careful to filter out bad suggestions, such as
    // naming the `'self` lifetime in methods, etc.
    fn region_name_is_suggestable(name: &RegionName) -> bool {
        match name.source {
            RegionNameSource::NamedEarlyParamRegion(..)
            | RegionNameSource::NamedLateParamRegion(..)
            | RegionNameSource::Static => true,

            // Don't give suggestions for upvars, closure return types, or other unnameable
            // regions.
            RegionNameSource::SynthesizedFreeEnvRegion(..)
            | RegionNameSource::AnonRegionFromArgument(..)
            | RegionNameSource::AnonRegionFromUpvar(..)
            | RegionNameSource::AnonRegionFromOutput(..)
            | RegionNameSource::AnonRegionFromYieldTy(..)
            | RegionNameSource::AnonRegionFromAsyncFn(..)
            | RegionNameSource::AnonRegionFromImplSignature(..) => {
                debug!("Region {:?} is NOT suggestable", name);
                false
            }
        }
    }

    /// Returns a name for the region if it is suggestable. See `region_name_is_suggestable`.
    fn region_vid_to_name(
        &self,
        mbcx: &MirBorrowckCtxt<'_, '_, '_>,
        region: RegionVid,
    ) -> Option<RegionName> {
        mbcx.give_region_a_name(region).filter(Self::region_name_is_suggestable)
    }

    /// Compiles a list of all suggestions to be printed in the final big suggestion.
    fn compile_all_suggestions(
        &self,
        mbcx: &MirBorrowckCtxt<'_, '_, '_>,
    ) -> SmallVec<[SuggestedConstraint; 2]> {
        let mut suggested = SmallVec::new();

        // Keep track of variables that we have already suggested unifying so that we don't print
        // out silly duplicate messages.
        let mut unified_already = FxIndexSet::default();

        for (fr, outlived) in &self.constraints_to_add {
            let Some(fr_name) = self.region_vid_to_name(mbcx, *fr) else {
                continue;
            };

            let outlived = outlived
                .iter()
                // if there is a `None`, we will just omit that constraint
                .filter_map(|fr| self.region_vid_to_name(mbcx, *fr).map(|rname| (fr, rname)))
                .collect::<Vec<_>>();

            // No suggestable outlived lifetimes.
            if outlived.is_empty() {
                continue;
            }

            // There are three types of suggestions we can make:
            // 1) Suggest a bound: 'a: 'b
            // 2) Suggest replacing 'a with 'static. If any of `outlived` is `'static`, then we
            //    should just replace 'a with 'static.
            // 3) Suggest unifying 'a with 'b if we have both 'a: 'b and 'b: 'a

            if outlived
                .iter()
                .any(|(_, outlived_name)| matches!(outlived_name.source, RegionNameSource::Static))
            {
                suggested.push(SuggestedConstraint::Static(fr_name));
            } else {
                // We want to isolate out all lifetimes that should be unified and print out
                // separate messages for them.

                let (unified, other): (Vec<_>, Vec<_>) = outlived.into_iter().partition(
                    // Do we have both 'fr: 'r and 'r: 'fr?
                    |(r, _)| {
                        self.constraints_to_add
                            .get(r)
                            .is_some_and(|r_outlived| r_outlived.as_slice().contains(fr))
                    },
                );

                for (r, bound) in unified.into_iter() {
                    if !unified_already.contains(fr) {
                        suggested.push(SuggestedConstraint::Equal(fr_name, bound));
                        unified_already.insert(r);
                    }
                }

                if !other.is_empty() {
                    let other = other.iter().map(|(_, rname)| *rname).collect::<SmallVec<_>>();
                    suggested.push(SuggestedConstraint::Outlives(fr_name, other))
                }
            }
        }

        suggested
    }

    /// Add the outlives constraint `fr: outlived_fr` to the set of constraints we need to suggest.
    pub(crate) fn collect_constraint(&mut self, fr: RegionVid, outlived_fr: RegionVid) {
        debug!("Collected {:?}: {:?}", fr, outlived_fr);

        // Add to set of constraints for final help note.
        self.constraints_to_add.entry(fr).or_default().push(outlived_fr);
    }

    /// Emit an intermediate note on the given `Diag` if the involved regions are suggestable.
    pub(crate) fn intermediate_suggestion(
        &mut self,
        mbcx: &MirBorrowckCtxt<'_, '_, '_>,
        errci: &ErrorConstraintInfo<'_>,
        diag: &mut Diag<'_>,
    ) {
        // Emit an intermediate note.
        let fr_name = self.region_vid_to_name(mbcx, errci.fr);
        let outlived_fr_name = self.region_vid_to_name(mbcx, errci.outlived_fr);

        if let (Some(fr_name), Some(outlived_fr_name)) = (fr_name, outlived_fr_name)
            && !matches!(outlived_fr_name.source, RegionNameSource::Static)
        {
            diag.help(format!(
                "consider adding the following bound: `{fr_name}: {outlived_fr_name}`",
            ));
        }
    }

    /// If there is a suggestion to emit, add a diagnostic to the buffer. This is the final
    /// suggestion including all collected constraints.
    pub(crate) fn add_suggestion(&self, mbcx: &mut MirBorrowckCtxt<'_, '_, '_>) {
        // No constraints to add? Done.
        if self.constraints_to_add.is_empty() {
            debug!("No constraints to suggest.");
            return;
        }

        // If there is only one constraint to suggest, then we already suggested it in the
        // intermediate suggestion above.
        if self.constraints_to_add.len() == 1
            && self.constraints_to_add.values().next().unwrap().len() == 1
        {
            debug!("Only 1 suggestion. Skipping.");
            return;
        }

        // Get all suggestable constraints.
        let suggested = self.compile_all_suggestions(mbcx);

        // If there are no suggestable constraints...
        if suggested.is_empty() {
            debug!("Only 1 suggestable constraint. Skipping.");
            return;
        }

        // If there is exactly one suggestable constraints, then just suggest it. Otherwise, emit a
        // list of diagnostics.
        let mut diag = if let [constraint] = suggested.as_slice() {
            mbcx.dcx().struct_help(match constraint {
                SuggestedConstraint::Outlives(a, bs) => {
                    let bs: SmallVec<[String; 2]> = bs.iter().map(|r| r.to_string()).collect();
                    format!("add bound `{a}: {}`", bs.join(" + "))
                }

                SuggestedConstraint::Equal(a, b) => {
                    format!("`{a}` and `{b}` must be the same: replace one with the other")
                }
                SuggestedConstraint::Static(a) => format!("replace `{a}` with `'static`"),
            })
        } else {
            // Create a new diagnostic.
            let mut diag = mbcx
                .infcx
                .tcx
                .dcx()
                .struct_help("the following changes may resolve your lifetime errors");

            // Add suggestions.
            for constraint in suggested {
                match constraint {
                    SuggestedConstraint::Outlives(a, bs) => {
                        let bs: SmallVec<[String; 2]> = bs.iter().map(|r| r.to_string()).collect();
                        diag.help(format!("add bound `{a}: {}`", bs.join(" + ")));
                    }
                    SuggestedConstraint::Equal(a, b) => {
                        diag.help(format!(
                            "`{a}` and `{b}` must be the same: replace one with the other",
                        ));
                    }
                    SuggestedConstraint::Static(a) => {
                        diag.help(format!("replace `{a}` with `'static`"));
                    }
                }
            }

            diag
        };

        // We want this message to appear after other messages on the mir def.
        let mir_span = mbcx.body.span;
        diag.sort_span = mir_span.shrink_to_hi();

        // Buffer the diagnostic
        mbcx.buffer_non_error(diag);
    }
}
