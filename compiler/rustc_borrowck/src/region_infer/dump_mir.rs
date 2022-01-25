//! As part of generating the regions, if you enable `-Zdump-mir=nll`,
//! we will generate an annotated copy of the MIR that includes the
//! state of region inference. This code handles emitting the region
//! context internal state.

use super::{OutlivesConstraint, RegionInferenceContext};
use crate::type_check::Locations;
use rustc_infer::infer::NllRegionVariableOrigin;
use rustc_middle::ty::TyCtxt;
use std::io::{self, Write};

// Room for "'_#NNNNr" before things get misaligned.
// Easy enough to fix if this ever doesn't seem like
// enough.
const REGION_WIDTH: usize = 8;

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Write out our state into the `.mir` files.
    pub(crate) fn dump_mir(&self, tcx: TyCtxt<'tcx>, out: &mut dyn Write) -> io::Result<()> {
        writeln!(out, "| Free Region Mapping")?;

        for region in self.regions() {
            if let NllRegionVariableOrigin::FreeRegion = self.definitions[region].origin {
                let classification = self.universal_regions.region_classification(region).unwrap();
                let outlived_by = self.universal_region_relations.regions_outlived_by(region);
                writeln!(
                    out,
                    "| {r:rw$?} | {c:cw$?} | {ob:?}",
                    r = region,
                    rw = REGION_WIDTH,
                    c = classification,
                    cw = 8, // "External" at most
                    ob = outlived_by
                )?;
            }
        }

        writeln!(out, "|")?;
        writeln!(out, "| Inferred Region Values")?;
        for region in self.regions() {
            writeln!(
                out,
                "| {r:rw$?} | {ui:4?} | {v}",
                r = region,
                rw = REGION_WIDTH,
                ui = self.region_universe(region),
                v = self.region_value_str(region),
            )?;
        }

        writeln!(out, "|")?;
        writeln!(out, "| Inference Constraints")?;
        self.for_each_constraint(tcx, &mut |msg| writeln!(out, "| {}", msg))?;

        Ok(())
    }

    /// Debugging aid: Invokes the `with_msg` callback repeatedly with
    /// our internal region constraints. These are dumped into the
    /// -Zdump-mir file so that we can figure out why the region
    /// inference resulted in the values that it did when debugging.
    fn for_each_constraint(
        &self,
        tcx: TyCtxt<'tcx>,
        with_msg: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<()> {
        for region in self.definitions.indices() {
            let value = self.liveness_constraints.region_value_str(region);
            if value != "{}" {
                with_msg(&format!("{:?} live at {}", region, value))?;
            }
        }

        let mut constraints: Vec<_> = self.constraints.outlives().iter().collect();
        constraints.sort_by_key(|c| (c.sup, c.sub));
        for constraint in &constraints {
            let OutlivesConstraint { sup, sub, locations, category, variance_info: _ } = constraint;
            let (name, arg) = match locations {
                Locations::All(span) => {
                    ("All", tcx.sess.source_map().span_to_embeddable_string(*span))
                }
                Locations::Single(loc) => ("Single", format!("{:?}", loc)),
            };
            with_msg(&format!("{:?}: {:?} due to {:?} at {}({})", sup, sub, category, name, arg))?;
        }

        Ok(())
    }
}
