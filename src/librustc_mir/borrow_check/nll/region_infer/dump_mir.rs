// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! As part of generating the regions, if you enable `-Zdump-mir=nll`,
//! we will generate an annotated copy of the MIR that includes the
//! state of region inference. This code handles emitting the region
//! context internal state.

use std::io::{self, Write};
use super::{OutlivesConstraint, RegionInferenceContext};

// Room for "'_#NNNNr" before things get misaligned.
// Easy enough to fix if this ever doesn't seem like
// enough.
const REGION_WIDTH: usize = 8;

impl<'tcx> RegionInferenceContext<'tcx> {
    /// Write out our state into the `.mir` files.
    pub(crate) fn dump_mir(&self, out: &mut dyn Write) -> io::Result<()> {
        writeln!(out, "| Free Region Mapping")?;

        for region in self.regions() {
            if self.definitions[region].is_universal {
                let classification = self.universal_regions.region_classification(region).unwrap();
                let outlived_by = self.universal_regions.regions_outlived_by(region);
                writeln!(
                    out,
                    "| {r:rw$} | {c:cw$} | {ob}",
                    r = format!("{:?}", region),
                    rw = REGION_WIDTH,
                    c = format!("{:?}", classification),
                    cw = 8, // "External" at most
                    ob = format!("{:?}", outlived_by)
                )?;
            }
        }

        writeln!(out, "|")?;
        writeln!(out, "| Inferred Region Values")?;
        for region in self.regions() {
            writeln!(
                out,
                "| {r:rw$} | {v}",
                r = format!("{:?}", region),
                rw = REGION_WIDTH,
                v = self.region_value_str(region),
            )?;
        }

        writeln!(out, "|")?;
        writeln!(out, "| Inference Constraints")?;
        self.for_each_constraint(&mut |msg| writeln!(out, "| {}", msg))?;

        Ok(())
    }

    /// Debugging aid: Invokes the `with_msg` callback repeatedly with
    /// our internal region constraints.  These are dumped into the
    /// -Zdump-mir file so that we can figure out why the region
    /// inference resulted in the values that it did when debugging.
    fn for_each_constraint(
        &self,
        with_msg: &mut dyn FnMut(&str) -> io::Result<()>,
    ) -> io::Result<()> {
        for region in self.definitions.indices() {
            let value = self.liveness_constraints.region_value_str(region);
            if value != "{}" {
                with_msg(&format!("{:?} live at {}", region, value))?;
            }
        }

        let mut constraints: Vec<_> = self.constraints.iter().collect();
        constraints.sort_unstable();
        for constraint in &constraints {
            let OutlivesConstraint {
                sup,
                sub,
                locations,
            } = constraint;
            with_msg(&format!(
                "{:?}: {:?} due to {:?}",
                sup,
                sub,
                locations,
            ))?;
        }

        Ok(())
    }
}

