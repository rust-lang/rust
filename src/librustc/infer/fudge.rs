// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ty::{self, TyCtxt};
use ty::fold::{TypeFoldable, TypeFolder};

use super::InferCtxt;
use super::RegionVariableOrigin;

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    /// This rather funky routine is used while processing expected
    /// types. What happens here is that we want to propagate a
    /// coercion through the return type of a fn to its
    /// argument. Consider the type of `Option::Some`, which is
    /// basically `for<T> fn(T) -> Option<T>`. So if we have an
    /// expression `Some(&[1, 2, 3])`, and that has the expected type
    /// `Option<&[u32]>`, we would like to type check `&[1, 2, 3]`
    /// with the expectation of `&[u32]`. This will cause us to coerce
    /// from `&[u32; 3]` to `&[u32]` and make the users life more
    /// pleasant.
    ///
    /// The way we do this is using `fudge_regions_if_ok`. What the
    /// routine actually does is to start a snapshot and execute the
    /// closure `f`. In our example above, what this closure will do
    /// is to unify the expectation (`Option<&[u32]>`) with the actual
    /// return type (`Option<?T>`, where `?T` represents the variable
    /// instantiated for `T`).  This will cause `?T` to be unified
    /// with `&?a [u32]`, where `?a` is a fresh lifetime variable. The
    /// input type (`?T`) is then returned by `f()`.
    ///
    /// At this point, `fudge_regions_if_ok` will normalize all type
    /// variables, converting `?T` to `&?a [u32]` and end the
    /// snapshot.  The problem is that we can't just return this type
    /// out, because it references the region variable `?a`, and that
    /// region variable was popped when we popped the snapshot.
    ///
    /// So what we do is to keep a list (`region_vars`, in the code below)
    /// of region variables created during the snapshot (here, `?a`). We
    /// fold the return value and replace any such regions with a *new*
    /// region variable (e.g., `?b`) and return the result (`&?b [u32]`).
    /// This can then be used as the expectation for the fn argument.
    ///
    /// The important point here is that, for soundness purposes, the
    /// regions in question are not particularly important. We will
    /// use the expected types to guide coercions, but we will still
    /// type-check the resulting types from those coercions against
    /// the actual types (`?T`, `Option<?T`) -- and remember that
    /// after the snapshot is popped, the variable `?T` is no longer
    /// unified.
    ///
    /// Assumptions:
    /// - no new type variables are created during `f()` (asserted
    ///   below); this simplifies our logic since we don't have to
    ///   check for escaping type variables
    pub fn fudge_regions_if_ok<T, E, F>(&self,
                                        origin: &RegionVariableOrigin,
                                        f: F) -> Result<T, E> where
        F: FnOnce() -> Result<T, E>,
        T: TypeFoldable<'tcx>,
    {
        let (region_vars, value) = self.probe(|snapshot| {
            let vars_at_start = self.type_variables.borrow().num_vars();

            match f() {
                Ok(value) => {
                    let value = self.resolve_type_vars_if_possible(&value);

                    // At this point, `value` could in principle refer
                    // to regions that have been created during the
                    // snapshot (we assert below that `f()` does not
                    // create any new type variables, so there
                    // shouldn't be any of those). Once we exit
                    // `probe()`, those are going to be popped, so we
                    // will have to eliminate any references to them.

                    assert_eq!(self.type_variables.borrow().num_vars(), vars_at_start,
                               "type variables were created during fudge_regions_if_ok");
                    let region_vars =
                        self.region_vars.vars_created_since_snapshot(
                            &snapshot.region_vars_snapshot);

                    Ok((region_vars, value))
                }
                Err(e) => Err(e),
            }
        })?;

        // At this point, we need to replace any of the now-popped
        // region variables that appear in `value` with a fresh region
        // variable. We can't do this during the probe because they
        // would just get popped then too. =)

        // Micro-optimization: if no variables have been created, then
        // `value` can't refer to any of them. =) So we can just return it.
        if region_vars.is_empty() {
            return Ok(value);
        }

        let mut fudger = RegionFudger {
            infcx: self,
            region_vars: &region_vars,
            origin: origin
        };

        Ok(value.fold_with(&mut fudger))
    }
}

pub struct RegionFudger<'a, 'gcx: 'a+'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    region_vars: &'a Vec<ty::RegionVid>,
    origin: &'a RegionVariableOrigin,
}

impl<'a, 'gcx, 'tcx> TypeFolder<'gcx, 'tcx> for RegionFudger<'a, 'gcx, 'tcx> {
    fn tcx<'b>(&'b self) -> TyCtxt<'b, 'gcx, 'tcx> {
        self.infcx.tcx
    }

    fn fold_region(&mut self, r: &'tcx ty::Region) -> &'tcx ty::Region {
        match *r {
            ty::ReVar(v) if self.region_vars.contains(&v) => {
                self.infcx.next_region_var(self.origin.clone())
            }
            _ => {
                r
            }
        }
    }
}
