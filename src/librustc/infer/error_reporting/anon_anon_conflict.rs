// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Error Reporting for Anonymous Region Lifetime Errors
//! where both the regions are anonymous.
use hir;
use infer::InferCtxt;
use ty::{self, Region};
use infer::region_inference::RegionResolutionError::*;
use infer::region_inference::RegionResolutionError;
use hir::map as hir_map;
use middle::resolve_lifetime as rl;
use hir::intravisit::{self, Visitor, NestedVisitorMap};

impl<'a, 'gcx, 'tcx> InferCtxt<'a, 'gcx, 'tcx> {
    // This method prints the error message for lifetime errors when both the concerned regions
    // are anonymous.
    // Consider a case where we have
    // fn foo(x: &mut Vec<&u8>, y: &u8)
    //    { x.push(y); }.
    // The example gives
    // fn foo(x: &mut Vec<&u8>, y: &u8) {
    //                    ---      --- these references must have the same lifetime
    //            x.push(y);
    //            ^ data from `y` flows into `x` here
    // It will later be extended to trait objects and structs.
    pub fn try_report_anon_anon_conflict(&self, error: &RegionResolutionError<'tcx>) -> bool {

        let (span, sub, sup) = match *error {
            ConcreteFailure(ref origin, sub, sup) => (origin.span(), sub, sup),
            _ => return false, // inapplicable
        };

        // Determine whether the sub and sup consist of both anonymous (elided) regions.
        let (ty1, ty2) = if self.is_suitable_anonymous_region(sup).is_some() &&
                            self.is_suitable_anonymous_region(sub).is_some() {
            if let (Some(anon_reg1), Some(anon_reg2)) =
                (self.is_suitable_anonymous_region(sup), self.is_suitable_anonymous_region(sub)) {
                let ((_, br1), (_, br2)) = (anon_reg1, anon_reg2);
                if self.find_anon_type(sup, &br1).is_some() &&
                   self.find_anon_type(sub, &br2).is_some() {
                    (self.find_anon_type(sup, &br1).unwrap(),
                     self.find_anon_type(sub, &br2).unwrap())
                } else {
                    return false;
                }
            } else {
                return false;
            }
        } else {
            return false; // inapplicable
        };

        if let (Some(sup_arg), Some(sub_arg)) =
            (self.find_arg_with_anonymous_region(sup, sup),
             self.find_arg_with_anonymous_region(sub, sub)) {
            let ((anon_arg1, _, _, _), (anon_arg2, _, _, _)) = (sup_arg, sub_arg);

            let span_label_var1 = if let Some(simple_name) = anon_arg1.pat.simple_name() {
                format!(" from `{}` ", simple_name)
            } else {
                format!(" ")
            };

            let span_label_var2 = if let Some(simple_name) = anon_arg2.pat.simple_name() {
                format!(" into `{}` ", simple_name)
            } else {
                format!(" ")
            };

            struct_span_err!(self.tcx.sess, span, E0623, "lifetime mismatch")
                .span_label(ty1.span,
                            format!("these references are not declared with the same lifetime..."))
                .span_label(ty2.span, format!(""))
                .span_label(span,
                            format!("...but data{}flows{}here", span_label_var1, span_label_var2))
                .emit();
        } else {
            return false;
        }

        return true;
    }

    /// This function calls the `visit_ty` method for the parameters
    /// corresponding to the anonymous regions. The `nested_visitor.found_type`
    /// contains the anonymous type.
    ///
    /// # Arguments
    ///
    /// region - the anonymous region corresponding to the anon_anon conflict
    /// br - the bound region corresponding to the above region which is of type `BrAnon(_)`
    ///
    /// # Example
    /// ```
    /// fn foo(x: &mut Vec<&u8>, y: &u8)
    ///    { x.push(y); }
    /// ```
    /// The function returns the nested type corresponding to the anonymous region
    /// for e.g. `&u8` and Vec<`&u8`.
    fn find_anon_type(&self, region: Region<'tcx>, br: &ty::BoundRegion) -> Option<&hir::Ty> {
        if let Some(anon_reg) = self.is_suitable_anonymous_region(region) {
            let (def_id, _) = anon_reg;
            if let Some(node_id) = self.tcx.hir.as_local_node_id(def_id) {
                let ret_ty = self.tcx.type_of(def_id);
                if let ty::TyFnDef(_, _) = ret_ty.sty {
                    if let hir_map::NodeItem(it) = self.tcx.hir.get(node_id) {
                        if let hir::ItemFn(ref fndecl, _, _, _, _, _) = it.node {
                            return fndecl
                                       .inputs
                                       .iter()
                                       .filter_map(|arg| {
                                let mut nested_visitor = FindNestedTypeVisitor {
                                    infcx: &self,
                                    hir_map: &self.tcx.hir,
                                    bound_region: *br,
                                    found_type: None,
                                };
                                nested_visitor.visit_ty(&**arg);
                                if nested_visitor.found_type.is_some() {
                                    nested_visitor.found_type
                                } else {
                                    None
                                }
                            })
                                       .next();
                        }
                    }
                }
            }
        }
        None
    }
}

// The FindNestedTypeVisitor captures the corresponding `hir::Ty` of the
// anonymous region. The example above would lead to a conflict between
// the two anonymous lifetimes for &u8 in x and y respectively. This visitor
// would be invoked twice, once for each lifetime, and would
// walk the types like &mut Vec<&u8> and &u8 looking for the HIR
// where that lifetime appears. This allows us to highlight the
// specific part of the type in the error message.
struct FindNestedTypeVisitor<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    infcx: &'a InferCtxt<'a, 'gcx, 'tcx>,
    hir_map: &'a hir::map::Map<'gcx>,
    // The bound_region corresponding to the Refree(freeregion)
    // associated with the anonymous region we are looking for.
    bound_region: ty::BoundRegion,
    // The type where the anonymous lifetime appears
    // for e.g. Vec<`&u8`> and <`&u8`>
    found_type: Option<&'gcx hir::Ty>,
}

impl<'a, 'gcx, 'tcx> Visitor<'gcx> for FindNestedTypeVisitor<'a, 'gcx, 'tcx> {
    fn nested_visit_map<'this>(&'this mut self) -> NestedVisitorMap<'this, 'gcx> {
        NestedVisitorMap::OnlyBodies(&self.hir_map)
    }

    fn visit_ty(&mut self, arg: &'gcx hir::Ty) {
        // Find the index of the anonymous region that was part of the
        // error. We will then search the function parameters for a bound
        // region at the right depth with the same index.
        let br_index = match self.bound_region {
            ty::BrAnon(index) => index,
            _ => return,
        };

        match arg.node {
            hir::TyRptr(ref lifetime, _) => {
                match self.infcx.tcx.named_region_map.defs.get(&lifetime.id) {
                    // the lifetime of the TyRptr
                    Some(&rl::Region::LateBoundAnon(debuijn_index, anon_index)) => {
                        if debuijn_index.depth == 1 && anon_index == br_index {
                            self.found_type = Some(arg);
                            return; // we can stop visiting now
                        }
                    }
                    Some(&rl::Region::Static) |
                    Some(&rl::Region::EarlyBound(_, _)) |
                    Some(&rl::Region::LateBound(_, _)) |
                    Some(&rl::Region::Free(_, _)) |
                    None => {
                        debug!("no arg found");
                    }
                }
            }
            _ => {}
        }
        // walk the embedded contents: e.g., if we are visiting `Vec<&Foo>`,
        // go on to visit `&Foo`
        intravisit::walk_ty(self, arg);
    }
}
