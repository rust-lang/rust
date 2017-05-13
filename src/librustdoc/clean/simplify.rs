// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simplification of where clauses and parameter bounds into a prettier and
//! more canonical form.
//!
//! Currently all cross-crate-inlined function use `rustc::ty` to reconstruct
//! the AST (e.g. see all of `clean::inline`), but this is not always a
//! non-lossy transformation. The current format of storage for where clauses
//! for functions and such is simply a list of predicates. One example of this
//! is that the AST predicate of:
//!
//!     where T: Trait<Foo=Bar>
//!
//! is encoded as:
//!
//!     where T: Trait, <T as Trait>::Foo = Bar
//!
//! This module attempts to reconstruct the original where and/or parameter
//! bounds by special casing scenarios such as these. Fun!

use std::mem;
use std::collections::BTreeMap;

use rustc::hir::def_id::DefId;
use rustc::ty;

use clean::PathParameters as PP;
use clean::WherePredicate as WP;
use clean;
use core::DocContext;

pub fn where_clauses(cx: &DocContext, clauses: Vec<WP>) -> Vec<WP> {
    // First, partition the where clause into its separate components
    let mut params = BTreeMap::new();
    let mut lifetimes = Vec::new();
    let mut equalities = Vec::new();
    let mut tybounds = Vec::new();
    for clause in clauses {
        match clause {
            WP::BoundPredicate { ty, bounds } => {
                match ty {
                    clean::Generic(s) => params.entry(s).or_insert(Vec::new())
                                               .extend(bounds),
                    t => tybounds.push((t, ty_bounds(bounds))),
                }
            }
            WP::RegionPredicate { lifetime, bounds } => {
                lifetimes.push((lifetime, bounds));
            }
            WP::EqPredicate { lhs, rhs } => equalities.push((lhs, rhs)),
        }
    }

    // Simplify the type parameter bounds on all the generics
    let mut params = params.into_iter().map(|(k, v)| {
        (k, ty_bounds(v))
    }).collect::<BTreeMap<_, _>>();

    // Look for equality predicates on associated types that can be merged into
    // general bound predicates
    equalities.retain(|&(ref lhs, ref rhs)| {
        let (self_, trait_, name) = match *lhs {
            clean::QPath { ref self_type, ref trait_, ref name } => {
                (self_type, trait_, name)
            }
            _ => return true,
        };
        let generic = match **self_ {
            clean::Generic(ref s) => s,
            _ => return true,
        };
        let trait_did = match **trait_ {
            clean::ResolvedPath { did, .. } => did,
            _ => return true,
        };
        let bounds = match params.get_mut(generic) {
            Some(bound) => bound,
            None => return true,
        };
        !bounds.iter_mut().any(|b| {
            let trait_ref = match *b {
                clean::TraitBound(ref mut tr, _) => tr,
                clean::RegionBound(..) => return false,
            };
            let (did, path) = match trait_ref.trait_ {
                clean::ResolvedPath { did, ref mut path, ..} => (did, path),
                _ => return false,
            };
            // If this QPath's trait `trait_did` is the same as, or a supertrait
            // of, the bound's trait `did` then we can keep going, otherwise
            // this is just a plain old equality bound.
            if !trait_is_same_or_supertrait(cx, did, trait_did) {
                return false
            }
            let last = path.segments.last_mut().unwrap();
            match last.params {
                PP::AngleBracketed { ref mut bindings, .. } => {
                    bindings.push(clean::TypeBinding {
                        name: name.clone(),
                        ty: rhs.clone(),
                    });
                }
                PP::Parenthesized { ref mut output, .. } => {
                    assert!(output.is_none());
                    *output = Some(rhs.clone());
                }
            };
            true
        })
    });

    // And finally, let's reassemble everything
    let mut clauses = Vec::new();
    clauses.extend(lifetimes.into_iter().map(|(lt, bounds)| {
        WP::RegionPredicate { lifetime: lt, bounds: bounds }
    }));
    clauses.extend(params.into_iter().map(|(k, v)| {
        WP::BoundPredicate {
            ty: clean::Generic(k),
            bounds: v,
        }
    }));
    clauses.extend(tybounds.into_iter().map(|(ty, bounds)| {
        WP::BoundPredicate { ty: ty, bounds: bounds }
    }));
    clauses.extend(equalities.into_iter().map(|(lhs, rhs)| {
        WP::EqPredicate { lhs: lhs, rhs: rhs }
    }));
    clauses
}

pub fn ty_params(mut params: Vec<clean::TyParam>) -> Vec<clean::TyParam> {
    for param in &mut params {
        param.bounds = ty_bounds(mem::replace(&mut param.bounds, Vec::new()));
    }
    params
}

fn ty_bounds(bounds: Vec<clean::TyParamBound>) -> Vec<clean::TyParamBound> {
    bounds
}

fn trait_is_same_or_supertrait(cx: &DocContext, child: DefId,
                               trait_: DefId) -> bool {
    if child == trait_ {
        return true
    }
    let predicates = cx.tcx.item_super_predicates(child).predicates;
    predicates.iter().filter_map(|pred| {
        if let ty::Predicate::Trait(ref pred) = *pred {
            if pred.0.trait_ref.self_ty().is_self() {
                Some(pred.def_id())
            } else {
                None
            }
        } else {
            None
        }
    }).any(|did| trait_is_same_or_supertrait(cx, did, trait_))
}
