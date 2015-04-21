// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use visitor::FmtVisitor;

use syntax::ast;
use syntax::parse::token;
use syntax::print::pprust;

impl<'a> FmtVisitor<'a> {
    pub fn rewrite_pred(&self, predicate: &ast::WherePredicate) -> String
    {
        // TODO dead spans
        // TODO assumes we'll always fit on one line...
        match predicate {
            &ast::WherePredicate::BoundPredicate(ast::WhereBoundPredicate{ref bound_lifetimes,
                                                                          ref bounded_ty,
                                                                          ref bounds,
                                                                          ..}) => {
                if bound_lifetimes.len() > 0 {
                    format!("for<{}> {}: {}",
                            bound_lifetimes.iter().map(|l| self.rewrite_lifetime_def(l)).collect::<Vec<_>>().connect(", "),
                            pprust::ty_to_string(bounded_ty),
                            bounds.iter().map(|b| self.rewrite_ty_bound(b)).collect::<Vec<_>>().connect("+"))

                } else {
                    format!("{}: {}",
                            pprust::ty_to_string(bounded_ty),
                            bounds.iter().map(|b| self.rewrite_ty_bound(b)).collect::<Vec<_>>().connect("+"))
                }
            }
            &ast::WherePredicate::RegionPredicate(ast::WhereRegionPredicate{ref lifetime,
                                                                            ref bounds,
                                                                            ..}) => {
                format!("{}: {}",
                        pprust::lifetime_to_string(lifetime),
                        bounds.iter().map(|l| pprust::lifetime_to_string(l)).collect::<Vec<_>>().connect("+"))
            }
            &ast::WherePredicate::EqPredicate(ast::WhereEqPredicate{ref path, ref ty, ..}) => {
                format!("{} = {}", pprust::path_to_string(path), pprust::ty_to_string(ty))
            }
        }
    }

    pub fn rewrite_lifetime_def(&self, lifetime: &ast::LifetimeDef) -> String
    {
        if lifetime.bounds.len() == 0 {
            return pprust::lifetime_to_string(&lifetime.lifetime);
        }

        format!("{}: {}",
                pprust::lifetime_to_string(&lifetime.lifetime),
                lifetime.bounds.iter().map(|l| pprust::lifetime_to_string(l)).collect::<Vec<_>>().connect("+"))
    }

    pub fn rewrite_ty_bound(&self, bound: &ast::TyParamBound) -> String
    {
        match *bound {
            ast::TyParamBound::TraitTyParamBound(ref tref, ast::TraitBoundModifier::None) => {
                self.rewrite_poly_trait_ref(tref)
            }
            ast::TyParamBound::TraitTyParamBound(ref tref, ast::TraitBoundModifier::Maybe) => {
                format!("?{}", self.rewrite_poly_trait_ref(tref))
            }
            ast::TyParamBound::RegionTyParamBound(ref l) => {
                pprust::lifetime_to_string(l)
            }
        }
    }

    pub fn rewrite_ty_param(&self, ty_param: &ast::TyParam) -> String
    {
        let mut result = String::with_capacity(128);
        result.push_str(&token::get_ident(ty_param.ident));
        if ty_param.bounds.len() > 0 {
            result.push_str(": ");
            result.push_str(&ty_param.bounds.iter().map(|b| self.rewrite_ty_bound(b)).collect::<Vec<_>>().connect(", "));
        }
        if let Some(ref def) = ty_param.default {
            result.push_str(" = ");
            result.push_str(&pprust::ty_to_string(&def));
        }

        result
    }

    fn rewrite_poly_trait_ref(&self, t: &ast::PolyTraitRef) -> String
    {
        if t.bound_lifetimes.len() > 0 {
            format!("for<{}> {}",
                    t.bound_lifetimes.iter().map(|l| self.rewrite_lifetime_def(l)).collect::<Vec<_>>().connect(", "),
                    pprust::path_to_string(&t.trait_ref.path))

        } else {
            pprust::path_to_string(&t.trait_ref.path)
        }
    }
}
