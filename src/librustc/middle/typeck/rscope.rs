// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


use middle::ty;

use std::result;
use syntax::ast;
use syntax::codemap::Span;
use syntax::opt_vec::OptVec;
use syntax::opt_vec;
use syntax::parse::token::special_idents;

#[deriving(ToStr)]
pub struct RegionError {
    msg: ~str,
    replacement: ty::Region
}

pub trait RegionScope {
    fn anon_region(&self, span: Span) -> Result<ty::Region, RegionError>;
    fn self_region(&self, span: Span) -> Result<ty::Region, RegionError>;
    fn named_region(&self, span: Span, id: ast::Ident)
                      -> Result<ty::Region, RegionError>;
}

#[deriving(Clone)]
pub struct EmptyRscope;
impl RegionScope for EmptyRscope {
    fn anon_region(&self, _span: Span) -> Result<ty::Region, RegionError> {
        result::Err(RegionError {
            msg: ~"only 'static is allowed here",
            replacement: ty::re_static
        })
    }
    fn self_region(&self, _span: Span) -> Result<ty::Region, RegionError> {
        self.anon_region(_span)
    }
    fn named_region(&self, _span: Span, _id: ast::Ident)
        -> Result<ty::Region, RegionError>
    {
        self.anon_region(_span)
    }
}

#[deriving(Clone)]
pub struct RegionParamNames(OptVec<ast::Ident>);

impl RegionParamNames {
    fn has_self(&self) -> bool {
        self.has_ident(special_idents::self_)
    }

    fn has_ident(&self, ident: ast::Ident) -> bool {
        for region_param_name in self.iter() {
            if *region_param_name == ident {
                return true;
            }
        }
        false
    }

    pub fn add_generics(&mut self, generics: &ast::Generics) {
        match generics.lifetimes {
            opt_vec::Empty => {}
            opt_vec::Vec(ref new_lifetimes) => {
                match **self {
                    opt_vec::Empty => {
                        *self = RegionParamNames(
                            opt_vec::Vec(new_lifetimes.map(|lt| lt.ident)));
                    }
                    opt_vec::Vec(ref mut existing_lifetimes) => {
                        for new_lifetime in new_lifetimes.iter() {
                            existing_lifetimes.push(new_lifetime.ident);
                        }
                    }
                }
            }
        }
    }

    // Convenience function to produce the error for an unresolved name. The
    // optional argument specifies a custom replacement.
    pub fn undeclared_name(custom_replacement: Option<ty::Region>)
                        -> Result<ty::Region, RegionError> {
        let replacement = match custom_replacement {
            None => ty::re_bound(ty::br_self),
            Some(custom_replacement) => custom_replacement
        };
        Err(RegionError {
            msg: ~"this lifetime must be declared",
            replacement: replacement
        })
    }

    pub fn from_generics(generics: &ast::Generics) -> RegionParamNames {
        match generics.lifetimes {
            opt_vec::Empty => RegionParamNames(opt_vec::Empty),
            opt_vec::Vec(ref lifetimes) => {
                RegionParamNames(opt_vec::Vec(lifetimes.map(|lt| lt.ident)))
            }
        }
    }

    pub fn from_lifetimes(lifetimes: &opt_vec::OptVec<ast::Lifetime>)
                       -> RegionParamNames {
        match *lifetimes {
            opt_vec::Empty => RegionParamNames::new(),
            opt_vec::Vec(ref v) => {
                RegionParamNames(opt_vec::Vec(v.map(|lt| lt.ident)))
            }
        }
    }

    fn new() -> RegionParamNames {
        RegionParamNames(opt_vec::Empty)
    }
}

#[deriving(Clone)]
struct RegionParameterization {
    variance: ty::region_variance,
    region_param_names: RegionParamNames,
}

impl RegionParameterization {
    pub fn from_variance_and_generics(variance: Option<ty::region_variance>,
                                      generics: &ast::Generics)
                                   -> Option<RegionParameterization> {
        match variance {
            None => None,
            Some(variance) => {
                Some(RegionParameterization {
                    variance: variance,
                    region_param_names:
                        RegionParamNames::from_generics(generics),
                })
            }
        }
    }
}

#[deriving(Clone)]
pub struct MethodRscope {
    explicit_self: ast::explicit_self_,
    variance: Option<ty::region_variance>,
    region_param_names: RegionParamNames,
}

impl MethodRscope {
    // `generics` here refers to the generics of the outer item (impl or
    // trait).
    pub fn new(explicit_self: ast::explicit_self_,
               variance: Option<ty::region_variance>,
               rcvr_generics: &ast::Generics)
            -> MethodRscope {
        let region_param_names =
            RegionParamNames::from_generics(rcvr_generics);
        MethodRscope {
            explicit_self: explicit_self,
            variance: variance,
            region_param_names: region_param_names
        }
    }

    pub fn region_param_names(&self) -> RegionParamNames {
        self.region_param_names.clone()
    }
}

impl RegionScope for MethodRscope {
    fn anon_region(&self, _span: Span) -> Result<ty::Region, RegionError> {
        result::Err(RegionError {
            msg: ~"anonymous lifetimes are not permitted here",
            replacement: ty::re_bound(ty::br_self)
        })
    }
    fn self_region(&self, _span: Span) -> Result<ty::Region, RegionError> {
        assert!(self.variance.is_some());
        match self.variance {
            None => {}  // must be borrowed self, so this is OK
            Some(_) => {
                if !self.region_param_names.has_self() {
                    return Err(RegionError {
                        msg: ~"the `self` lifetime must be declared",
                        replacement: ty::re_bound(ty::br_self)
                    })
                }
            }
        }
        result::Ok(ty::re_bound(ty::br_self))
    }
    fn named_region(&self, span: Span, id: ast::Ident)
                      -> Result<ty::Region, RegionError> {
        if !self.region_param_names.has_ident(id) {
            return RegionParamNames::undeclared_name(None);
        }
        do EmptyRscope.named_region(span, id).or_else |_e| {
            result::Err(RegionError {
                msg: ~"lifetime is not in scope",
                replacement: ty::re_bound(ty::br_self)
            })
        }
    }
}

#[deriving(Clone)]
pub struct TypeRscope(Option<RegionParameterization>);

impl TypeRscope {
    fn replacement(&self) -> ty::Region {
        if self.is_some() {
            ty::re_bound(ty::br_self)
        } else {
            ty::re_static
        }
    }
}
impl RegionScope for TypeRscope {
    fn anon_region(&self, _span: Span) -> Result<ty::Region, RegionError> {
        result::Err(RegionError {
            msg: ~"anonymous lifetimes are not permitted here",
            replacement: self.replacement()
        })
    }
    fn self_region(&self, _span: Span) -> Result<ty::Region, RegionError> {
        match **self {
            None => {
                // if the self region is used, region parameterization should
                // have inferred that this type is RP
                fail!("region parameterization should have inferred that \
                        this type is RP");
            }
            Some(ref region_parameterization) => {
                if !region_parameterization.region_param_names.has_self() {
                    return Err(RegionError {
                        msg: ~"the `self` lifetime must be declared",
                        replacement: ty::re_bound(ty::br_self)
                    })
                }
            }
        }
        result::Ok(ty::re_bound(ty::br_self))
    }
    fn named_region(&self, span: Span, id: ast::Ident)
                      -> Result<ty::Region, RegionError> {
        do EmptyRscope.named_region(span, id).or_else |_e| {
            result::Err(RegionError {
                msg: ~"only 'self is allowed as part of a type declaration",
                replacement: self.replacement()
            })
        }
    }
}

pub fn bound_self_region(rp: Option<ty::region_variance>)
                      -> OptVec<ty::Region> {
    match rp {
      Some(_) => opt_vec::with(ty::re_bound(ty::br_self)),
      None => opt_vec::Empty
    }
}

pub struct BindingRscope {
    base: @RegionScope,
    anon_bindings: @mut uint,
    region_param_names: RegionParamNames,
}

impl Clone for BindingRscope {
    fn clone(&self) -> BindingRscope {
        BindingRscope {
            base: self.base,
            anon_bindings: self.anon_bindings,
            region_param_names: self.region_param_names.clone(),
        }
    }
}

pub fn in_binding_rscope<RS:RegionScope + Clone + 'static>(
        this: &RS,
        region_param_names: RegionParamNames)
     -> BindingRscope {
    let base = @(*this).clone();
    let base = base as @RegionScope;
    BindingRscope {
        base: base,
        anon_bindings: @mut 0,
        region_param_names: region_param_names,
    }
}

impl RegionScope for BindingRscope {
    fn anon_region(&self, _span: Span) -> Result<ty::Region, RegionError> {
        let idx = *self.anon_bindings;
        *self.anon_bindings += 1;
        result::Ok(ty::re_bound(ty::br_anon(idx)))
    }
    fn self_region(&self, span: Span) -> Result<ty::Region, RegionError> {
        self.base.self_region(span)
    }
    fn named_region(&self,
                    span: Span,
                    id: ast::Ident) -> Result<ty::Region, RegionError>
    {
        do self.base.named_region(span, id).or_else |_e| {
            let result = ty::re_bound(ty::br_named(id));
            if self.region_param_names.has_ident(id) {
                result::Ok(result)
            } else {
                RegionParamNames::undeclared_name(Some(result))
            }
        }
    }
}
