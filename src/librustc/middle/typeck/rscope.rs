// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::prelude::*;

use middle::ty;

use core::result::Result;
use core::result;
use syntax::ast;
use syntax::codemap::span;

pub struct RegionError {
    msg: ~str,
    replacement: ty::Region
}

pub trait region_scope {
    fn anon_region(&self, span: span) -> Result<ty::Region, RegionError>;
    fn self_region(&self, span: span) -> Result<ty::Region, RegionError>;
    fn named_region(&self, span: span, id: ast::ident)
                      -> Result<ty::Region, RegionError>;
}

pub enum empty_rscope { empty_rscope }
impl region_scope for empty_rscope {
    fn anon_region(&self, _span: span) -> Result<ty::Region, RegionError> {
        result::Err(RegionError {
            msg: ~"only 'static is allowed here",
            replacement: ty::re_static
        })
    }
    fn self_region(&self, _span: span) -> Result<ty::Region, RegionError> {
        self.anon_region(_span)
    }
    fn named_region(&self, _span: span, _id: ast::ident)
        -> Result<ty::Region, RegionError>
    {
        self.anon_region(_span)
    }
}

pub struct MethodRscope {
    self_ty: ast::self_ty_,
    region_parameterization: Option<ty::region_variance>
}
impl region_scope for MethodRscope {
    fn anon_region(&self, _span: span) -> Result<ty::Region, RegionError> {
        result::Err(RegionError {
            msg: ~"anonymous lifetimes are not permitted here",
            replacement: ty::re_bound(ty::br_self)
        })
    }
    fn self_region(&self, _span: span) -> Result<ty::Region, RegionError> {
        fail_unless!(self.region_parameterization.is_some() ||
            self.self_ty.is_borrowed());
        result::Ok(ty::re_bound(ty::br_self))
    }
    fn named_region(&self, span: span, id: ast::ident)
                      -> Result<ty::Region, RegionError> {
        do empty_rscope.named_region(span, id).chain_err |_e| {
            result::Err(RegionError {
                msg: ~"lifetime is not in scope",
                replacement: ty::re_bound(ty::br_self)
            })
        }
    }
}

pub enum type_rscope = Option<ty::region_variance>;
impl type_rscope {
    priv fn replacement(&self) -> ty::Region {
        if self.is_some() {
            ty::re_bound(ty::br_self)
        } else {
            ty::re_static
        }
    }
}
impl region_scope for type_rscope {
    fn anon_region(&self, _span: span) -> Result<ty::Region, RegionError> {
        result::Err(RegionError {
            msg: ~"anonymous lifetimes are not permitted here",
            replacement: self.replacement()
        })
    }
    fn self_region(&self, _span: span) -> Result<ty::Region, RegionError> {
        // if the self region is used, region parameterization should
        // have inferred that this type is RP
        fail_unless!(self.is_some());
        result::Ok(ty::re_bound(ty::br_self))
    }
    fn named_region(&self, span: span, id: ast::ident)
                      -> Result<ty::Region, RegionError> {
        do empty_rscope.named_region(span, id).chain_err |_e| {
            result::Err(RegionError {
                msg: ~"only 'self is allowed allowed as \
                       part of a type declaration",
                replacement: self.replacement()
            })
        }
    }
}

pub fn bound_self_region(rp: Option<ty::region_variance>)
                      -> Option<ty::Region> {
    match rp {
      Some(_) => Some(ty::re_bound(ty::br_self)),
      None => None
    }
}

pub struct binding_rscope {
    base: @region_scope,
    anon_bindings: @mut uint,
}

pub fn in_binding_rscope<RS:region_scope + Copy + Durable>(self: &RS)
    -> binding_rscope {
    let base = @(copy *self) as @region_scope;
    binding_rscope { base: base, anon_bindings: @mut 0 }
}
impl region_scope for binding_rscope {
    fn anon_region(&self, _span: span) -> Result<ty::Region, RegionError> {
        let idx = *self.anon_bindings;
        *self.anon_bindings += 1;
        result::Ok(ty::re_bound(ty::br_anon(idx)))
    }
    fn self_region(&self, span: span) -> Result<ty::Region, RegionError> {
        self.base.self_region(span)
    }
    fn named_region(&self,
                    span: span,
                    id: ast::ident) -> Result<ty::Region, RegionError>
    {
        do self.base.named_region(span, id).chain_err |_e| {
            result::Ok(ty::re_bound(ty::br_named(id)))
        }
    }
}
