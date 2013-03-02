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
use syntax::parse::token::special_idents;

pub trait region_scope {
    pure fn anon_region(&self, span: span) -> Result<ty::Region, ~str>;
    pure fn self_region(&self, span: span) -> Result<ty::Region, ~str>;
    pure fn named_region(&self, span: span, id: ast::ident)
                      -> Result<ty::Region, ~str>;
}

pub enum empty_rscope { empty_rscope }

impl region_scope for empty_rscope {
    pure fn anon_region(&self, _span: span) -> Result<ty::Region, ~str> {
        result::Ok(ty::re_static)
    }
    pure fn self_region(&self, _span: span) -> Result<ty::Region, ~str> {
        result::Err(~"only the static region is allowed here")
    }
    pure fn named_region(&self, _span: span, _id: ast::ident)
        -> Result<ty::Region, ~str> {
        result::Err(~"only the static region is allowed here")
    }
}

pub enum type_rscope = Option<ty::region_variance>;

impl region_scope for type_rscope {
    pure fn anon_region(&self, _span: span) -> Result<ty::Region, ~str> {
        match **self {
          Some(_) => result::Ok(ty::re_bound(ty::br_self)),
          None => result::Err(~"to use region types here, the containing \
                                type must be declared with a region bound")
        }
    }
    pure fn self_region(&self, span: span) -> Result<ty::Region, ~str> {
        self.anon_region(span)
    }
    pure fn named_region(&self, span: span, id: ast::ident)
                      -> Result<ty::Region, ~str> {
        do empty_rscope.named_region(span, id).chain_err |_e| {
            result::Err(~"named regions other than `self` are not \
                          allowed as part of a type declaration")
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

pub struct anon_rscope { anon: ty::Region, base: @region_scope }
pub fn in_anon_rscope<RS:region_scope + Copy + Durable>(self: RS,
                                                        r: ty::Region)
                                                     -> @anon_rscope {
    @anon_rscope {anon: r, base: @self as @region_scope}
}

impl region_scope for @anon_rscope {
    pure fn anon_region(&self, _span: span) -> Result<ty::Region, ~str> {
        result::Ok(self.anon)
    }
    pure fn self_region(&self, span: span) -> Result<ty::Region, ~str> {
        self.base.self_region(span)
    }
    pure fn named_region(&self, span: span, id: ast::ident)
                      -> Result<ty::Region, ~str> {
        self.base.named_region(span, id)
    }
}

pub struct binding_rscope {
    base: region_scope,
    anon_bindings: uint,
}

pub fn in_binding_rscope<RS:region_scope + Copy + Durable>(self: RS)
    -> @mut binding_rscope {
    let base = @self as @region_scope;
    @mut binding_rscope { base: base, anon_bindings: 0 }
}

impl region_scope for @mut binding_rscope {
    pure fn anon_region(&self, _span: span) -> Result<ty::Region, ~str> {
        // XXX: Unsafe to work around purity
        unsafe {
            let idx = self.anon_bindings;
            self.anon_bindings += 1;
            result::Ok(ty::re_bound(ty::br_anon(idx)))
        }
    }
    pure fn self_region(&self, span: span) -> Result<ty::Region, ~str> {
        self.base.self_region(span)
    }
    pure fn named_region(&self, span: span, id: ast::ident)
                      -> Result<ty::Region, ~str> {
        do self.base.named_region(span, id).chain_err |_e| {
            result::Ok(ty::re_bound(ty::br_named(id)))
        }
    }
}
