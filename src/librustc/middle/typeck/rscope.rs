// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::result::Result;
use syntax::parse::token::special_idents;

trait region_scope {
    fn anon_region(span: span) -> Result<ty::Region, ~str>;
    fn self_region(span: span) -> Result<ty::Region, ~str>;
    fn named_region(span: span, id: ast::ident) -> Result<ty::Region, ~str>;
}

enum empty_rscope { empty_rscope }
impl empty_rscope: region_scope {
    fn anon_region(_span: span) -> Result<ty::Region, ~str> {
        result::Ok(ty::re_static)
    }
    fn self_region(_span: span) -> Result<ty::Region, ~str> {
        result::Err(~"only the static region is allowed here")
    }
    fn named_region(_span: span, _id: ast::ident)
        -> Result<ty::Region, ~str>
    {
        result::Err(~"only the static region is allowed here")
    }
}

enum type_rscope = Option<ty::region_variance>;
impl type_rscope: region_scope {
    fn anon_region(_span: span) -> Result<ty::Region, ~str> {
        match *self {
          Some(_) => result::Ok(ty::re_bound(ty::br_self)),
          None => result::Err(~"to use region types here, the containing \
                                type must be declared with a region bound")
        }
    }
    fn self_region(span: span) -> Result<ty::Region, ~str> {
        self.anon_region(span)
    }
    fn named_region(span: span, id: ast::ident) -> Result<ty::Region, ~str> {
        do empty_rscope.named_region(span, id).chain_err |_e| {
            result::Err(~"named regions other than `self` are not \
                          allowed as part of a type declaration")
        }
    }
}

fn bound_self_region(rp: Option<ty::region_variance>) -> Option<ty::Region> {
    match rp {
      Some(_) => Some(ty::re_bound(ty::br_self)),
      None => None
    }
}

enum anon_rscope = {anon: ty::Region, base: region_scope};
fn in_anon_rscope<RS: region_scope Copy Owned>(self: RS, r: ty::Region)
    -> @anon_rscope {
    @anon_rscope({anon: r, base: self as region_scope})
}
impl @anon_rscope: region_scope {
    fn anon_region(_span: span) -> Result<ty::Region, ~str> {
        result::Ok(self.anon)
    }
    fn self_region(span: span) -> Result<ty::Region, ~str> {
        self.base.self_region(span)
    }
    fn named_region(span: span, id: ast::ident) -> Result<ty::Region, ~str> {
        self.base.named_region(span, id)
    }
}

struct binding_rscope {
    base: region_scope,
    mut anon_bindings: uint,
}
fn in_binding_rscope<RS: region_scope Copy Owned>(self: RS)
    -> @binding_rscope {
    let base = self as region_scope;
    @binding_rscope { base: base, anon_bindings: 0 }
}
impl @binding_rscope: region_scope {
    fn anon_region(_span: span) -> Result<ty::Region, ~str> {
        let idx = self.anon_bindings;
        self.anon_bindings += 1;
        result::Ok(ty::re_bound(ty::br_anon(idx)))
    }
    fn self_region(span: span) -> Result<ty::Region, ~str> {
        self.base.self_region(span)
    }
    fn named_region(span: span, id: ast::ident) -> Result<ty::Region, ~str> {
        do self.base.named_region(span, id).chain_err |_e| {
            result::Ok(ty::re_bound(ty::br_named(id)))
        }
    }
}
