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

use std::vec;
use syntax::ast;
use syntax::codemap::Span;
use syntax::opt_vec::OptVec;

pub trait RegionScope {
    fn anon_regions(&self,
                    span: Span,
                    count: uint)
                    -> Option<~[ty::Region]>;
}

// A scope in which all regions must be explicitly named
pub struct ExplicitRscope;

impl RegionScope for ExplicitRscope {
    fn anon_regions(&self,
                    _span: Span,
                    _count: uint)
                    -> Option<~[ty::Region]> {
        None
    }
}

pub struct BindingRscope {
    binder_id: ast::NodeId,
    anon_bindings: @mut uint
}

impl BindingRscope {
    pub fn new(binder_id: ast::NodeId) -> BindingRscope {
        BindingRscope {
            binder_id: binder_id,
            anon_bindings: @mut 0
        }
    }
}

impl RegionScope for BindingRscope {
    fn anon_regions(&self,
                    _: Span,
                    count: uint)
                    -> Option<~[ty::Region]> {
        let idx = *self.anon_bindings;
        *self.anon_bindings += count;
        Some(vec::from_fn(count, |i| ty::re_fn_bound(self.binder_id,
                                                     ty::br_anon(idx + i))))
    }
}

pub fn bound_type_regions(defs: &[ty::RegionParameterDef])
                          -> OptVec<ty::Region> {
    assert!(defs.iter().all(|def| def.def_id.crate == ast::LOCAL_CRATE));
    defs.iter().enumerate().map(
        |(i, def)| ty::re_type_bound(def.def_id.node, i, def.ident)).collect()
}
