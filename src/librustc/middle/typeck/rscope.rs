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

/// Defines strategies for handling regions that are omitted.  For
/// example, if one writes the type `&Foo`, then the lifetime of of
/// this borrowed pointer has been omitted. When converting this
/// type, the generic functions in astconv will invoke `anon_regions`
/// on the provided region-scope to decide how to translate this
/// omitted region.
///
/// It is not always legal to omit regions, therefore `anon_regions`
/// can return `Err(())` to indicate that this is not a scope in which
/// regions can legally be omitted.
pub trait RegionScope {
    fn anon_regions(&self,
                    span: Span,
                    count: uint)
                    -> Result<~[ty::Region], ()>;
}

// A scope in which all regions must be explicitly named
pub struct ExplicitRscope;

impl RegionScope for ExplicitRscope {
    fn anon_regions(&self,
                    _span: Span,
                    _count: uint)
                    -> Result<~[ty::Region], ()> {
        Err(())
    }
}

/// A scope in which we generate anonymous, late-bound regions for
/// omitted regions. This occurs in function signatures.
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
                    -> Result<~[ty::Region], ()> {
        let idx = *self.anon_bindings;
        *self.anon_bindings += count;
        Ok(vec::from_fn(count, |i| ty::ReLateBound(self.binder_id,
                                                   ty::BrAnon(idx + i))))
    }
}

pub fn bound_type_regions(defs: &[ty::RegionParameterDef])
                          -> OptVec<ty::Region> {
    assert!(defs.iter().all(|def| def.def_id.crate == ast::LOCAL_CRATE));
    defs.iter().enumerate().map(
        |(i, def)| ty::ReEarlyBound(def.def_id.node, i, def.ident)).collect()
}
