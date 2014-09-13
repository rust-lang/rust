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

use std::cell::Cell;
use syntax::ast;
use syntax::codemap::Span;

/// Defines strategies for handling regions that are omitted.  For
/// example, if one writes the type `&Foo`, then the lifetime of
/// this reference has been omitted. When converting this
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
                    -> Result<Vec<ty::Region>, Option<Vec<(String, uint)>>>;

    fn default_region_bound(&self, span: Span) -> Option<ty::Region>;
}

// A scope in which all regions must be explicitly named. This is used
// for types that appear in structs and so on.
pub struct ExplicitRscope;

impl RegionScope for ExplicitRscope {
    fn default_region_bound(&self, _span: Span) -> Option<ty::Region> {
        None
    }

    fn anon_regions(&self,
                    _span: Span,
                    _count: uint)
                    -> Result<Vec<ty::Region>, Option<Vec<(String, uint)>>> {
        Err(None)
    }
}

// Same as `ExplicitRscope`, but provides some extra information for diagnostics
pub struct UnelidableRscope(Vec<(String, uint)>);

impl UnelidableRscope {
    pub fn new(v: Vec<(String, uint)>) -> UnelidableRscope {
        UnelidableRscope(v)
    }
}

impl RegionScope for UnelidableRscope {
    fn default_region_bound(&self, _span: Span) -> Option<ty::Region> {
        None
    }

    fn anon_regions(&self,
                    _span: Span,
                    _count: uint)
                    -> Result<Vec<ty::Region>, Option<Vec<(String, uint)>>> {
        let UnelidableRscope(ref v) = *self;
        Err(Some(v.clone()))
    }
}

// A scope in which any omitted region defaults to `default`. This is
// used after the `->` in function signatures, but also for backwards
// compatibility with object types. The latter use may go away.
pub struct SpecificRscope {
    default: ty::Region
}

impl SpecificRscope {
    pub fn new(r: ty::Region) -> SpecificRscope {
        SpecificRscope { default: r }
    }
}

impl RegionScope for SpecificRscope {
    fn default_region_bound(&self, _span: Span) -> Option<ty::Region> {
        Some(self.default)
    }

    fn anon_regions(&self,
                    _span: Span,
                    count: uint)
                    -> Result<Vec<ty::Region>, Option<Vec<(String, uint)>>>
    {
        Ok(Vec::from_elem(count, self.default))
    }
}

/// A scope in which we generate anonymous, late-bound regions for
/// omitted regions. This occurs in function signatures.
pub struct BindingRscope {
    binder_id: ast::NodeId,
    anon_bindings: Cell<uint>,
}

impl BindingRscope {
    pub fn new(binder_id: ast::NodeId) -> BindingRscope {
        BindingRscope {
            binder_id: binder_id,
            anon_bindings: Cell::new(0),
        }
    }

    fn next_region(&self) -> ty::Region {
        let idx = self.anon_bindings.get();
        self.anon_bindings.set(idx + 1);
        ty::ReLateBound(self.binder_id, ty::BrAnon(idx))
    }
}

impl RegionScope for BindingRscope {
    fn default_region_bound(&self, _span: Span) -> Option<ty::Region>
    {
        Some(self.next_region())
    }

    fn anon_regions(&self,
                    _: Span,
                    count: uint)
                    -> Result<Vec<ty::Region>, Option<Vec<(String, uint)>>>
    {
        Ok(Vec::from_fn(count, |_| self.next_region()))
    }
}

