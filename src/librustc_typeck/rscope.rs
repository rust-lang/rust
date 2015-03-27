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
use middle::ty_fold;

use std::cell::Cell;
use std::iter::repeat;
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
                    count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<(String, usize)>>>;

    /// If an object omits any explicit lifetime bound, and none can
    /// be derived from the object traits, what should we use? If
    /// `None` is returned, an explicit annotation is required.
    fn object_lifetime_default(&self, span: Span) -> Option<ty::Region>;
}

// A scope in which all regions must be explicitly named. This is used
// for types that appear in structs and so on.
#[derive(Copy)]
pub struct ExplicitRscope;

impl RegionScope for ExplicitRscope {
    fn object_lifetime_default(&self, _span: Span) -> Option<ty::Region> {
        Some(ty::ReStatic)
    }

    fn anon_regions(&self,
                    _span: Span,
                    _count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<(String, usize)>>> {
        Err(None)
    }
}

// Same as `ExplicitRscope`, but provides some extra information for diagnostics
pub struct UnelidableRscope(Vec<(String, usize)>);

impl UnelidableRscope {
    pub fn new(v: Vec<(String, usize)>) -> UnelidableRscope {
        UnelidableRscope(v)
    }
}

impl RegionScope for UnelidableRscope {
    fn object_lifetime_default(&self, _span: Span) -> Option<ty::Region> {
        Some(ty::ReStatic)
    }

    fn anon_regions(&self,
                    _span: Span,
                    _count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<(String, usize)>>> {
        let UnelidableRscope(ref v) = *self;
        Err(Some(v.clone()))
    }
}

// A scope in which omitted anonymous region defaults to
// `default`. This is used after the `->` in function signatures. The
// latter use may go away. Note that object-lifetime defaults work a
// bit differently, as specified in RFC #599.
pub struct ElidableRscope {
    default: ty::Region,
}

impl ElidableRscope {
    pub fn new(r: ty::Region) -> ElidableRscope {
        ElidableRscope { default: r }
    }
}

impl RegionScope for ElidableRscope {
    fn object_lifetime_default(&self, _span: Span) -> Option<ty::Region> {
        // Per RFC #599, object-lifetimes default to 'static unless
        // overridden by context, and this takes precedence over
        // lifetime elision.
        Some(ty::ReStatic)
    }

    fn anon_regions(&self,
                    _span: Span,
                    count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<(String, usize)>>>
    {
        Ok(repeat(self.default).take(count).collect())
    }
}

/// A scope in which we generate anonymous, late-bound regions for
/// omitted regions. This occurs in function signatures.
pub struct BindingRscope {
    anon_bindings: Cell<u32>,
}

impl BindingRscope {
    pub fn new() -> BindingRscope {
        BindingRscope {
            anon_bindings: Cell::new(0),
        }
    }

    fn next_region(&self) -> ty::Region {
        let idx = self.anon_bindings.get();
        self.anon_bindings.set(idx + 1);
        ty::ReLateBound(ty::DebruijnIndex::new(1), ty::BrAnon(idx))
    }
}

impl RegionScope for BindingRscope {
    fn object_lifetime_default(&self, _span: Span) -> Option<ty::Region> {
        // Per RFC #599, object-lifetimes default to 'static unless
        // overridden by context, and this takes precedence over the
        // binding defaults.
        Some(ty::ReStatic)
    }

    fn anon_regions(&self,
                    _: Span,
                    count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<(String, usize)>>>
    {
        Ok((0..count).map(|_| self.next_region()).collect())
    }
}

/// A scope which overrides the default object lifetime but has no other effect.
pub struct ObjectLifetimeDefaultRscope<'r> {
    base_scope: &'r (RegionScope+'r),
    default: Option<ty::ObjectLifetimeDefault>,
}

impl<'r> ObjectLifetimeDefaultRscope<'r> {
    pub fn new(base_scope: &'r (RegionScope+'r),
               default: Option<ty::ObjectLifetimeDefault>)
               -> ObjectLifetimeDefaultRscope<'r>
    {
        ObjectLifetimeDefaultRscope {
            base_scope: base_scope,
            default: default,
        }
    }
}

impl<'r> RegionScope for ObjectLifetimeDefaultRscope<'r> {
    fn object_lifetime_default(&self, span: Span) -> Option<ty::Region> {
        match self.default {
            None => self.base_scope.object_lifetime_default(span),
            Some(ty::ObjectLifetimeDefault::Ambiguous) => None,
            Some(ty::ObjectLifetimeDefault::Specific(r)) => Some(r),
        }
    }

    fn anon_regions(&self,
                    span: Span,
                    count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<(String, usize)>>>
    {
        self.base_scope.anon_regions(span, count)
    }
}

/// A scope which simply shifts the Debruijn index of other scopes
/// to account for binding levels.
pub struct ShiftedRscope<'r> {
    base_scope: &'r (RegionScope+'r)
}

impl<'r> ShiftedRscope<'r> {
    pub fn new(base_scope: &'r (RegionScope+'r)) -> ShiftedRscope<'r> {
        ShiftedRscope { base_scope: base_scope }
    }
}

impl<'r> RegionScope for ShiftedRscope<'r> {
    fn object_lifetime_default(&self, span: Span) -> Option<ty::Region> {
        self.base_scope.object_lifetime_default(span)
            .map(|r| ty_fold::shift_region(r, 1))
    }

    fn anon_regions(&self,
                    span: Span,
                    count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<(String, usize)>>>
    {
        match self.base_scope.anon_regions(span, count) {
            Ok(mut v) => {
                for r in &mut v {
                    *r = ty_fold::shift_region(*r, 1);
                }
                Ok(v)
            }
            Err(errs) => {
                Err(errs)
            }
        }
    }
}
