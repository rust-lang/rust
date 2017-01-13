// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty;

use syntax_pos::Span;

/// Defines strategies for handling regions that are omitted.  For
/// example, if one writes the type `&Foo`, then the lifetime of
/// this reference has been omitted. When converting this
/// type, the generic functions in astconv will invoke `anon_region`
/// on the provided region-scope to decide how to translate this
/// omitted region.
///
/// It is not always legal to omit regions, therefore `anon_region`
/// can return `Err(())` to indicate that this is not a scope in which
/// regions can legally be omitted.
pub trait RegionScope {
    /// If an object omits any explicit lifetime bound, and none can
    /// be derived from the object traits, what should we use? If
    /// `None` is returned, an explicit annotation is required.
    fn object_lifetime_default(&self, span: Span) -> Option<ty::Region>;

    /// The "base" default is the initial default for a scope. This is
    /// 'static except for in fn bodies, where it is a fresh inference
    /// variable. You shouldn't call this except for as part of
    /// computing `object_lifetime_default` (in particular, in legacy
    /// modes, it may not be relevant).
    fn base_object_lifetime_default(&self, span: Span) -> ty::Region;
}

// A scope in which all regions must be explicitly named. This is used
// for types that appear in structs and so on.
#[derive(Copy, Clone)]
pub struct ExplicitRscope;

impl RegionScope for ExplicitRscope {
    fn object_lifetime_default(&self, span: Span) -> Option<ty::Region> {
        Some(self.base_object_lifetime_default(span))
    }

    fn base_object_lifetime_default(&self, _span: Span) -> ty::Region {
        ty::ReStatic
    }
}

/// A scope which overrides the default object lifetime but has no other effect.
pub struct ObjectLifetimeDefaultRscope<'r> {
    base_scope: &'r (RegionScope+'r),
    default: ty::ObjectLifetimeDefault<'r>,
}

impl<'r> ObjectLifetimeDefaultRscope<'r> {
    pub fn new(base_scope: &'r (RegionScope+'r),
               default: ty::ObjectLifetimeDefault<'r>)
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
            ty::ObjectLifetimeDefault::Ambiguous =>
                None,

            ty::ObjectLifetimeDefault::BaseDefault =>
                // NB: This behavior changed in Rust 1.3.
                Some(self.base_object_lifetime_default(span)),

            ty::ObjectLifetimeDefault::Specific(r) =>
                Some(*r),
        }
    }

    fn base_object_lifetime_default(&self, span: Span) -> ty::Region {
        self.base_scope.base_object_lifetime_default(span)
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
            .map(|r| ty::fold::shift_region(r, 1))
    }

    fn base_object_lifetime_default(&self, span: Span) -> ty::Region {
        ty::fold::shift_region(self.base_scope.base_object_lifetime_default(span), 1)
    }
}
