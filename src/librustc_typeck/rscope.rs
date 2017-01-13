// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir::def_id::DefId;
use rustc::ty;
use rustc::ty::subst::Substs;

use astconv::AstConv;

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

    /// If this scope allows anonymized types, return the generics in
    /// scope, that anonymized types will close over. For example,
    /// if you have a function like:
    ///
    ///     fn foo<'a, T>() -> impl Trait { ... }
    ///
    /// then, for the rscope that is used when handling the return type,
    /// `anon_type_scope()` would return a `Some(AnonTypeScope {...})`,
    /// on which `.fresh_substs(...)` can be used to obtain identity
    /// Substs for `'a` and `T`, to track them in `TyAnon`. This property
    /// is controlled by the region scope because it's fine-grained enough
    /// to allow restriction of anonymized types to the syntactical extent
    /// of a function's return type.
    fn anon_type_scope(&self) -> Option<AnonTypeScope> {
        None
    }
}

#[derive(Copy, Clone)]
pub struct AnonTypeScope {
    enclosing_item: DefId
}

impl<'gcx: 'tcx, 'tcx> AnonTypeScope {
    pub fn new(enclosing_item: DefId) -> AnonTypeScope {
        AnonTypeScope {
            enclosing_item: enclosing_item
        }
    }

    pub fn fresh_substs(&self, astconv: &AstConv<'gcx, 'tcx>, span: Span)
                        -> &'tcx Substs<'tcx> {
        use collect::mk_item_substs;

        mk_item_substs(astconv, span, self.enclosing_item)
    }
}

/// A scope wrapper which optionally allows anonymized types.
#[derive(Copy, Clone)]
pub struct MaybeWithAnonTypes<R> {
    base_scope: R,
    anon_scope: Option<AnonTypeScope>
}

impl<R: RegionScope> MaybeWithAnonTypes<R>  {
    pub fn new(base_scope: R, anon_scope: Option<AnonTypeScope>) -> Self {
        MaybeWithAnonTypes {
            base_scope: base_scope,
            anon_scope: anon_scope
        }
    }
}

impl<R: RegionScope> RegionScope for MaybeWithAnonTypes<R> {
    fn object_lifetime_default(&self, span: Span) -> Option<ty::Region> {
        self.base_scope.object_lifetime_default(span)
    }

    fn base_object_lifetime_default(&self, span: Span) -> ty::Region {
        self.base_scope.base_object_lifetime_default(span)
    }

    fn anon_type_scope(&self) -> Option<AnonTypeScope> {
        self.anon_scope
    }
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

    fn anon_type_scope(&self) -> Option<AnonTypeScope> {
        self.base_scope.anon_type_scope()
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

    fn anon_type_scope(&self) -> Option<AnonTypeScope> {
        self.base_scope.anon_type_scope()
    }
}
