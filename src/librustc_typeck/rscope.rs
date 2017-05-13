// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::hir;
use rustc::hir::def_id::DefId;
use rustc::ty;
use rustc::ty::subst::Substs;

use astconv::AstConv;

use std::cell::Cell;
use syntax_pos::Span;

#[derive(Clone)]
pub struct ElisionFailureInfo {
    /// Where we can find the argument pattern.
    pub parent: Option<hir::BodyId>,
    /// The index of the argument in the original definition.
    pub index: usize,
    pub lifetime_count: usize,
    pub have_bound_regions: bool
}

pub type ElidedLifetime = Result<ty::Region, Option<Vec<ElisionFailureInfo>>>;

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
                    -> Result<Vec<ty::Region>, Option<Vec<ElisionFailureInfo>>>;

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

    fn anon_regions(&self,
                    span: Span,
                    count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<ElisionFailureInfo>>> {
        self.base_scope.anon_regions(span, count)
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
    fn anon_regions(&self,
                    _span: Span,
                    _count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<ElisionFailureInfo>>> {
        Err(None)
    }

    fn object_lifetime_default(&self, span: Span) -> Option<ty::Region> {
        Some(self.base_object_lifetime_default(span))
    }

    fn base_object_lifetime_default(&self, _span: Span) -> ty::Region {
        ty::ReStatic
    }
}

// Same as `ExplicitRscope`, but provides some extra information for diagnostics
pub struct UnelidableRscope(Option<Vec<ElisionFailureInfo>>);

impl UnelidableRscope {
    pub fn new(v: Option<Vec<ElisionFailureInfo>>) -> UnelidableRscope {
        UnelidableRscope(v)
    }
}

impl RegionScope for UnelidableRscope {
    fn anon_regions(&self,
                    _span: Span,
                    _count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<ElisionFailureInfo>>> {
        let UnelidableRscope(ref v) = *self;
        Err(v.clone())
    }

    fn object_lifetime_default(&self, span: Span) -> Option<ty::Region> {
        Some(self.base_object_lifetime_default(span))
    }

    fn base_object_lifetime_default(&self, _span: Span) -> ty::Region {
        ty::ReStatic
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
    fn object_lifetime_default(&self, span: Span) -> Option<ty::Region> {
        // Per RFC #599, object-lifetimes default to 'static unless
        // overridden by context, and this takes precedence over
        // lifetime elision.
        Some(self.base_object_lifetime_default(span))
    }

    fn base_object_lifetime_default(&self, _span: Span) -> ty::Region {
        ty::ReStatic
    }

    fn anon_regions(&self,
                    _span: Span,
                    count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<ElisionFailureInfo>>>
    {
        Ok(vec![self.default; count])
    }
}

/// A scope that behaves as an ElidabeRscope with a `'static` default region
/// that should also warn if the `static_in_const` feature is unset.
#[derive(Copy, Clone)]
pub struct StaticRscope<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> {
    tcx: &'a ty::TyCtxt<'a, 'gcx, 'tcx>,
}

impl<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> StaticRscope<'a, 'gcx, 'tcx> {
    /// create a new StaticRscope from a reference to the `TyCtxt`
    pub fn new(tcx: &'a ty::TyCtxt<'a, 'gcx, 'tcx>) -> Self {
        StaticRscope { tcx: tcx }
    }
}

impl<'a, 'gcx: 'a + 'tcx, 'tcx: 'a> RegionScope for StaticRscope<'a, 'gcx, 'tcx> {
    fn anon_regions(&self,
                    span: Span,
                    count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<ElisionFailureInfo>>> {
        if !self.tcx.sess.features.borrow().static_in_const {
            self.tcx
                .sess
                .struct_span_err(span,
                                 "this needs a `'static` lifetime or the \
                                 `static_in_const` feature, see #35897")
                .emit();
        }
        Ok(vec![ty::ReStatic; count])
    }

    fn object_lifetime_default(&self, span: Span) -> Option<ty::Region> {
        Some(self.base_object_lifetime_default(span))
    }

    fn base_object_lifetime_default(&self, _span: Span) -> ty::Region {
        ty::ReStatic
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
    fn object_lifetime_default(&self, span: Span) -> Option<ty::Region> {
        // Per RFC #599, object-lifetimes default to 'static unless
        // overridden by context, and this takes precedence over the
        // binding defaults in a fn signature.
        Some(self.base_object_lifetime_default(span))
    }

    fn base_object_lifetime_default(&self, _span: Span) -> ty::Region {
        ty::ReStatic
    }

    fn anon_regions(&self,
                    _: Span,
                    count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<ElisionFailureInfo>>>
    {
        Ok((0..count).map(|_| self.next_region()).collect())
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

    fn anon_regions(&self,
                    span: Span,
                    count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<ElisionFailureInfo>>>
    {
        self.base_scope.anon_regions(span, count)
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

    fn anon_regions(&self,
                    span: Span,
                    count: usize)
                    -> Result<Vec<ty::Region>, Option<Vec<ElisionFailureInfo>>>
    {
        match self.base_scope.anon_regions(span, count) {
            Ok(mut v) => {
                for r in &mut v {
                    *r = ty::fold::shift_region(*r, 1);
                }
                Ok(v)
            }
            Err(errs) => {
                Err(errs)
            }
        }
    }

    fn anon_type_scope(&self) -> Option<AnonTypeScope> {
        self.base_scope.anon_type_scope()
    }
}
