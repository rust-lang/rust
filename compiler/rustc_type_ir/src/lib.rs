#![feature(never_type)]
#![feature(const_panic)]
#![feature(control_flow_enum)]

#[macro_use]
extern crate bitflags;

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};

bitflags! {
    /// Flags that we track on types. These flags are propagated upwards
    /// through the type during type construction, so that we can quickly check
    /// whether the type has various kinds of types in it without recursing
    /// over the type itself.
    pub struct TypeFlags: u32 {
        // Does this have parameters? Used to determine whether substitution is
        // required.
        /// Does this have `Param`?
        const HAS_TY_PARAM                = 1 << 0;
        /// Does this have `ReEarlyBound`?
        const HAS_RE_PARAM                = 1 << 1;
        /// Does this have `ConstKind::Param`?
        const HAS_CT_PARAM                = 1 << 2;

        const NEEDS_SUBST                 = TypeFlags::HAS_TY_PARAM.bits
                                          | TypeFlags::HAS_RE_PARAM.bits
                                          | TypeFlags::HAS_CT_PARAM.bits;

        /// Does this have `Infer`?
        const HAS_TY_INFER                = 1 << 3;
        /// Does this have `ReVar`?
        const HAS_RE_INFER                = 1 << 4;
        /// Does this have `ConstKind::Infer`?
        const HAS_CT_INFER                = 1 << 5;

        /// Does this have inference variables? Used to determine whether
        /// inference is required.
        const NEEDS_INFER                 = TypeFlags::HAS_TY_INFER.bits
                                          | TypeFlags::HAS_RE_INFER.bits
                                          | TypeFlags::HAS_CT_INFER.bits;

        /// Does this have `Placeholder`?
        const HAS_TY_PLACEHOLDER          = 1 << 6;
        /// Does this have `RePlaceholder`?
        const HAS_RE_PLACEHOLDER          = 1 << 7;
        /// Does this have `ConstKind::Placeholder`?
        const HAS_CT_PLACEHOLDER          = 1 << 8;

        /// `true` if there are "names" of regions and so forth
        /// that are local to a particular fn/inferctxt
        const HAS_FREE_LOCAL_REGIONS      = 1 << 9;

        /// `true` if there are "names" of types and regions and so forth
        /// that are local to a particular fn
        const HAS_FREE_LOCAL_NAMES        = TypeFlags::HAS_TY_PARAM.bits
                                          | TypeFlags::HAS_CT_PARAM.bits
                                          | TypeFlags::HAS_TY_INFER.bits
                                          | TypeFlags::HAS_CT_INFER.bits
                                          | TypeFlags::HAS_TY_PLACEHOLDER.bits
                                          | TypeFlags::HAS_CT_PLACEHOLDER.bits
                                          | TypeFlags::HAS_FREE_LOCAL_REGIONS.bits;

        /// Does this have `Projection`?
        const HAS_TY_PROJECTION           = 1 << 10;
        /// Does this have `Opaque`?
        const HAS_TY_OPAQUE               = 1 << 11;
        /// Does this have `ConstKind::Unevaluated`?
        const HAS_CT_PROJECTION           = 1 << 12;

        /// Could this type be normalized further?
        const HAS_PROJECTION              = TypeFlags::HAS_TY_PROJECTION.bits
                                          | TypeFlags::HAS_TY_OPAQUE.bits
                                          | TypeFlags::HAS_CT_PROJECTION.bits;

        /// Is an error type/const reachable?
        const HAS_ERROR                   = 1 << 13;

        /// Does this have any region that "appears free" in the type?
        /// Basically anything but `ReLateBound` and `ReErased`.
        const HAS_FREE_REGIONS            = 1 << 14;

        /// Does this have any `ReLateBound` regions? Used to check
        /// if a global bound is safe to evaluate.
        const HAS_RE_LATE_BOUND           = 1 << 15;

        /// Does this have any `ReErased` regions?
        const HAS_RE_ERASED               = 1 << 16;

        /// Does this value have parameters/placeholders/inference variables which could be
        /// replaced later, in a way that would change the results of `impl` specialization?
        const STILL_FURTHER_SPECIALIZABLE = 1 << 17;
    }
}

rustc_index::newtype_index! {
    /// A [De Bruijn index][dbi] is a standard means of representing
    /// regions (and perhaps later types) in a higher-ranked setting. In
    /// particular, imagine a type like this:
    ///
    ///     for<'a> fn(for<'b> fn(&'b isize, &'a isize), &'a char)
    ///     ^          ^            |          |           |
    ///     |          |            |          |           |
    ///     |          +------------+ 0        |           |
    ///     |                                  |           |
    ///     +----------------------------------+ 1         |
    ///     |                                              |
    ///     +----------------------------------------------+ 0
    ///
    /// In this type, there are two binders (the outer fn and the inner
    /// fn). We need to be able to determine, for any given region, which
    /// fn type it is bound by, the inner or the outer one. There are
    /// various ways you can do this, but a De Bruijn index is one of the
    /// more convenient and has some nice properties. The basic idea is to
    /// count the number of binders, inside out. Some examples should help
    /// clarify what I mean.
    ///
    /// Let's start with the reference type `&'b isize` that is the first
    /// argument to the inner function. This region `'b` is assigned a De
    /// Bruijn index of 0, meaning "the innermost binder" (in this case, a
    /// fn). The region `'a` that appears in the second argument type (`&'a
    /// isize`) would then be assigned a De Bruijn index of 1, meaning "the
    /// second-innermost binder". (These indices are written on the arrows
    /// in the diagram).
    ///
    /// What is interesting is that De Bruijn index attached to a particular
    /// variable will vary depending on where it appears. For example,
    /// the final type `&'a char` also refers to the region `'a` declared on
    /// the outermost fn. But this time, this reference is not nested within
    /// any other binders (i.e., it is not an argument to the inner fn, but
    /// rather the outer one). Therefore, in this case, it is assigned a
    /// De Bruijn index of 0, because the innermost binder in that location
    /// is the outer fn.
    ///
    /// [dbi]: https://en.wikipedia.org/wiki/De_Bruijn_index
    pub struct DebruijnIndex {
        DEBUG_FORMAT = "DebruijnIndex({})",
        const INNERMOST = 0,
    }
}

impl DebruijnIndex {
    /// Returns the resulting index when this value is moved into
    /// `amount` number of new binders. So, e.g., if you had
    ///
    ///    for<'a> fn(&'a x)
    ///
    /// and you wanted to change it to
    ///
    ///    for<'a> fn(for<'b> fn(&'a x))
    ///
    /// you would need to shift the index for `'a` into a new binder.
    #[must_use]
    pub fn shifted_in(self, amount: u32) -> DebruijnIndex {
        DebruijnIndex::from_u32(self.as_u32() + amount)
    }

    /// Update this index in place by shifting it "in" through
    /// `amount` number of binders.
    pub fn shift_in(&mut self, amount: u32) {
        *self = self.shifted_in(amount);
    }

    /// Returns the resulting index when this value is moved out from
    /// `amount` number of new binders.
    #[must_use]
    pub fn shifted_out(self, amount: u32) -> DebruijnIndex {
        DebruijnIndex::from_u32(self.as_u32() - amount)
    }

    /// Update in place by shifting out from `amount` binders.
    pub fn shift_out(&mut self, amount: u32) {
        *self = self.shifted_out(amount);
    }

    /// Adjusts any De Bruijn indices so as to make `to_binder` the
    /// innermost binder. That is, if we have something bound at `to_binder`,
    /// it will now be bound at INNERMOST. This is an appropriate thing to do
    /// when moving a region out from inside binders:
    ///
    /// ```
    ///             for<'a>   fn(for<'b>   for<'c>   fn(&'a u32), _)
    /// // Binder:  D3           D2        D1            ^^
    /// ```
    ///
    /// Here, the region `'a` would have the De Bruijn index D3,
    /// because it is the bound 3 binders out. However, if we wanted
    /// to refer to that region `'a` in the second argument (the `_`),
    /// those two binders would not be in scope. In that case, we
    /// might invoke `shift_out_to_binder(D3)`. This would adjust the
    /// De Bruijn index of `'a` to D1 (the innermost binder).
    ///
    /// If we invoke `shift_out_to_binder` and the region is in fact
    /// bound by one of the binders we are shifting out of, that is an
    /// error (and should fail an assertion failure).
    pub fn shifted_out_to_binder(self, to_binder: DebruijnIndex) -> Self {
        self.shifted_out(to_binder.as_u32() - INNERMOST.as_u32())
    }
}

impl<CTX> HashStable<CTX> for DebruijnIndex {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        self.as_u32().hash_stable(ctx, hasher);
    }
}
