bitflags::bitflags! {
    /// Flags that we track on types. These flags are propagated upwards
    /// through the type during type construction, so that we can quickly check
    /// whether the type has various kinds of types in it without recursing
    /// over the type itself.
    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    pub struct TypeFlags: u32 {
        // Does this have parameters? Used to determine whether instantiation is
        // required.
        /// Does this have `Param`?
        const HAS_TY_PARAM                = 1 << 0;
        /// Does this have `ReEarlyParam`?
        const HAS_RE_PARAM                = 1 << 1;
        /// Does this have `ConstKind::Param`?
        const HAS_CT_PARAM                = 1 << 2;

        const HAS_PARAM                   = TypeFlags::HAS_TY_PARAM.bits()
                                          | TypeFlags::HAS_RE_PARAM.bits()
                                          | TypeFlags::HAS_CT_PARAM.bits();

        /// Does this have `Infer`?
        const HAS_TY_INFER                = 1 << 3;
        /// Does this have `ReVar`?
        const HAS_RE_INFER                = 1 << 4;
        /// Does this have `ConstKind::Infer`?
        const HAS_CT_INFER                = 1 << 5;

        /// Does this have inference variables? Used to determine whether
        /// inference is required.
        const HAS_INFER                   = TypeFlags::HAS_TY_INFER.bits()
                                          | TypeFlags::HAS_RE_INFER.bits()
                                          | TypeFlags::HAS_CT_INFER.bits();

        /// Does this have `Placeholder`?
        const HAS_TY_PLACEHOLDER          = 1 << 6;
        /// Does this have `RePlaceholder`?
        const HAS_RE_PLACEHOLDER          = 1 << 7;
        /// Does this have `ConstKind::Placeholder`?
        const HAS_CT_PLACEHOLDER          = 1 << 8;

        /// Does this have placeholders?
        const HAS_PLACEHOLDER             = TypeFlags::HAS_TY_PLACEHOLDER.bits()
                                          | TypeFlags::HAS_RE_PLACEHOLDER.bits()
                                          | TypeFlags::HAS_CT_PLACEHOLDER.bits();

        /// `true` if there are "names" of regions and so forth
        /// that are local to a particular fn/inferctxt
        const HAS_FREE_LOCAL_REGIONS      = 1 << 9;

        /// `true` if there are "names" of types and regions and so forth
        /// that are local to a particular fn
        const HAS_FREE_LOCAL_NAMES        = TypeFlags::HAS_TY_PARAM.bits()
                                          | TypeFlags::HAS_CT_PARAM.bits()
                                          | TypeFlags::HAS_TY_INFER.bits()
                                          | TypeFlags::HAS_CT_INFER.bits()
                                          | TypeFlags::HAS_TY_PLACEHOLDER.bits()
                                          | TypeFlags::HAS_CT_PLACEHOLDER.bits()
                                          // We consider 'freshened' types and constants
                                          // to depend on a particular fn.
                                          // The freshening process throws away information,
                                          // which can make things unsuitable for use in a global
                                          // cache. Note that there is no 'fresh lifetime' flag -
                                          // freshening replaces all lifetimes with `ReErased`,
                                          // which is different from how types/const are freshened.
                                          | TypeFlags::HAS_TY_FRESH.bits()
                                          | TypeFlags::HAS_CT_FRESH.bits()
                                          | TypeFlags::HAS_FREE_LOCAL_REGIONS.bits()
                                          | TypeFlags::HAS_RE_ERASED.bits();

        /// Does this have `Projection`?
        const HAS_TY_PROJECTION           = 1 << 10;
        /// Does this have `Weak`?
        const HAS_TY_WEAK                 = 1 << 11;
        /// Does this have `Opaque`?
        const HAS_TY_OPAQUE               = 1 << 12;
        /// Does this have `Inherent`?
        const HAS_TY_INHERENT             = 1 << 13;
        /// Does this have `ConstKind::Unevaluated`?
        const HAS_CT_PROJECTION           = 1 << 14;

        /// Does this have `Alias` or `ConstKind::Unevaluated`?
        ///
        /// Rephrased, could this term be normalized further?
        const HAS_ALIAS                   = TypeFlags::HAS_TY_PROJECTION.bits()
                                          | TypeFlags::HAS_TY_WEAK.bits()
                                          | TypeFlags::HAS_TY_OPAQUE.bits()
                                          | TypeFlags::HAS_TY_INHERENT.bits()
                                          | TypeFlags::HAS_CT_PROJECTION.bits();

        /// Is an error type/lifetime/const reachable?
        const HAS_ERROR                   = 1 << 15;

        /// Does this have any region that "appears free" in the type?
        /// Basically anything but `ReBound` and `ReErased`.
        const HAS_FREE_REGIONS            = 1 << 16;

        /// Does this have any `ReBound` regions?
        const HAS_RE_BOUND                = 1 << 17;
        /// Does this have any `Bound` types?
        const HAS_TY_BOUND                = 1 << 18;
        /// Does this have any `ConstKind::Bound` consts?
        const HAS_CT_BOUND                = 1 << 19;
        /// Does this have any bound variables?
        /// Used to check if a global bound is safe to evaluate.
        const HAS_BOUND_VARS              = TypeFlags::HAS_RE_BOUND.bits()
                                          | TypeFlags::HAS_TY_BOUND.bits()
                                          | TypeFlags::HAS_CT_BOUND.bits();

        /// Does this have any `ReErased` regions?
        const HAS_RE_ERASED               = 1 << 20;

        /// Does this value have parameters/placeholders/inference variables which could be
        /// replaced later, in a way that would change the results of `impl` specialization?
        const STILL_FURTHER_SPECIALIZABLE = TypeFlags::HAS_TY_PARAM.bits()
                                          | TypeFlags::HAS_TY_PLACEHOLDER.bits()
                                          | TypeFlags::HAS_TY_INFER.bits()
                                          | TypeFlags::HAS_CT_PARAM.bits()
                                          | TypeFlags::HAS_CT_PLACEHOLDER.bits()
                                          | TypeFlags::HAS_CT_INFER.bits();

        /// Does this value have `InferTy::FreshTy/FreshIntTy/FreshFloatTy`?
        const HAS_TY_FRESH                = 1 << 21;

        /// Does this value have `InferConst::Fresh`?
        const HAS_CT_FRESH                = 1 << 22;

        /// Does this have any binders with bound vars (e.g. that need to be anonymized)?
        const HAS_BINDER_VARS             = 1 << 23;
    }
}
