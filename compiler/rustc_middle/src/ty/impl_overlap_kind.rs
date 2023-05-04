#[derive(Debug, PartialEq, Eq)]
pub enum ImplOverlapKind {
    /// These impls are always allowed to overlap.
    Permitted {
        /// Whether or not the impl is permitted due to the trait being a `#[marker]` trait
        marker: bool,
    },
    /// These impls are allowed to overlap, but that raises
    /// an issue #33140 future-compatibility warning.
    ///
    /// Some background: in Rust 1.0, the trait-object types `Send + Sync` (today's
    /// `dyn Send + Sync`) and `Sync + Send` (now `dyn Sync + Send`) were different.
    ///
    /// The widely-used version 0.1.0 of the crate `traitobject` had accidentally relied
    /// that difference, making what reduces to the following set of impls:
    ///
    /// ```compile_fail,(E0119)
    /// trait Trait {}
    /// impl Trait for dyn Send + Sync {}
    /// impl Trait for dyn Sync + Send {}
    /// ```
    ///
    /// Obviously, once we made these types be identical, that code causes a coherence
    /// error and a fairly big headache for us. However, luckily for us, the trait
    /// `Trait` used in this case is basically a marker trait, and therefore having
    /// overlapping impls for it is sound.
    ///
    /// To handle this, we basically regard the trait as a marker trait, with an additional
    /// future-compatibility warning. To avoid accidentally "stabilizing" this feature,
    /// it has the following restrictions:
    ///
    /// 1. The trait must indeed be a marker-like trait (i.e., no items), and must be
    /// positive impls.
    /// 2. The trait-ref of both impls must be equal.
    /// 3. The trait-ref of both impls must be a trait object type consisting only of
    /// marker traits.
    /// 4. Neither of the impls can have any where-clauses.
    ///
    /// Once `traitobject` 0.1.0 is no longer an active concern, this hack can be removed.
    Issue33140,
}
