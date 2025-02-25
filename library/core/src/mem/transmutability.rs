use crate::marker::{ConstParamTy_, UnsizedConstParamTy};

/// Marks that `Src` is transmutable into `Self`.
///
/// # Implementation
///
/// This trait cannot be implemented explicitly. It is implemented on-the-fly by
/// the compiler for all types `Src` and `Self` such that, given a set of safety
/// obligations on the programmer (see [`Assume`]), the compiler has proved that
/// the bits of a value of type `Src` can be soundly reinterpreted as a `Self`.
///
/// # Safety
///
/// If `Dst: TransmuteFrom<Src, ASSUMPTIONS>`, the compiler guarantees that
/// `Src` is soundly *union-transmutable* into a value of type `Dst`, provided
/// that the programmer has guaranteed that the given [`ASSUMPTIONS`](Assume)
/// are satisfied.
///
/// A union-transmute is any bit-reinterpretation conversion in the form of:
///
/// ```rust
/// pub unsafe fn transmute_via_union<Src, Dst>(src: Src) -> Dst {
///     use core::mem::ManuallyDrop;
///
///     #[repr(C)]
///     union Transmute<Src, Dst> {
///         src: ManuallyDrop<Src>,
///         dst: ManuallyDrop<Dst>,
///     }
///
///     let transmute = Transmute {
///         src: ManuallyDrop::new(src),
///     };
///
///     let dst = unsafe { transmute.dst };
///
///     ManuallyDrop::into_inner(dst)
/// }
/// ```
///
/// Note that this construction is more permissive than
/// [`mem::transmute_copy`](super::transmute_copy); union-transmutes permit
/// conversions that extend the bits of `Src` with trailing padding to fill
/// trailing uninitialized bytes of `Self`; e.g.:
///
/// ```rust
/// #![feature(transmutability)]
///
/// use core::mem::{Assume, TransmuteFrom};
///
/// let src = 42u8; // size = 1
///
/// #[repr(C, align(2))]
/// struct Dst(u8); // size = 2
//
/// let _ = unsafe {
///     <Dst as TransmuteFrom<u8, { Assume::SAFETY }>>::transmute(src)
/// };
/// ```
///
/// # Caveats
///
/// ## Portability
///
/// Implementations of this trait do not provide any guarantee of portability
/// across toolchains, targets or compilations. This trait may be implemented
/// for certain combinations of `Src`, `Self` and `ASSUME` on some toolchains,
/// targets or compilations, but not others. For example, if the layouts of
/// `Src` or `Self` are non-deterministic, the presence or absence of an
/// implementation of this trait may also be non-deterministic. Even if `Src`
/// and `Self` have deterministic layouts (e.g., they are `repr(C)` structs),
/// Rust does not specify the alignments of its primitive integer types, and
/// layouts that involve these types may vary across toolchains, targets or
/// compilations.
///
/// ## Stability
///
/// Implementations of this trait do not provide any guarantee of SemVer
/// stability across the crate versions that define the `Src` and `Self` types.
/// If SemVer stability is crucial to your application, you must consult the
/// documentation of `Src` and `Self`s' defining crates. Note that the presence
/// of `repr(C)`, alone, does not carry a safety invariant of SemVer stability.
/// Furthermore, stability does not imply portability. For example, the size of
/// `usize` is stable, but not portable.
#[unstable(feature = "transmutability", issue = "99571")]
#[lang = "transmute_trait"]
#[rustc_deny_explicit_impl]
#[rustc_do_not_implement_via_object]
#[rustc_coinductive]
pub unsafe trait TransmuteFrom<Src, const ASSUME: Assume = { Assume::NOTHING }>
where
    Src: ?Sized,
{
    /// Transmutes a `Src` value into a `Self`.
    ///
    /// # Safety
    ///
    /// The safety obligations of the caller depend on the value of `ASSUME`:
    /// - If [`ASSUME.alignment`](Assume::alignment), the caller must guarantee
    ///   that the addresses of references in the returned `Self` satisfy the
    ///   alignment requirements of their referent types.
    /// - If [`ASSUME.lifetimes`](Assume::lifetimes), the caller must guarantee
    ///   that references in the returned `Self` will not outlive their
    ///   referents.
    /// - If [`ASSUME.safety`](Assume::safety), the returned value might not
    ///   satisfy the library safety invariants of `Self`, and the caller must
    ///   guarantee that undefined behavior does not arise from uses of the
    ///   returned value.
    /// - If [`ASSUME.validity`](Assume::validity), the caller must guarantee
    ///   that `src` is a bit-valid instance of `Self`.
    ///
    /// When satisfying the above obligations (if any), the caller must *not*
    /// assume that this trait provides any inherent guarantee of layout
    /// [portability](#portability) or [stability](#stability).
    unsafe fn transmute(src: Src) -> Self
    where
        Src: Sized,
        Self: Sized,
    {
        use super::ManuallyDrop;

        #[repr(C)]
        union Transmute<Src, Dst> {
            src: ManuallyDrop<Src>,
            dst: ManuallyDrop<Dst>,
        }

        let transmute = Transmute { src: ManuallyDrop::new(src) };

        // SAFETY: It is safe to reinterpret the bits of `src` as a value of
        // type `Self`, because, by combination of invariant on this trait and
        // contract on the caller, `src` has been proven to satisfy both the
        // language and library invariants of `Self`. For all invariants not
        // `ASSUME`'d by the caller, the safety obligation is supplied by the
        // compiler. Conversely, for all invariants `ASSUME`'d by the caller,
        // the safety obligation is supplied by contract on the caller.
        let dst = unsafe { transmute.dst };

        ManuallyDrop::into_inner(dst)
    }
}

/// Configurable proof assumptions of [`TransmuteFrom`].
///
/// When `false`, the respective proof obligation belongs to the compiler. When
/// `true`, the onus of the safety proof belongs to the programmer.
#[unstable(feature = "transmutability", issue = "99571")]
#[lang = "transmute_opts"]
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct Assume {
    /// When `false`, [`TransmuteFrom`] is not implemented for transmutations
    /// that might violate the alignment requirements of references; e.g.:
    ///
    /// ```compile_fail,E0277
    /// #![feature(transmutability)]
    /// use core::mem::{align_of, TransmuteFrom};
    ///
    /// assert_eq!(align_of::<[u8; 2]>(), 1);
    /// assert_eq!(align_of::<u16>(), 2);
    ///
    /// let src: &[u8; 2] = &[0xFF, 0xFF];
    ///
    /// // SAFETY: No safety obligations.
    /// let dst: &u16 = unsafe {
    ///     <_ as TransmuteFrom<_>>::transmute(src)
    /// };
    /// ```
    ///
    /// When `true`, [`TransmuteFrom`] assumes that *you* have ensured
    /// that references in the transmuted value satisfy the alignment
    /// requirements of their referent types; e.g.:
    ///
    /// ```rust
    /// #![feature(pointer_is_aligned_to, transmutability)]
    /// use core::mem::{align_of, Assume, TransmuteFrom};
    ///
    /// let src: &[u8; 2] = &[0xFF, 0xFF];
    ///
    /// let maybe_dst: Option<&u16> = if <*const _>::is_aligned_to(src, align_of::<u16>()) {
    ///     // SAFETY: We have checked above that the address of `src` satisfies the
    ///     // alignment requirements of `u16`.
    ///     Some(unsafe {
    ///         <_ as TransmuteFrom<_, { Assume::ALIGNMENT }>>::transmute(src)
    ///     })
    /// } else {
    ///     None
    /// };
    ///
    /// assert!(matches!(maybe_dst, Some(&u16::MAX) | None));
    /// ```
    pub alignment: bool,

    /// When `false`, [`TransmuteFrom`] is not implemented for transmutations
    /// that extend the lifetimes of references.
    ///
    /// When `true`, [`TransmuteFrom`] assumes that *you* have ensured that
    /// references in the transmuted value do not outlive their referents.
    pub lifetimes: bool,

    /// When `false`, [`TransmuteFrom`] is not implemented for transmutations
    /// that might violate the library safety invariants of the destination
    /// type; e.g.:
    ///
    /// ```compile_fail,E0277
    /// #![feature(transmutability)]
    /// use core::mem::TransmuteFrom;
    ///
    /// let src: u8 = 3;
    ///
    /// struct EvenU8 {
    ///     // SAFETY: `val` must be an even number.
    ///     val: u8,
    /// }
    ///
    /// // SAFETY: No safety obligations.
    /// let dst: EvenU8 = unsafe {
    ///     <_ as TransmuteFrom<_>>::transmute(src)
    /// };
    /// ```
    ///
    /// When `true`, [`TransmuteFrom`] assumes that *you* have ensured
    /// that undefined behavior does not arise from using the transmuted value;
    /// e.g.:
    ///
    /// ```rust
    /// #![feature(transmutability)]
    /// use core::mem::{Assume, TransmuteFrom};
    ///
    /// let src: u8 = 42;
    ///
    /// struct EvenU8 {
    ///     // SAFETY: `val` must be an even number.
    ///     val: u8,
    /// }
    ///
    /// let maybe_dst: Option<EvenU8> = if src % 2 == 0 {
    ///     // SAFETY: We have checked above that the value of `src` is even.
    ///     Some(unsafe {
    ///         <_ as TransmuteFrom<_, { Assume::SAFETY }>>::transmute(src)
    ///     })
    /// } else {
    ///     None
    /// };
    ///
    /// assert!(matches!(maybe_dst, Some(EvenU8 { val: 42 })));
    /// ```
    pub safety: bool,

    /// When `false`, [`TransmuteFrom`] is not implemented for transmutations
    /// that might violate the language-level bit-validity invariant of the
    /// destination type; e.g.:
    ///
    /// ```compile_fail,E0277
    /// #![feature(transmutability)]
    /// use core::mem::TransmuteFrom;
    ///
    /// let src: u8 = 3;
    ///
    /// // SAFETY: No safety obligations.
    /// let dst: bool = unsafe {
    ///     <_ as TransmuteFrom<_>>::transmute(src)
    /// };
    /// ```
    ///
    /// When `true`, [`TransmuteFrom`] assumes that *you* have ensured
    /// that the value being transmuted is a bit-valid instance of the
    /// transmuted value; e.g.:
    ///
    /// ```rust
    /// #![feature(transmutability)]
    /// use core::mem::{Assume, TransmuteFrom};
    ///
    /// let src: u8 = 1;
    ///
    /// let maybe_dst: Option<bool> = if src == 0 || src == 1 {
    ///     // SAFETY: We have checked above that the value of `src` is a bit-valid
    ///     // instance of `bool`.
    ///     Some(unsafe {
    ///         <_ as TransmuteFrom<_, { Assume::VALIDITY }>>::transmute(src)
    ///     })
    /// } else {
    ///     None
    /// };
    ///
    /// assert_eq!(maybe_dst, Some(true));
    /// ```
    pub validity: bool,
}

#[unstable(feature = "transmutability", issue = "99571")]
impl ConstParamTy_ for Assume {}
#[unstable(feature = "transmutability", issue = "99571")]
impl UnsizedConstParamTy for Assume {}

impl Assume {
    /// With this, [`TransmuteFrom`] does not assume you have ensured any safety
    /// obligations are met, and relies only upon its own analysis to (dis)prove
    /// transmutability.
    #[unstable(feature = "transmutability", issue = "99571")]
    pub const NOTHING: Self =
        Self { alignment: false, lifetimes: false, safety: false, validity: false };

    /// With this, [`TransmuteFrom`] assumes only that you have ensured that
    /// references in the transmuted value satisfy the alignment requirements of
    /// their referent types. See [`Assume::alignment`] for examples.
    #[unstable(feature = "transmutability", issue = "99571")]
    pub const ALIGNMENT: Self = Self { alignment: true, ..Self::NOTHING };

    /// With this, [`TransmuteFrom`] assumes only that you have ensured that
    /// references in the transmuted value do not outlive their referents. See
    /// [`Assume::lifetimes`] for examples.
    #[unstable(feature = "transmutability", issue = "99571")]
    pub const LIFETIMES: Self = Self { lifetimes: true, ..Self::NOTHING };

    /// With this, [`TransmuteFrom`] assumes only that you have ensured that
    /// undefined behavior does not arise from using the transmuted value. See
    /// [`Assume::safety`] for examples.
    #[unstable(feature = "transmutability", issue = "99571")]
    pub const SAFETY: Self = Self { safety: true, ..Self::NOTHING };

    /// With this, [`TransmuteFrom`] assumes only that you have ensured that the
    /// value being transmuted is a bit-valid instance of the transmuted value.
    /// See [`Assume::validity`] for examples.
    #[unstable(feature = "transmutability", issue = "99571")]
    pub const VALIDITY: Self = Self { validity: true, ..Self::NOTHING };

    /// Combine the assumptions of `self` and `other_assumptions`.
    ///
    /// This is especially useful for extending [`Assume`] in generic contexts;
    /// e.g.:
    ///
    /// ```rust
    /// #![feature(
    ///     adt_const_params,
    ///     generic_const_exprs,
    ///     pointer_is_aligned_to,
    ///     transmutability,
    /// )]
    /// #![allow(incomplete_features)]
    /// use core::mem::{align_of, Assume, TransmuteFrom};
    ///
    /// /// Attempts to transmute `src` to `&Dst`.
    /// ///
    /// /// Returns `None` if `src` violates the alignment requirements of `&Dst`.
    /// ///
    /// /// # Safety
    /// ///
    /// /// The caller guarantees that the obligations required by `ASSUME`, except
    /// /// alignment, are satisfied.
    /// unsafe fn try_transmute_ref<'a, Src, Dst, const ASSUME: Assume>(src: &'a Src) -> Option<&'a Dst>
    /// where
    ///     &'a Dst: TransmuteFrom<&'a Src, { ASSUME.and(Assume::ALIGNMENT) }>,
    /// {
    ///     if <*const _>::is_aligned_to(src, align_of::<Dst>()) {
    ///         // SAFETY: By the above dynamic check, we have ensured that the address
    ///         // of `src` satisfies the alignment requirements of `&Dst`. By contract
    ///         // on the caller, the safety obligations required by `ASSUME` have also
    ///         // been satisfied.
    ///         Some(unsafe {
    ///             <_ as TransmuteFrom<_, { ASSUME.and(Assume::ALIGNMENT) }>>::transmute(src)
    ///         })
    ///     } else {
    ///         None
    ///     }
    /// }
    ///
    /// let src: &[u8; 2] = &[0xFF, 0xFF];
    ///
    /// // SAFETY: No safety obligations.
    /// let maybe_dst: Option<&u16> = unsafe {
    ///     try_transmute_ref::<_, _, { Assume::NOTHING }>(src)
    /// };
    ///```
    #[unstable(feature = "transmutability", issue = "99571")]
    pub const fn and(self, other_assumptions: Self) -> Self {
        Self {
            alignment: self.alignment || other_assumptions.alignment,
            lifetimes: self.lifetimes || other_assumptions.lifetimes,
            safety: self.safety || other_assumptions.safety,
            validity: self.validity || other_assumptions.validity,
        }
    }

    /// Remove `other_assumptions` the obligations of `self`; e.g.:
    ///
    /// ```rust
    /// #![feature(transmutability)]
    /// use core::mem::Assume;
    ///
    /// let assumptions = Assume::ALIGNMENT.and(Assume::SAFETY);
    /// let to_be_removed = Assume::SAFETY.and(Assume::VALIDITY);
    ///
    /// assert_eq!(
    ///     assumptions.but_not(to_be_removed),
    ///     Assume::ALIGNMENT,
    /// );
    /// ```
    #[unstable(feature = "transmutability", issue = "99571")]
    pub const fn but_not(self, other_assumptions: Self) -> Self {
        Self {
            alignment: self.alignment && !other_assumptions.alignment,
            lifetimes: self.lifetimes && !other_assumptions.lifetimes,
            safety: self.safety && !other_assumptions.safety,
            validity: self.validity && !other_assumptions.validity,
        }
    }
}

// FIXME(jswrenn): This const op is not actually usable. Why?
// https://github.com/rust-lang/rust/pull/100726#issuecomment-1219928926
#[unstable(feature = "transmutability", issue = "99571")]
impl core::ops::Add for Assume {
    type Output = Assume;

    fn add(self, other_assumptions: Assume) -> Assume {
        self.and(other_assumptions)
    }
}

// FIXME(jswrenn): This const op is not actually usable. Why?
// https://github.com/rust-lang/rust/pull/100726#issuecomment-1219928926
#[unstable(feature = "transmutability", issue = "99571")]
impl core::ops::Sub for Assume {
    type Output = Assume;

    fn sub(self, other_assumptions: Assume) -> Assume {
        self.but_not(other_assumptions)
    }
}
