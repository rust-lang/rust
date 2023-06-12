use crate::marker::ConstParamTy;

/// Are values of a type transmutable into values of another type?
///
/// This trait is implemented on-the-fly by the compiler for types `Src` and `Self` when the bits of
/// any value of type `Self` are safely transmutable into a value of type `Dst`, in a given `Context`,
/// notwithstanding whatever safety checks you have asked the compiler to [`Assume`] are satisfied.
#[unstable(feature = "transmutability", issue = "99571")]
#[lang = "transmute_trait"]
pub unsafe trait BikeshedIntrinsicFrom<Src, Context, const ASSUME: Assume = { Assume::NOTHING }>
where
    Src: ?Sized,
{
}

/// What transmutation safety conditions shall the compiler assume that *you* are checking?
#[unstable(feature = "transmutability", issue = "99571")]
#[lang = "transmute_opts"]
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct Assume {
    /// When `true`, the compiler assumes that *you* are ensuring (either dynamically or statically) that
    /// destination referents do not have stricter alignment requirements than source referents.
    pub alignment: bool,

    /// When `true`, the compiler assume that *you* are ensuring that lifetimes are not extended in a manner
    /// that violates Rust's memory model.
    pub lifetimes: bool,

    /// When `true`, the compiler assumes that *you* have ensured that it is safe for you to violate the
    /// type and field privacy of the destination type (and sometimes of the source type, too).
    pub safety: bool,

    /// When `true`, the compiler assumes that *you* are ensuring that the source type is actually a valid
    /// instance of the destination type.
    pub validity: bool,
}

#[unstable(feature = "transmutability", issue = "99571")]
impl ConstParamTy for Assume {}

impl Assume {
    /// Do not assume that *you* have ensured any safety properties are met.
    #[unstable(feature = "transmutability", issue = "99571")]
    pub const NOTHING: Self =
        Self { alignment: false, lifetimes: false, safety: false, validity: false };

    /// Assume only that alignment conditions are met.
    #[unstable(feature = "transmutability", issue = "99571")]
    pub const ALIGNMENT: Self = Self { alignment: true, ..Self::NOTHING };

    /// Assume only that lifetime conditions are met.
    #[unstable(feature = "transmutability", issue = "99571")]
    pub const LIFETIMES: Self = Self { lifetimes: true, ..Self::NOTHING };

    /// Assume only that safety conditions are met.
    #[unstable(feature = "transmutability", issue = "99571")]
    pub const SAFETY: Self = Self { safety: true, ..Self::NOTHING };

    /// Assume only that dynamically-satisfiable validity conditions are met.
    #[unstable(feature = "transmutability", issue = "99571")]
    pub const VALIDITY: Self = Self { validity: true, ..Self::NOTHING };

    /// Assume both `self` and `other_assumptions`.
    #[unstable(feature = "transmutability", issue = "99571")]
    pub const fn and(self, other_assumptions: Self) -> Self {
        Self {
            alignment: self.alignment || other_assumptions.alignment,
            lifetimes: self.lifetimes || other_assumptions.lifetimes,
            safety: self.safety || other_assumptions.safety,
            validity: self.validity || other_assumptions.validity,
        }
    }

    /// Assume `self`, excepting `other_assumptions`.
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
