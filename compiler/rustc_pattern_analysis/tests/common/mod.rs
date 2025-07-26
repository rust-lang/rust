#![allow(dead_code, unreachable_pub)]
use rustc_pattern_analysis::constructor::{
    Constructor, ConstructorSet, IntRange, MaybeInfiniteInt, RangeEnd, VariantVisibility,
};
use rustc_pattern_analysis::pat::DeconstructedPat;
use rustc_pattern_analysis::usefulness::{PlaceValidity, UsefulnessReport};
use rustc_pattern_analysis::{MatchArm, PatCx, PrivateUninhabitedField};

/// Sets up `tracing` for easier debugging. Tries to look like the `rustc` setup.
fn init_tracing() {
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    let _ = tracing_tree::HierarchicalLayer::default()
        .with_writer(std::io::stderr)
        .with_ansi(true)
        .with_targets(true)
        .with_indent_amount(2)
        .with_subscriber(
            tracing_subscriber::Registry::default()
                .with(tracing_subscriber::EnvFilter::from_default_env()),
        )
        .try_init();
}

pub(super) const UNIT: Ty = Ty::Tuple(&[]);
pub(super) const NEVER: Ty = Ty::Enum(&[]);

/// A simple set of types.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(super) enum Ty {
    /// Booleans
    Bool,
    /// 8-bit unsigned integers
    U8,
    /// Tuples.
    Tuple(&'static [Ty]),
    /// Enum with one variant of each given type.
    Enum(&'static [Ty]),
    /// A struct with `arity` fields of type `ty`.
    BigStruct { arity: usize, ty: &'static Ty },
    /// A enum with `arity` variants of type `ty`.
    BigEnum { arity: usize, ty: &'static Ty },
    /// Like `Enum` but non-exhaustive.
    NonExhaustiveEnum(&'static [Ty]),
}

/// The important logic.
impl Ty {
    pub(super) fn sub_tys(&self, ctor: &Constructor<Cx>) -> Vec<Self> {
        use Constructor::*;
        match (ctor, *self) {
            (Struct, Ty::Tuple(tys)) => tys.iter().copied().collect(),
            (Struct, Ty::BigStruct { arity, ty }) => (0..arity).map(|_| *ty).collect(),
            (Variant(i), Ty::Enum(tys) | Ty::NonExhaustiveEnum(tys)) => vec![tys[*i]],
            (Variant(_), Ty::BigEnum { ty, .. }) => vec![*ty],
            (Bool(..) | IntRange(..) | NonExhaustive | Missing | Wildcard, _) => vec![],
            _ => panic!("Unexpected ctor {ctor:?} for type {self:?}"),
        }
    }

    fn is_empty(&self) -> bool {
        match *self {
            Ty::Bool | Ty::U8 => false,
            Ty::Tuple(tys) => tys.iter().any(|ty| ty.is_empty()),
            Ty::Enum(tys) => tys.iter().all(|ty| ty.is_empty()),
            Ty::BigStruct { arity, ty } => arity != 0 && ty.is_empty(),
            Ty::BigEnum { arity, ty } => arity == 0 || ty.is_empty(),
            Ty::NonExhaustiveEnum(..) => false,
        }
    }

    fn ctor_set(&self) -> ConstructorSet<Cx> {
        match *self {
            Ty::Bool => ConstructorSet::Bool,
            Ty::U8 => ConstructorSet::Integers {
                range_1: IntRange::from_range(
                    MaybeInfiniteInt::new_finite_uint(0),
                    MaybeInfiniteInt::new_finite_uint(255),
                    RangeEnd::Included,
                ),
                range_2: None,
            },
            Ty::Tuple(..) | Ty::BigStruct { .. } => ConstructorSet::Struct { empty: false },
            Ty::Enum(tys) if tys.is_empty() => ConstructorSet::NoConstructors,
            Ty::Enum(tys) => ConstructorSet::Variants {
                variants: tys
                    .iter()
                    .map(|ty| {
                        if ty.is_empty() {
                            VariantVisibility::Empty
                        } else {
                            VariantVisibility::Visible
                        }
                    })
                    .collect(),
                non_exhaustive: false,
            },
            Ty::NonExhaustiveEnum(tys) => ConstructorSet::Variants {
                variants: tys
                    .iter()
                    .map(|ty| {
                        if ty.is_empty() {
                            VariantVisibility::Empty
                        } else {
                            VariantVisibility::Visible
                        }
                    })
                    .collect(),
                non_exhaustive: true,
            },
            Ty::BigEnum { arity: 0, .. } => ConstructorSet::NoConstructors,
            Ty::BigEnum { arity, ty } => {
                let vis = if ty.is_empty() {
                    VariantVisibility::Empty
                } else {
                    VariantVisibility::Visible
                };
                ConstructorSet::Variants {
                    variants: (0..arity).map(|_| vis).collect(),
                    non_exhaustive: false,
                }
            }
        }
    }

    fn write_variant_name(
        &self,
        f: &mut std::fmt::Formatter<'_>,
        ctor: &Constructor<Cx>,
    ) -> std::fmt::Result {
        match (*self, ctor) {
            (Ty::Tuple(..), _) => Ok(()),
            (Ty::BigStruct { .. }, _) => write!(f, "BigStruct"),
            (Ty::Enum(..) | Ty::NonExhaustiveEnum(..), Constructor::Variant(i)) => {
                write!(f, "Enum::Variant{i}")
            }
            (Ty::BigEnum { .. }, Constructor::Variant(i)) => write!(f, "BigEnum::Variant{i}"),
            _ => write!(f, "{:?}::{:?}", self, ctor),
        }
    }
}

/// Compute usefulness in our simple context (and set up tracing for easier debugging).
pub(super) fn compute_match_usefulness<'p>(
    arms: &[MatchArm<'p, Cx>],
    ty: Ty,
    scrut_validity: PlaceValidity,
    complexity_limit: usize,
    exhaustive_witnesses: bool,
) -> Result<UsefulnessReport<'p, Cx>, ()> {
    init_tracing();
    rustc_pattern_analysis::usefulness::compute_match_usefulness(
        &Cx { exhaustive_witnesses },
        arms,
        ty,
        scrut_validity,
        complexity_limit,
    )
}

#[derive(Debug)]
pub(super) struct Cx {
    exhaustive_witnesses: bool,
}

/// The context for pattern analysis. Forwards anything interesting to `Ty` methods.
impl PatCx for Cx {
    type Ty = Ty;
    type Error = ();
    type VariantIdx = usize;
    type StrLit = ();
    type ArmData = ();
    type PatData = ();

    fn is_exhaustive_patterns_feature_on(&self) -> bool {
        false
    }

    fn exhaustive_witnesses(&self) -> bool {
        self.exhaustive_witnesses
    }

    fn ctor_arity(&self, ctor: &Constructor<Self>, ty: &Self::Ty) -> usize {
        ty.sub_tys(ctor).len()
    }

    fn ctor_sub_tys(
        &self,
        ctor: &Constructor<Self>,
        ty: &Self::Ty,
    ) -> impl Iterator<Item = (Self::Ty, PrivateUninhabitedField)> + ExactSizeIterator {
        ty.sub_tys(ctor).into_iter().map(|ty| (ty, PrivateUninhabitedField(false)))
    }

    fn ctors_for_ty(&self, ty: &Self::Ty) -> Result<ConstructorSet<Self>, Self::Error> {
        Ok(ty.ctor_set())
    }

    fn write_variant_name(
        f: &mut std::fmt::Formatter<'_>,
        ctor: &Constructor<Self>,
        ty: &Self::Ty,
    ) -> std::fmt::Result {
        ty.write_variant_name(f, ctor)
    }

    fn bug(&self, fmt: std::fmt::Arguments<'_>) -> Self::Error {
        panic!("{}", fmt)
    }

    /// Abort when reaching the complexity limit. This is what we'll check in tests.
    fn complexity_exceeded(&self) -> Result<(), Self::Error> {
        Err(())
    }

    fn report_mixed_deref_pat_ctors(
        &self,
        _deref_pat: &DeconstructedPat<Self>,
        _normal_pat: &DeconstructedPat<Self>,
    ) -> Self::Error {
        panic!("`rustc_pattern_analysis::tests` currently doesn't test deref pattern errors")
    }
}

/// Construct a single pattern; see `pats!()`.
#[allow(unused_macros)]
macro_rules! pat {
    ($($rest:tt)*) => {{
        let mut vec = pats!($($rest)*);
        vec.pop().unwrap()
    }};
}

/// A macro to construct patterns. Called like `pats!(type_expr; pattern, pattern, ..)` and returns
/// a `Vec<DeconstructedPat>`. A pattern can be nested and looks like `Constructor(pat, pat)` or
/// `Constructor { .i: pat, .j: pat }`, where `Constructor` is `Struct`, `Variant.i` (with index
/// `i`), as well as booleans and integer ranges.
///
/// The general structure of the macro is a tt-muncher with several stages identified with
/// `@something(args)`. The args are a key-value list (the keys ensure we don't mix the arguments
/// around) which is passed down and modified as needed. We then parse token-trees from
/// left-to-right. Non-trivial recursion happens when we parse the arguments to a pattern: we
/// recurse to parse the tokens inside `{..}`/`(..)`, and then we continue parsing anything that
/// follows.
macro_rules! pats {
    // Entrypoint
    // Parse `type; ..`
    ($ty:expr; $($rest:tt)*) => {{
        #[allow(unused)]
        use rustc_pattern_analysis::{
            constructor::{Constructor, IntRange, MaybeInfiniteInt, RangeEnd},
            pat::{DeconstructedPat, IndexedPat},
        };
        let ty = $ty;
        // The heart of the macro is designed to push `IndexedPat`s into a `Vec`, so we work around
        // that.
        #[allow(unused)]
        let sub_tys = ::std::iter::repeat(&ty);
        #[allow(unused)]
        let mut vec: Vec<IndexedPat<_>> = Vec::new();
        pats!(@ctor(vec:vec, sub_tys:sub_tys, idx:0) $($rest)*);
        vec.into_iter().map(|ipat| ipat.pat).collect::<Vec<_>>()
    }};

    // Parse `constructor ..`

    (@ctor($($args:tt)*) true $($rest:tt)*) => {{
        let ctor = Constructor::Bool(true);
        pats!(@pat($($args)*, ctor:ctor) $($rest)*)
    }};
    (@ctor($($args:tt)*) false $($rest:tt)*) => {{
        let ctor = Constructor::Bool(false);
        pats!(@pat($($args)*, ctor:ctor) $($rest)*)
    }};
    (@ctor($($args:tt)*) Struct $($rest:tt)*) => {{
        let ctor = Constructor::Struct;
        pats!(@pat($($args)*, ctor:ctor) $($rest)*)
    }};
    (@ctor($($args:tt)*) ( $($fields:tt)* ) $($rest:tt)*) => {{
        let ctor = Constructor::Struct; // tuples
        pats!(@pat($($args)*, ctor:ctor) ( $($fields)* ) $($rest)*)
    }};
    (@ctor($($args:tt)*) Variant.$variant:ident $($rest:tt)*) => {{
        let ctor = Constructor::Variant($variant);
        pats!(@pat($($args)*, ctor:ctor) $($rest)*)
    }};
    (@ctor($($args:tt)*) Variant.$variant:literal $($rest:tt)*) => {{
        let ctor = Constructor::Variant($variant);
        pats!(@pat($($args)*, ctor:ctor) $($rest)*)
    }};
    (@ctor($($args:tt)*) _ $($rest:tt)*) => {{
        let ctor = Constructor::Wildcard;
        pats!(@pat($($args)*, ctor:ctor) $($rest)*)
    }};
    // Nothing
    (@ctor($($args:tt)*)) => {};

    // Integers and int ranges
    (@ctor($($args:tt)*) $($start:literal)?..$end:literal $($rest:tt)*) => {{
        let ctor = Constructor::IntRange(IntRange::from_range(
            pats!(@rangeboundary- $($start)?),
            pats!(@rangeboundary+ $end),
            RangeEnd::Excluded,
        ));
        pats!(@pat($($args)*, ctor:ctor) $($rest)*)
    }};
    (@ctor($($args:tt)*) $($start:literal)?.. $($rest:tt)*) => {{
        let ctor = Constructor::IntRange(IntRange::from_range(
            pats!(@rangeboundary- $($start)?),
            pats!(@rangeboundary+),
            RangeEnd::Excluded,
        ));
        pats!(@pat($($args)*, ctor:ctor) $($rest)*)
    }};
    (@ctor($($args:tt)*) $($start:literal)?..=$end:literal $($rest:tt)*) => {{
        let ctor = Constructor::IntRange(IntRange::from_range(
            pats!(@rangeboundary- $($start)?),
            pats!(@rangeboundary+ $end),
            RangeEnd::Included,
        ));
        pats!(@pat($($args)*, ctor:ctor) $($rest)*)
    }};
    (@ctor($($args:tt)*) $int:literal $($rest:tt)*) => {{
        let ctor = Constructor::IntRange(IntRange::from_range(
            pats!(@rangeboundary- $int),
            pats!(@rangeboundary+ $int),
            RangeEnd::Included,
        ));
        pats!(@pat($($args)*, ctor:ctor) $($rest)*)
    }};
    // Utility to manage range boundaries.
    (@rangeboundary $sign:tt $int:literal) => { MaybeInfiniteInt::new_finite_uint($int) };
    (@rangeboundary -) => { MaybeInfiniteInt::NegInfinity };
    (@rangeboundary +) => { MaybeInfiniteInt::PosInfinity };

    // Parse subfields: `(..)` or `{..}`

    // Constructor with no fields, e.g. `bool` or `Variant.1`.
    (@pat($($args:tt)*) $(,)?) => {
        pats!(@pat($($args)*) {})
    };
    (@pat($($args:tt)*) , $($rest:tt)*) => {
        pats!(@pat($($args)*) {}, $($rest)*)
    };
    // `(..)` and `{..}` are treated the same.
    (@pat($($args:tt)*) ( $($subpat:tt)* ) $($rest:tt)*) => {{
        pats!(@pat($($args)*) { $($subpat)* } $($rest)*)
    }};
    (@pat(vec:$vec:expr, sub_tys:$sub_tys:expr, idx:$idx:expr, ctor:$ctor:expr) { $($fields:tt)* } $($rest:tt)*) => {{
        let sub_tys = $sub_tys;
        let index = $idx;
        // Silly dance to work with both a vec and `iter::repeat()`.
        let ty = *(&sub_tys).clone().into_iter().nth(index).unwrap();
        let ctor = $ctor;
        let ctor_sub_tys = &ty.sub_tys(&ctor);
        #[allow(unused_mut)]
        let mut fields = Vec::new();
        // Parse subpatterns (note the leading comma).
        pats!(@fields(idx:0, vec:fields, sub_tys:ctor_sub_tys) ,$($fields)*);
        let arity = ctor_sub_tys.len();
        let pat = DeconstructedPat::new(ctor, fields, arity, ty, ()).at_index(index);
        $vec.push(pat);

        // Continue parsing further patterns.
        pats!(@fields(idx:index+1, vec:$vec, sub_tys:sub_tys) $($rest)*);
    }};

    // Parse fields one by one.

    // No fields left.
    (@fields($($args:tt)*) $(,)?) => {};
    // `.i: pat` sets the current index to `i`.
    (@fields(idx:$_idx:expr, $($args:tt)*) , .$idx:literal : $($rest:tt)*) => {{
        pats!(@ctor($($args)*, idx:$idx) $($rest)*);
    }};
    (@fields(idx:$_idx:expr, $($args:tt)*) , .$idx:ident : $($rest:tt)*) => {{
        pats!(@ctor($($args)*, idx:$idx) $($rest)*);
    }};
    // Field without an explicit index; we use the current index which gets incremented above.
    (@fields(idx:$idx:expr, $($args:tt)*) , $($rest:tt)*) => {{
        pats!(@ctor($($args)*, idx:$idx) $($rest)*);
    }};
}
