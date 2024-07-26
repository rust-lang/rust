#![allow(clippy::derived_hash_with_manual_eq)]

use derive_where::derive_where;

#[cfg(feature = "nightly")]
use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
#[cfg(feature = "nightly")]
use rustc_data_structures::unify::{NoError, UnifyKey, UnifyValue};
#[cfg(feature = "nightly")]
use rustc_macros::{Decodable, Encodable, HashStable_NoContext, TyDecodable, TyEncodable};
use rustc_type_ir_macros::{Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic};
use std::fmt;

pub use self::closure::*;
use self::TyKind::*;
use crate::inherent::*;
use crate::{self as ty, DebruijnIndex, Interner};

use rustc_ast_ir::Mutability;

mod closure;

/// Specifies how a trait object is represented.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable, HashStable_NoContext))]
pub enum DynKind {
    /// An unsized `dyn Trait` object
    Dyn,
    /// A sized `dyn* Trait` object
    ///
    /// These objects are represented as a `(data, vtable)` pair where `data` is a value of some
    /// ptr-sized and ptr-aligned dynamically determined type `T` and `vtable` is a pointer to the
    /// vtable of `impl T for Trait`. This allows a `dyn*` object to be treated agnostically with
    /// respect to whether it points to a `Box<T>`, `Rc<T>`, etc.
    DynStar,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable, HashStable_NoContext))]
pub enum AliasTyKind {
    /// A projection `<Type as Trait>::AssocType`.
    /// Can get normalized away if monomorphic enough.
    Projection,
    /// An associated type in an inherent `impl`
    Inherent,
    /// An opaque type (usually from `impl Trait` in type aliases or function return types)
    /// Can only be normalized away in RevealAll mode
    Opaque,
    /// A type alias that actually checks its trait bounds.
    /// Currently only used if the type alias references opaque types.
    /// Can always be normalized away.
    Weak,
}

impl AliasTyKind {
    pub fn descr(self) -> &'static str {
        match self {
            AliasTyKind::Projection => "associated type",
            AliasTyKind::Inherent => "inherent associated type",
            AliasTyKind::Opaque => "opaque type",
            AliasTyKind::Weak => "type alias",
        }
    }
}

/// Defines the kinds of types used by the type system.
///
/// Types written by the user start out as `hir::TyKind` and get
/// converted to this representation using `<dyn HirTyLowerer>::lower_ty`.
#[cfg_attr(feature = "nightly", rustc_diagnostic_item = "IrTyKind")]
#[derive_where(Clone, Copy, Hash, Eq; I: Interner)]
#[cfg_attr(feature = "nightly", derive(TyEncodable, TyDecodable, HashStable_NoContext))]
pub enum TyKind<I: Interner> {
    /// The primitive boolean type. Written as `bool`.
    Bool,

    /// The primitive character type; holds a Unicode scalar value
    /// (a non-surrogate code point). Written as `char`.
    Char,

    /// A primitive signed integer type. For example, `i32`.
    Int(IntTy),

    /// A primitive unsigned integer type. For example, `u32`.
    Uint(UintTy),

    /// A primitive floating-point type. For example, `f64`.
    Float(FloatTy),

    /// Algebraic data types (ADT). For example: structures, enumerations and unions.
    ///
    /// For example, the type `List<i32>` would be represented using the `AdtDef`
    /// for `struct List<T>` and the args `[i32]`.
    ///
    /// Note that generic parameters in fields only get lazily instantiated
    /// by using something like `adt_def.all_fields().map(|field| field.ty(interner, args))`.
    Adt(I::AdtDef, I::GenericArgs),

    /// An unsized FFI type that is opaque to Rust. Written as `extern type T`.
    Foreign(I::DefId),

    /// The pointee of a string slice. Written as `str`.
    Str,

    /// An array with the given length. Written as `[T; N]`.
    Array(I::Ty, I::Const),

    /// A pattern newtype. Takes any type and restricts its valid values to its pattern.
    /// This will also change the layout to take advantage of this restriction.
    /// Only `Copy` and `Clone` will automatically get implemented for pattern types.
    /// Auto-traits treat this as if it were an aggregate with a single nested type.
    /// Only supports integer range patterns for now.
    Pat(I::Ty, I::Pat),

    /// The pointee of an array slice. Written as `[T]`.
    Slice(I::Ty),

    /// A raw pointer. Written as `*mut T` or `*const T`
    RawPtr(I::Ty, Mutability),

    /// A reference; a pointer with an associated lifetime. Written as
    /// `&'a mut T` or `&'a T`.
    Ref(I::Region, I::Ty, Mutability),

    /// The anonymous type of a function declaration/definition. Each
    /// function has a unique type.
    ///
    /// For the function `fn foo() -> i32 { 3 }` this type would be
    /// shown to the user as `fn() -> i32 {foo}`.
    ///
    /// For example the type of `bar` here:
    /// ```rust
    /// fn foo() -> i32 { 1 }
    /// let bar = foo; // bar: fn() -> i32 {foo}
    /// ```
    FnDef(I::DefId, I::GenericArgs),

    /// A pointer to a function. Written as `fn() -> i32`.
    ///
    /// Note that both functions and closures start out as either
    /// [FnDef] or [Closure] which can be then be coerced to this variant.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```rust
    /// fn foo() -> i32 { 1 }
    /// let bar: fn() -> i32 = foo;
    /// ```
    FnPtr(ty::Binder<I, FnSig<I>>),

    /// A trait object. Written as `dyn for<'b> Trait<'b, Assoc = u32> + Send + 'a`.
    Dynamic(I::BoundExistentialPredicates, I::Region, DynKind),

    /// The anonymous type of a closure. Used to represent the type of `|a| a`.
    ///
    /// Closure args contain both the - potentially instantiated - generic parameters
    /// of its parent and some synthetic parameters. See the documentation for
    /// `ClosureArgs` for more details.
    Closure(I::DefId, I::GenericArgs),

    /// The anonymous type of a closure. Used to represent the type of `async |a| a`.
    ///
    /// Coroutine-closure args contain both the - potentially instantiated - generic
    /// parameters of its parent and some synthetic parameters. See the documentation
    /// for `CoroutineClosureArgs` for more details.
    CoroutineClosure(I::DefId, I::GenericArgs),

    /// The anonymous type of a coroutine. Used to represent the type of
    /// `|a| yield a`.
    ///
    /// For more info about coroutine args, visit the documentation for
    /// `CoroutineArgs`.
    Coroutine(I::DefId, I::GenericArgs),

    /// A type representing the types stored inside a coroutine.
    /// This should only appear as part of the `CoroutineArgs`.
    ///
    /// Unlike upvars, the witness can reference lifetimes from
    /// inside of the coroutine itself. To deal with them in
    /// the type of the coroutine, we convert them to higher ranked
    /// lifetimes bound by the witness itself.
    ///
    /// This contains the `DefId` and the `GenericArgsRef` of the coroutine.
    /// The actual witness types are computed on MIR by the `mir_coroutine_witnesses` query.
    ///
    /// Looking at the following example, the witness for this coroutine
    /// may end up as something like `for<'a> [Vec<i32>, &'a Vec<i32>]`:
    ///
    /// ```
    /// #![feature(coroutines)]
    /// #[coroutine] static |a| {
    ///     let x = &vec![3];
    ///     yield a;
    ///     yield x[0];
    /// }
    /// # ;
    /// ```
    CoroutineWitness(I::DefId, I::GenericArgs),

    /// The never type `!`.
    Never,

    /// A tuple type. For example, `(i32, bool)`.
    Tuple(I::Tys),

    /// A projection, opaque type, weak type alias, or inherent associated type.
    /// All of these types are represented as pairs of def-id and args, and can
    /// be normalized, so they are grouped conceptually.
    Alias(AliasTyKind, AliasTy<I>),

    /// A type parameter; for example, `T` in `fn f<T>(x: T) {}`.
    Param(I::ParamTy),

    /// Bound type variable, used to represent the `'a` in `for<'a> fn(&'a ())`.
    ///
    /// For canonical queries, we replace inference variables with bound variables,
    /// so e.g. when checking whether `&'_ (): Trait<_>` holds, we canonicalize that to
    /// `for<'a, T> &'a (): Trait<T>` and then convert the introduced bound variables
    /// back to inference variables in a new inference context when inside of the query.
    ///
    /// It is conventional to render anonymous bound types like `^N` or `^D_N`,
    /// where `N` is the bound variable's anonymous index into the binder, and
    /// `D` is the debruijn index, or totally omitted if the debruijn index is zero.
    ///
    /// See the `rustc-dev-guide` for more details about
    /// [higher-ranked trait bounds][1] and [canonical queries][2].
    ///
    /// [1]: https://rustc-dev-guide.rust-lang.org/traits/hrtb.html
    /// [2]: https://rustc-dev-guide.rust-lang.org/traits/canonical-queries.html
    Bound(DebruijnIndex, I::BoundTy),

    /// A placeholder type, used during higher ranked subtyping to instantiate
    /// bound variables.
    ///
    /// It is conventional to render anonymous placeholder types like `!N` or `!U_N`,
    /// where `N` is the placeholder variable's anonymous index (which corresponds
    /// to the bound variable's index from the binder from which it was instantiated),
    /// and `U` is the universe index in which it is instantiated, or totally omitted
    /// if the universe index is zero.
    Placeholder(I::PlaceholderTy),

    /// A type variable used during type checking.
    ///
    /// Similar to placeholders, inference variables also live in a universe to
    /// correctly deal with higher ranked types. Though unlike placeholders,
    /// that universe is stored in the `InferCtxt` instead of directly
    /// inside of the type.
    Infer(InferTy),

    /// A placeholder for a type which could not be computed; this is
    /// propagated to avoid useless error messages.
    Error(I::ErrorGuaranteed),
}

impl<I: Interner> TyKind<I> {
    #[inline]
    pub fn is_primitive(&self) -> bool {
        matches!(self, Bool | Char | Int(_) | Uint(_) | Float(_))
    }
}

// This is manually implemented for `TyKind` because `std::mem::discriminant`
// returns an opaque value that is `PartialEq` but not `PartialOrd`
#[inline]
const fn tykind_discriminant<I: Interner>(value: &TyKind<I>) -> usize {
    match value {
        Bool => 0,
        Char => 1,
        Int(_) => 2,
        Uint(_) => 3,
        Float(_) => 4,
        Adt(_, _) => 5,
        Foreign(_) => 6,
        Str => 7,
        Array(_, _) => 8,
        Slice(_) => 9,
        RawPtr(_, _) => 10,
        Ref(_, _, _) => 11,
        FnDef(_, _) => 12,
        FnPtr(_) => 13,
        Dynamic(..) => 14,
        Closure(_, _) => 15,
        CoroutineClosure(_, _) => 16,
        Coroutine(_, _) => 17,
        CoroutineWitness(_, _) => 18,
        Never => 19,
        Tuple(_) => 20,
        Pat(_, _) => 21,
        Alias(_, _) => 22,
        Param(_) => 23,
        Bound(_, _) => 24,
        Placeholder(_) => 25,
        Infer(_) => 26,
        Error(_) => 27,
    }
}

// FIXME(GrigorenkoPV): consider not implementing PartialEq manually
// This is manually implemented because a derive would require `I: PartialEq`
impl<I: Interner> PartialEq for TyKind<I> {
    #[inline]
    fn eq(&self, other: &TyKind<I>) -> bool {
        // You might expect this `match` to be preceded with this:
        //
        //   tykind_discriminant(self) == tykind_discriminant(other) &&
        //
        // but the data patterns in practice are such that a comparison
        // succeeds 99%+ of the time, and it's faster to omit it.
        match (self, other) {
            (Int(a_i), Int(b_i)) => a_i == b_i,
            (Uint(a_u), Uint(b_u)) => a_u == b_u,
            (Float(a_f), Float(b_f)) => a_f == b_f,
            (Adt(a_d, a_s), Adt(b_d, b_s)) => a_d == b_d && a_s == b_s,
            (Foreign(a_d), Foreign(b_d)) => a_d == b_d,
            (Array(a_t, a_c), Array(b_t, b_c)) => a_t == b_t && a_c == b_c,
            (Pat(a_t, a_c), Pat(b_t, b_c)) => a_t == b_t && a_c == b_c,
            (Slice(a_t), Slice(b_t)) => a_t == b_t,
            (RawPtr(a_t, a_m), RawPtr(b_t, b_m)) => a_t == b_t && a_m == b_m,
            (Ref(a_r, a_t, a_m), Ref(b_r, b_t, b_m)) => a_r == b_r && a_t == b_t && a_m == b_m,
            (FnDef(a_d, a_s), FnDef(b_d, b_s)) => a_d == b_d && a_s == b_s,
            (FnPtr(a_s), FnPtr(b_s)) => a_s == b_s,
            (Dynamic(a_p, a_r, a_repr), Dynamic(b_p, b_r, b_repr)) => {
                a_p == b_p && a_r == b_r && a_repr == b_repr
            }
            (Closure(a_d, a_s), Closure(b_d, b_s)) => a_d == b_d && a_s == b_s,
            (CoroutineClosure(a_d, a_s), CoroutineClosure(b_d, b_s)) => a_d == b_d && a_s == b_s,
            (Coroutine(a_d, a_s), Coroutine(b_d, b_s)) => a_d == b_d && a_s == b_s,
            (CoroutineWitness(a_d, a_s), CoroutineWitness(b_d, b_s)) => a_d == b_d && a_s == b_s,
            (Tuple(a_t), Tuple(b_t)) => a_t == b_t,
            (Alias(a_i, a_p), Alias(b_i, b_p)) => a_i == b_i && a_p == b_p,
            (Param(a_p), Param(b_p)) => a_p == b_p,
            (Bound(a_d, a_b), Bound(b_d, b_b)) => a_d == b_d && a_b == b_b,
            (Placeholder(a_p), Placeholder(b_p)) => a_p == b_p,
            (Infer(a_t), Infer(b_t)) => a_t == b_t,
            (Error(a_e), Error(b_e)) => a_e == b_e,
            (Bool, Bool) | (Char, Char) | (Str, Str) | (Never, Never) => true,
            _ => {
                debug_assert!(
                    tykind_discriminant(self) != tykind_discriminant(other),
                    "This branch must be unreachable, maybe the match is missing an arm? self = {self:?}, other = {other:?}"
                );
                false
            }
        }
    }
}

// This is manually implemented because a derive would require `I: Debug`
impl<I: Interner> fmt::Debug for TyKind<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Bool => write!(f, "bool"),
            Char => write!(f, "char"),
            Int(i) => write!(f, "{i:?}"),
            Uint(u) => write!(f, "{u:?}"),
            Float(float) => write!(f, "{float:?}"),
            Adt(d, s) => {
                write!(f, "{d:?}")?;
                let mut s = s.iter();
                let first = s.next();
                match first {
                    Some(first) => write!(f, "<{:?}", first)?,
                    None => return Ok(()),
                };

                for arg in s {
                    write!(f, ", {:?}", arg)?;
                }

                write!(f, ">")
            }
            Foreign(d) => f.debug_tuple("Foreign").field(d).finish(),
            Str => write!(f, "str"),
            Array(t, c) => write!(f, "[{t:?}; {c:?}]"),
            Pat(t, p) => write!(f, "pattern_type!({t:?} is {p:?})"),
            Slice(t) => write!(f, "[{:?}]", &t),
            RawPtr(ty, mutbl) => write!(f, "*{} {:?}", mutbl.ptr_str(), ty),
            Ref(r, t, m) => write!(f, "&{:?} {}{:?}", r, m.prefix_str(), t),
            FnDef(d, s) => f.debug_tuple("FnDef").field(d).field(&s).finish(),
            FnPtr(s) => write!(f, "{s:?}"),
            Dynamic(p, r, repr) => match repr {
                DynKind::Dyn => write!(f, "dyn {p:?} + {r:?}"),
                DynKind::DynStar => write!(f, "dyn* {p:?} + {r:?}"),
            },
            Closure(d, s) => f.debug_tuple("Closure").field(d).field(&s).finish(),
            CoroutineClosure(d, s) => f.debug_tuple("CoroutineClosure").field(d).field(&s).finish(),
            Coroutine(d, s) => f.debug_tuple("Coroutine").field(d).field(&s).finish(),
            CoroutineWitness(d, s) => f.debug_tuple("CoroutineWitness").field(d).field(&s).finish(),
            Never => write!(f, "!"),
            Tuple(t) => {
                write!(f, "(")?;
                let mut count = 0;
                for ty in t.iter() {
                    if count > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{ty:?}")?;
                    count += 1;
                }
                // unary tuples need a trailing comma
                if count == 1 {
                    write!(f, ",")?;
                }
                write!(f, ")")
            }
            Alias(i, a) => f.debug_tuple("Alias").field(i).field(&a).finish(),
            Param(p) => write!(f, "{p:?}"),
            Bound(d, b) => crate::debug_bound_var(f, *d, b),
            Placeholder(p) => write!(f, "{p:?}"),
            Infer(t) => write!(f, "{:?}", t),
            TyKind::Error(_) => write!(f, "{{type error}}"),
        }
    }
}

/// Represents the projection of an associated, opaque, or lazy-type-alias type.
///
/// * For a projection, this would be `<Ty as Trait<...>>::N<...>`.
/// * For an inherent projection, this would be `Ty::N<...>`.
/// * For an opaque type, there is no explicit syntax.
#[derive_where(Clone, Copy, Hash, PartialEq, Eq, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
#[cfg_attr(feature = "nightly", derive(TyDecodable, TyEncodable, HashStable_NoContext))]
pub struct AliasTy<I: Interner> {
    /// The parameters of the associated or opaque type.
    ///
    /// For a projection, these are the generic parameters for the trait and the
    /// GAT parameters, if there are any.
    ///
    /// For an inherent projection, they consist of the self type and the GAT parameters,
    /// if there are any.
    ///
    /// For RPIT the generic parameters are for the generics of the function,
    /// while for TAIT it is used for the generic parameters of the alias.
    pub args: I::GenericArgs,

    /// The `DefId` of the `TraitItem` or `ImplItem` for the associated type `N` depending on whether
    /// this is a projection or an inherent projection or the `DefId` of the `OpaqueType` item if
    /// this is an opaque.
    ///
    /// During codegen, `interner.type_of(def_id)` can be used to get the type of the
    /// underlying type if the type is an opaque.
    ///
    /// Note that if this is an associated type, this is not the `DefId` of the
    /// `TraitRef` containing this associated type, which is in `interner.associated_item(def_id).container`,
    /// aka. `interner.parent(def_id)`.
    pub def_id: I::DefId,

    /// This field exists to prevent the creation of `AliasTy` without using [`AliasTy::new_from_args`].
    #[derive_where(skip(Debug))]
    pub(crate) _use_alias_ty_new_instead: (),
}

impl<I: Interner> AliasTy<I> {
    pub fn new_from_args(interner: I, def_id: I::DefId, args: I::GenericArgs) -> AliasTy<I> {
        interner.debug_assert_args_compatible(def_id, args);
        AliasTy { def_id, args, _use_alias_ty_new_instead: () }
    }

    pub fn new(
        interner: I,
        def_id: I::DefId,
        args: impl IntoIterator<Item: Into<I::GenericArg>>,
    ) -> AliasTy<I> {
        let args = interner.mk_args_from_iter(args.into_iter().map(Into::into));
        Self::new_from_args(interner, def_id, args)
    }

    pub fn kind(self, interner: I) -> AliasTyKind {
        interner.alias_ty_kind(self)
    }

    /// Whether this alias type is an opaque.
    pub fn is_opaque(self, interner: I) -> bool {
        matches!(self.kind(interner), AliasTyKind::Opaque)
    }

    pub fn to_ty(self, interner: I) -> I::Ty {
        Ty::new_alias(interner, self.kind(interner), self)
    }
}

/// The following methods work only with (trait) associated type projections.
impl<I: Interner> AliasTy<I> {
    pub fn self_ty(self) -> I::Ty {
        self.args.type_at(0)
    }

    pub fn with_self_ty(self, interner: I, self_ty: I::Ty) -> Self {
        AliasTy::new(
            interner,
            self.def_id,
            [self_ty.into()].into_iter().chain(self.args.iter().skip(1)),
        )
    }

    pub fn trait_def_id(self, interner: I) -> I::DefId {
        assert_eq!(self.kind(interner), AliasTyKind::Projection, "expected a projection");
        interner.parent(self.def_id)
    }

    /// Extracts the underlying trait reference and own args from this projection.
    /// For example, if this is a projection of `<T as StreamingIterator>::Item<'a>`,
    /// then this function would return a `T: StreamingIterator` trait reference and
    /// `['a]` as the own args.
    pub fn trait_ref_and_own_args(self, interner: I) -> (ty::TraitRef<I>, I::GenericArgsSlice) {
        debug_assert_eq!(self.kind(interner), AliasTyKind::Projection);
        interner.trait_ref_and_own_args_for_alias(self.def_id, self.args)
    }

    /// Extracts the underlying trait reference from this projection.
    /// For example, if this is a projection of `<T as Iterator>::Item`,
    /// then this function would return a `T: Iterator` trait reference.
    ///
    /// WARNING: This will drop the args for generic associated types
    /// consider calling [Self::trait_ref_and_own_args] to get those
    /// as well.
    pub fn trait_ref(self, interner: I) -> ty::TraitRef<I> {
        self.trait_ref_and_own_args(interner).0
    }
}

/// The following methods work only with inherent associated type projections.
impl<I: Interner> AliasTy<I> {
    /// Transform the generic parameters to have the given `impl` args as the base and the GAT args on top of that.
    ///
    /// Does the following transformation:
    ///
    /// ```text
    /// [Self, P_0...P_m] -> [I_0...I_n, P_0...P_m]
    ///
    ///     I_i impl args
    ///     P_j GAT args
    /// ```
    pub fn rebase_inherent_args_onto_impl(
        self,
        impl_args: I::GenericArgs,
        interner: I,
    ) -> I::GenericArgs {
        debug_assert_eq!(self.kind(interner), AliasTyKind::Inherent);
        interner.mk_args_from_iter(impl_args.iter().chain(self.args.iter().skip(1)))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable, HashStable_NoContext))]
pub enum IntTy {
    Isize,
    I8,
    I16,
    I32,
    I64,
    I128,
}

impl IntTy {
    pub fn name_str(&self) -> &'static str {
        match *self {
            IntTy::Isize => "isize",
            IntTy::I8 => "i8",
            IntTy::I16 => "i16",
            IntTy::I32 => "i32",
            IntTy::I64 => "i64",
            IntTy::I128 => "i128",
        }
    }

    pub fn bit_width(&self) -> Option<u64> {
        Some(match *self {
            IntTy::Isize => return None,
            IntTy::I8 => 8,
            IntTy::I16 => 16,
            IntTy::I32 => 32,
            IntTy::I64 => 64,
            IntTy::I128 => 128,
        })
    }

    pub fn normalize(&self, target_width: u32) -> Self {
        match self {
            IntTy::Isize => match target_width {
                16 => IntTy::I16,
                32 => IntTy::I32,
                64 => IntTy::I64,
                _ => unreachable!(),
            },
            _ => *self,
        }
    }

    pub fn to_unsigned(self) -> UintTy {
        match self {
            IntTy::Isize => UintTy::Usize,
            IntTy::I8 => UintTy::U8,
            IntTy::I16 => UintTy::U16,
            IntTy::I32 => UintTy::U32,
            IntTy::I64 => UintTy::U64,
            IntTy::I128 => UintTy::U128,
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Copy)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable, HashStable_NoContext))]
pub enum UintTy {
    Usize,
    U8,
    U16,
    U32,
    U64,
    U128,
}

impl UintTy {
    pub fn name_str(&self) -> &'static str {
        match *self {
            UintTy::Usize => "usize",
            UintTy::U8 => "u8",
            UintTy::U16 => "u16",
            UintTy::U32 => "u32",
            UintTy::U64 => "u64",
            UintTy::U128 => "u128",
        }
    }

    pub fn bit_width(&self) -> Option<u64> {
        Some(match *self {
            UintTy::Usize => return None,
            UintTy::U8 => 8,
            UintTy::U16 => 16,
            UintTy::U32 => 32,
            UintTy::U64 => 64,
            UintTy::U128 => 128,
        })
    }

    pub fn normalize(&self, target_width: u32) -> Self {
        match self {
            UintTy::Usize => match target_width {
                16 => UintTy::U16,
                32 => UintTy::U32,
                64 => UintTy::U64,
                _ => unreachable!(),
            },
            _ => *self,
        }
    }

    pub fn to_signed(self) -> IntTy {
        match self {
            UintTy::Usize => IntTy::Isize,
            UintTy::U8 => IntTy::I8,
            UintTy::U16 => IntTy::I16,
            UintTy::U32 => IntTy::I32,
            UintTy::U64 => IntTy::I64,
            UintTy::U128 => IntTy::I128,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable, HashStable_NoContext))]
pub enum FloatTy {
    F16,
    F32,
    F64,
    F128,
}

impl FloatTy {
    pub fn name_str(self) -> &'static str {
        match self {
            FloatTy::F16 => "f16",
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
            FloatTy::F128 => "f128",
        }
    }

    pub fn bit_width(self) -> u64 {
        match self {
            FloatTy::F16 => 16,
            FloatTy::F32 => 32,
            FloatTy::F64 => 64,
            FloatTy::F128 => 128,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntVarValue {
    Unknown,
    IntType(IntTy),
    UintType(UintTy),
}

impl IntVarValue {
    pub fn is_known(self) -> bool {
        match self {
            IntVarValue::IntType(_) | IntVarValue::UintType(_) => true,
            IntVarValue::Unknown => false,
        }
    }

    pub fn is_unknown(self) -> bool {
        !self.is_known()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FloatVarValue {
    Unknown,
    Known(FloatTy),
}

impl FloatVarValue {
    pub fn is_known(self) -> bool {
        match self {
            FloatVarValue::Known(_) => true,
            FloatVarValue::Unknown => false,
        }
    }

    pub fn is_unknown(self) -> bool {
        !self.is_known()
    }
}

rustc_index::newtype_index! {
    /// A **ty**pe **v**ariable **ID**.
    #[encodable]
    #[orderable]
    #[debug_format = "?{}t"]
    #[gate_rustc_only]
    pub struct TyVid {}
}

rustc_index::newtype_index! {
    /// An **int**egral (`u32`, `i32`, `usize`, etc.) type **v**ariable **ID**.
    #[encodable]
    #[orderable]
    #[debug_format = "?{}i"]
    #[gate_rustc_only]
    pub struct IntVid {}
}

rustc_index::newtype_index! {
    /// A **float**ing-point (`f32` or `f64`) type **v**ariable **ID**.
    #[encodable]
    #[orderable]
    #[debug_format = "?{}f"]
    #[gate_rustc_only]
    pub struct FloatVid {}
}

/// A placeholder for a type that hasn't been inferred yet.
///
/// E.g., if we have an empty array (`[]`), then we create a fresh
/// type variable for the element type since we won't know until it's
/// used what the element type is supposed to be.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "nightly", derive(Encodable, Decodable))]
pub enum InferTy {
    /// A type variable.
    TyVar(TyVid),
    /// An integral type variable (`{integer}`).
    ///
    /// These are created when the compiler sees an integer literal like
    /// `1` that could be several different types (`u8`, `i32`, `u32`, etc.).
    /// We don't know until it's used what type it's supposed to be, so
    /// we create a fresh type variable.
    IntVar(IntVid),
    /// A floating-point type variable (`{float}`).
    ///
    /// These are created when the compiler sees an float literal like
    /// `1.0` that could be either an `f32` or an `f64`.
    /// We don't know until it's used what type it's supposed to be, so
    /// we create a fresh type variable.
    FloatVar(FloatVid),

    /// A [`FreshTy`][Self::FreshTy] is one that is generated as a replacement
    /// for an unbound type variable. This is convenient for caching etc. See
    /// `rustc_infer::infer::freshen` for more details.
    ///
    /// Compare with [`TyVar`][Self::TyVar].
    FreshTy(u32),
    /// Like [`FreshTy`][Self::FreshTy], but as a replacement for [`IntVar`][Self::IntVar].
    FreshIntTy(u32),
    /// Like [`FreshTy`][Self::FreshTy], but as a replacement for [`FloatVar`][Self::FloatVar].
    FreshFloatTy(u32),
}

/// Raw `TyVid` are used as the unification key for `sub_relations`;
/// they carry no values.
#[cfg(feature = "nightly")]
impl UnifyKey for TyVid {
    type Value = ();
    #[inline]
    fn index(&self) -> u32 {
        self.as_u32()
    }
    #[inline]
    fn from_index(i: u32) -> TyVid {
        TyVid::from_u32(i)
    }
    fn tag() -> &'static str {
        "TyVid"
    }
}

#[cfg(feature = "nightly")]
impl UnifyValue for IntVarValue {
    type Error = NoError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, Self::Error> {
        match (*value1, *value2) {
            (IntVarValue::Unknown, IntVarValue::Unknown) => Ok(IntVarValue::Unknown),
            (
                IntVarValue::Unknown,
                known @ (IntVarValue::UintType(_) | IntVarValue::IntType(_)),
            )
            | (
                known @ (IntVarValue::UintType(_) | IntVarValue::IntType(_)),
                IntVarValue::Unknown,
            ) => Ok(known),
            _ => panic!("differing ints should have been resolved first"),
        }
    }
}

#[cfg(feature = "nightly")]
impl UnifyKey for IntVid {
    type Value = IntVarValue;
    #[inline] // make this function eligible for inlining - it is quite hot.
    fn index(&self) -> u32 {
        self.as_u32()
    }
    #[inline]
    fn from_index(i: u32) -> IntVid {
        IntVid::from_u32(i)
    }
    fn tag() -> &'static str {
        "IntVid"
    }
}

#[cfg(feature = "nightly")]
impl UnifyValue for FloatVarValue {
    type Error = NoError;

    fn unify_values(value1: &Self, value2: &Self) -> Result<Self, Self::Error> {
        match (*value1, *value2) {
            (FloatVarValue::Unknown, FloatVarValue::Unknown) => Ok(FloatVarValue::Unknown),
            (FloatVarValue::Unknown, FloatVarValue::Known(known))
            | (FloatVarValue::Known(known), FloatVarValue::Unknown) => {
                Ok(FloatVarValue::Known(known))
            }
            (FloatVarValue::Known(_), FloatVarValue::Known(_)) => {
                panic!("differing floats should have been resolved first")
            }
        }
    }
}

#[cfg(feature = "nightly")]
impl UnifyKey for FloatVid {
    type Value = FloatVarValue;
    #[inline]
    fn index(&self) -> u32 {
        self.as_u32()
    }
    #[inline]
    fn from_index(i: u32) -> FloatVid {
        FloatVid::from_u32(i)
    }
    fn tag() -> &'static str {
        "FloatVid"
    }
}

#[cfg(feature = "nightly")]
impl<CTX> HashStable<CTX> for InferTy {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        use InferTy::*;
        std::mem::discriminant(self).hash_stable(ctx, hasher);
        match self {
            TyVar(_) | IntVar(_) | FloatVar(_) => {
                panic!("type variables should not be hashed: {self:?}")
            }
            FreshTy(v) | FreshIntTy(v) | FreshFloatTy(v) => v.hash_stable(ctx, hasher),
        }
    }
}

impl fmt::Display for InferTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InferTy::*;
        match *self {
            TyVar(_) => write!(f, "_"),
            IntVar(_) => write!(f, "{}", "{integer}"),
            FloatVar(_) => write!(f, "{}", "{float}"),
            FreshTy(v) => write!(f, "FreshTy({v})"),
            FreshIntTy(v) => write!(f, "FreshIntTy({v})"),
            FreshFloatTy(v) => write!(f, "FreshFloatTy({v})"),
        }
    }
}

impl fmt::Debug for IntTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_str())
    }
}

impl fmt::Debug for UintTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_str())
    }
}

impl fmt::Debug for FloatTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name_str())
    }
}

impl fmt::Debug for InferTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InferTy::*;
        match *self {
            TyVar(ref v) => v.fmt(f),
            IntVar(ref v) => v.fmt(f),
            FloatVar(ref v) => v.fmt(f),
            FreshTy(v) => write!(f, "FreshTy({v:?})"),
            FreshIntTy(v) => write!(f, "FreshIntTy({v:?})"),
            FreshFloatTy(v) => write!(f, "FreshFloatTy({v:?})"),
        }
    }
}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash, Debug; I: Interner)]
#[cfg_attr(feature = "nightly", derive(TyEncodable, TyDecodable, HashStable_NoContext))]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
pub struct TypeAndMut<I: Interner> {
    pub ty: I::Ty,
    pub mutbl: Mutability,
}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash; I: Interner)]
#[cfg_attr(feature = "nightly", derive(TyEncodable, TyDecodable, HashStable_NoContext))]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
pub struct FnSig<I: Interner> {
    pub inputs_and_output: I::Tys,
    pub c_variadic: bool,
    pub safety: I::Safety,
    pub abi: I::Abi,
}

impl<I: Interner> FnSig<I> {
    pub fn split_inputs_and_output(self) -> (I::FnInputTys, I::Ty) {
        self.inputs_and_output.split_inputs_and_output()
    }

    pub fn inputs(self) -> I::FnInputTys {
        self.split_inputs_and_output().0
    }

    pub fn output(self) -> I::Ty {
        self.split_inputs_and_output().1
    }

    pub fn is_fn_trait_compatible(self) -> bool {
        let FnSig { safety, abi, c_variadic, .. } = self;
        !c_variadic && safety.is_safe() && abi.is_rust()
    }
}

impl<I: Interner> ty::Binder<I, FnSig<I>> {
    #[inline]
    pub fn inputs(self) -> ty::Binder<I, I::FnInputTys> {
        self.map_bound(|fn_sig| fn_sig.inputs())
    }

    #[inline]
    #[track_caller]
    pub fn input(self, index: usize) -> ty::Binder<I, I::Ty> {
        self.map_bound(|fn_sig| fn_sig.inputs().get(index).unwrap())
    }

    pub fn inputs_and_output(self) -> ty::Binder<I, I::Tys> {
        self.map_bound(|fn_sig| fn_sig.inputs_and_output)
    }

    #[inline]
    pub fn output(self) -> ty::Binder<I, I::Ty> {
        self.map_bound(|fn_sig| fn_sig.output())
    }

    pub fn c_variadic(self) -> bool {
        self.skip_binder().c_variadic
    }

    pub fn safety(self) -> I::Safety {
        self.skip_binder().safety
    }

    pub fn abi(self) -> I::Abi {
        self.skip_binder().abi
    }

    pub fn is_fn_trait_compatible(&self) -> bool {
        self.skip_binder().is_fn_trait_compatible()
    }
}

impl<I: Interner> fmt::Debug for FnSig<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sig = self;
        let FnSig { inputs_and_output: _, c_variadic, safety, abi } = sig;

        write!(f, "{}", safety.prefix_str())?;
        if !abi.is_rust() {
            write!(f, "extern \"{abi:?}\" ")?;
        }

        write!(f, "fn(")?;
        let (inputs, output) = sig.split_inputs_and_output();
        for (i, ty) in inputs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{ty:?}")?;
        }
        if *c_variadic {
            if inputs.is_empty() {
                write!(f, "...")?;
            } else {
                write!(f, ", ...")?;
            }
        }
        write!(f, ")")?;

        match output.kind() {
            Tuple(list) if list.is_empty() => Ok(()),
            _ => write!(f, " -> {:?}", sig.output()),
        }
    }
}
