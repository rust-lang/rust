#![allow(rustc::usage_of_ty_tykind)]

use rustc_data_structures::stable_hasher::{HashStable, StableHasher};
use rustc_data_structures::unify::{EqUnifyValue, UnifyKey};
use std::fmt;
use std::mem::discriminant;

use crate::HashStableContext;
use crate::Interner;
use crate::{DebruijnIndex, DebugWithInfcx, InferCtxtLike, WithInfcx};

use self::TyKind::*;

/// The movability of a coroutine / closure literal:
/// whether a coroutine contains self-references, causing it to be `!Unpin`.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable, Debug, Copy)]
#[derive(HashStable_Generic)]
pub enum Movability {
    /// May contain self-references, `!Unpin`.
    Static,
    /// Must not contain self-references, `Unpin`.
    Movable,
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Copy)]
#[derive(HashStable_Generic, Encodable, Decodable)]
pub enum Mutability {
    // N.B. Order is deliberate, so that Not < Mut
    Not,
    Mut,
}

impl Mutability {
    pub fn invert(self) -> Self {
        match self {
            Mutability::Mut => Mutability::Not,
            Mutability::Not => Mutability::Mut,
        }
    }

    /// Returns `""` (empty string) or `"mut "` depending on the mutability.
    pub fn prefix_str(self) -> &'static str {
        match self {
            Mutability::Mut => "mut ",
            Mutability::Not => "",
        }
    }

    /// Returns `"&"` or `"&mut "` depending on the mutability.
    pub fn ref_prefix_str(self) -> &'static str {
        match self {
            Mutability::Not => "&",
            Mutability::Mut => "&mut ",
        }
    }

    /// Returns `""` (empty string) or `"mutably "` depending on the mutability.
    pub fn mutably_str(self) -> &'static str {
        match self {
            Mutability::Not => "",
            Mutability::Mut => "mutably ",
        }
    }

    /// Return `true` if self is mutable
    pub fn is_mut(self) -> bool {
        matches!(self, Self::Mut)
    }

    /// Return `true` if self is **not** mutable
    pub fn is_not(self) -> bool {
        matches!(self, Self::Not)
    }
}

/// Specifies how a trait object is represented.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[derive(Encodable, Decodable, HashStable_Generic)]
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
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum AliasKind {
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

/// Defines the kinds of types used by the type system.
///
/// Types written by the user start out as `hir::TyKind` and get
/// converted to this representation using `AstConv::ast_ty_to_ty`.
#[rustc_diagnostic_item = "IrTyKind"]
#[derive(derivative::Derivative)]
#[derivative(
    Clone(bound = ""),
    PartialOrd(bound = ""),
    PartialOrd = "feature_allow_slow_enum",
    Ord(bound = ""),
    Ord = "feature_allow_slow_enum",
    Hash(bound = "")
)]
#[derive(TyEncodable, TyDecodable)]
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
    /// Note that generic parameters in fields only get lazily substituted
    /// by using something like `adt_def.all_fields().map(|field| field.ty(tcx, args))`.
    Adt(I::AdtDef, I::GenericArgs),

    /// An unsized FFI type that is opaque to Rust. Written as `extern type T`.
    Foreign(I::DefId),

    /// The pointee of a string slice. Written as `str`.
    Str,

    /// An array with the given length. Written as `[T; N]`.
    Array(I::Ty, I::Const),

    /// The pointee of an array slice. Written as `[T]`.
    Slice(I::Ty),

    /// A raw pointer. Written as `*mut T` or `*const T`
    RawPtr(I::TypeAndMut),

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
    FnPtr(I::PolyFnSig),

    /// A trait object. Written as `dyn for<'b> Trait<'b, Assoc = u32> + Send + 'a`.
    Dynamic(I::BoundExistentialPredicates, I::Region, DynKind),

    /// The anonymous type of a closure. Used to represent the type of `|a| a`.
    ///
    /// Closure args contain both the - potentially substituted - generic parameters
    /// of its parent and some synthetic parameters. See the documentation for
    /// `ClosureArgs` for more details.
    Closure(I::DefId, I::GenericArgs),

    /// The anonymous type of a coroutine. Used to represent the type of
    /// `|a| yield a`.
    ///
    /// For more info about coroutine args, visit the documentation for
    /// `CoroutineArgs`.
    Coroutine(I::DefId, I::GenericArgs, Movability),

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
    /// ```ignore UNSOLVED (ask @compiler-errors, should this error? can we just swap the yields?)
    /// #![feature(coroutines)]
    /// |a| {
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
    Alias(AliasKind, I::AliasTy),

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
    /// It is conventional to render anonymous placeholer types like `!N` or `!U_N`,
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
    Infer(I::InferTy),

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
        RawPtr(_) => 10,
        Ref(_, _, _) => 11,
        FnDef(_, _) => 12,
        FnPtr(_) => 13,
        Dynamic(..) => 14,
        Closure(_, _) => 15,
        Coroutine(_, _, _) => 16,
        CoroutineWitness(_, _) => 17,
        Never => 18,
        Tuple(_) => 19,
        Alias(_, _) => 20,
        Param(_) => 21,
        Bound(_, _) => 22,
        Placeholder(_) => 23,
        Infer(_) => 24,
        Error(_) => 25,
    }
}

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
            (Slice(a_t), Slice(b_t)) => a_t == b_t,
            (RawPtr(a_t), RawPtr(b_t)) => a_t == b_t,
            (Ref(a_r, a_t, a_m), Ref(b_r, b_t, b_m)) => a_r == b_r && a_t == b_t && a_m == b_m,
            (FnDef(a_d, a_s), FnDef(b_d, b_s)) => a_d == b_d && a_s == b_s,
            (FnPtr(a_s), FnPtr(b_s)) => a_s == b_s,
            (Dynamic(a_p, a_r, a_repr), Dynamic(b_p, b_r, b_repr)) => {
                a_p == b_p && a_r == b_r && a_repr == b_repr
            }
            (Closure(a_d, a_s), Closure(b_d, b_s)) => a_d == b_d && a_s == b_s,
            (Coroutine(a_d, a_s, a_m), Coroutine(b_d, b_s, b_m)) => {
                a_d == b_d && a_s == b_s && a_m == b_m
            }
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
                    "This branch must be unreachable, maybe the match is missing an arm? self = self = {self:?}, other = {other:?}"
                );
                false
            }
        }
    }
}

// This is manually implemented because a derive would require `I: Eq`
impl<I: Interner> Eq for TyKind<I> {}

impl<I: Interner> DebugWithInfcx<I> for TyKind<I> {
    fn fmt<Infcx: InferCtxtLike<Interner = I>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> fmt::Result {
        match this.data {
            Bool => write!(f, "bool"),
            Char => write!(f, "char"),
            Int(i) => write!(f, "{i:?}"),
            Uint(u) => write!(f, "{u:?}"),
            Float(float) => write!(f, "{float:?}"),
            Adt(d, s) => {
                write!(f, "{d:?}")?;
                let mut s = s.clone().into_iter();
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
            Foreign(d) => f.debug_tuple_field1_finish("Foreign", d),
            Str => write!(f, "str"),
            Array(t, c) => write!(f, "[{:?}; {:?}]", &this.wrap(t), &this.wrap(c)),
            Slice(t) => write!(f, "[{:?}]", &this.wrap(t)),
            RawPtr(p) => {
                let (ty, mutbl) = I::ty_and_mut_to_parts(p.clone());
                match mutbl {
                    Mutability::Mut => write!(f, "*mut "),
                    Mutability::Not => write!(f, "*const "),
                }?;
                write!(f, "{:?}", &this.wrap(ty))
            }
            Ref(r, t, m) => match m {
                Mutability::Mut => write!(f, "&{:?} mut {:?}", &this.wrap(r), &this.wrap(t)),
                Mutability::Not => write!(f, "&{:?} {:?}", &this.wrap(r), &this.wrap(t)),
            },
            FnDef(d, s) => f.debug_tuple_field2_finish("FnDef", d, &this.wrap(s)),
            FnPtr(s) => write!(f, "{:?}", &this.wrap(s)),
            Dynamic(p, r, repr) => match repr {
                DynKind::Dyn => write!(f, "dyn {:?} + {:?}", &this.wrap(p), &this.wrap(r)),
                DynKind::DynStar => {
                    write!(f, "dyn* {:?} + {:?}", &this.wrap(p), &this.wrap(r))
                }
            },
            Closure(d, s) => f.debug_tuple_field2_finish("Closure", d, &this.wrap(s)),
            Coroutine(d, s, m) => f.debug_tuple_field3_finish("Coroutine", d, &this.wrap(s), m),
            CoroutineWitness(d, s) => {
                f.debug_tuple_field2_finish("CoroutineWitness", d, &this.wrap(s))
            }
            Never => write!(f, "!"),
            Tuple(t) => {
                write!(f, "(")?;
                let mut count = 0;
                for ty in t.clone() {
                    if count > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{:?}", &this.wrap(ty))?;
                    count += 1;
                }
                // unary tuples need a trailing comma
                if count == 1 {
                    write!(f, ",")?;
                }
                write!(f, ")")
            }
            Alias(i, a) => f.debug_tuple_field2_finish("Alias", i, &this.wrap(a)),
            Param(p) => write!(f, "{p:?}"),
            Bound(d, b) => crate::debug_bound_var(f, *d, b),
            Placeholder(p) => write!(f, "{p:?}"),
            Infer(t) => write!(f, "{:?}", this.wrap(t)),
            TyKind::Error(_) => write!(f, "{{type error}}"),
        }
    }
}

// This is manually implemented because a derive would require `I: Debug`
impl<I: Interner> fmt::Debug for TyKind<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        WithInfcx::with_no_infcx(self).fmt(f)
    }
}

// This is not a derived impl because a derive would require `I: HashStable`
#[allow(rustc::usage_of_ty_tykind)]
impl<CTX: HashStableContext, I: Interner> HashStable<CTX> for TyKind<I>
where
    I::AdtDef: HashStable<CTX>,
    I::DefId: HashStable<CTX>,
    I::GenericArgs: HashStable<CTX>,
    I::Ty: HashStable<CTX>,
    I::Const: HashStable<CTX>,
    I::TypeAndMut: HashStable<CTX>,
    I::PolyFnSig: HashStable<CTX>,
    I::BoundExistentialPredicates: HashStable<CTX>,
    I::Region: HashStable<CTX>,
    I::Tys: HashStable<CTX>,
    I::AliasTy: HashStable<CTX>,
    I::BoundTy: HashStable<CTX>,
    I::ParamTy: HashStable<CTX>,
    I::PlaceholderTy: HashStable<CTX>,
    I::InferTy: HashStable<CTX>,
    I::ErrorGuaranteed: HashStable<CTX>,
{
    #[inline]
    fn hash_stable(&self, __hcx: &mut CTX, __hasher: &mut StableHasher) {
        std::mem::discriminant(self).hash_stable(__hcx, __hasher);
        match self {
            Bool => {}
            Char => {}
            Int(i) => {
                i.hash_stable(__hcx, __hasher);
            }
            Uint(u) => {
                u.hash_stable(__hcx, __hasher);
            }
            Float(f) => {
                f.hash_stable(__hcx, __hasher);
            }
            Adt(adt, args) => {
                adt.hash_stable(__hcx, __hasher);
                args.hash_stable(__hcx, __hasher);
            }
            Foreign(def_id) => {
                def_id.hash_stable(__hcx, __hasher);
            }
            Str => {}
            Array(t, c) => {
                t.hash_stable(__hcx, __hasher);
                c.hash_stable(__hcx, __hasher);
            }
            Slice(t) => {
                t.hash_stable(__hcx, __hasher);
            }
            RawPtr(tam) => {
                tam.hash_stable(__hcx, __hasher);
            }
            Ref(r, t, m) => {
                r.hash_stable(__hcx, __hasher);
                t.hash_stable(__hcx, __hasher);
                m.hash_stable(__hcx, __hasher);
            }
            FnDef(def_id, args) => {
                def_id.hash_stable(__hcx, __hasher);
                args.hash_stable(__hcx, __hasher);
            }
            FnPtr(polyfnsig) => {
                polyfnsig.hash_stable(__hcx, __hasher);
            }
            Dynamic(l, r, repr) => {
                l.hash_stable(__hcx, __hasher);
                r.hash_stable(__hcx, __hasher);
                repr.hash_stable(__hcx, __hasher);
            }
            Closure(def_id, args) => {
                def_id.hash_stable(__hcx, __hasher);
                args.hash_stable(__hcx, __hasher);
            }
            Coroutine(def_id, args, m) => {
                def_id.hash_stable(__hcx, __hasher);
                args.hash_stable(__hcx, __hasher);
                m.hash_stable(__hcx, __hasher);
            }
            CoroutineWitness(def_id, args) => {
                def_id.hash_stable(__hcx, __hasher);
                args.hash_stable(__hcx, __hasher);
            }
            Never => {}
            Tuple(args) => {
                args.hash_stable(__hcx, __hasher);
            }
            Alias(k, p) => {
                k.hash_stable(__hcx, __hasher);
                p.hash_stable(__hcx, __hasher);
            }
            Param(p) => {
                p.hash_stable(__hcx, __hasher);
            }
            Bound(d, b) => {
                d.hash_stable(__hcx, __hasher);
                b.hash_stable(__hcx, __hasher);
            }
            Placeholder(p) => {
                p.hash_stable(__hcx, __hasher);
            }
            Infer(i) => {
                i.hash_stable(__hcx, __hasher);
            }
            Error(d) => {
                d.hash_stable(__hcx, __hasher);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(Encodable, Decodable, HashStable_Generic)]
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
#[derive(Encodable, Decodable, HashStable_Generic)]
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
#[derive(Encodable, Decodable, HashStable_Generic)]
pub enum FloatTy {
    F32,
    F64,
}

impl FloatTy {
    pub fn name_str(self) -> &'static str {
        match self {
            FloatTy::F32 => "f32",
            FloatTy::F64 => "f64",
        }
    }

    pub fn bit_width(self) -> u64 {
        match self {
            FloatTy::F32 => 32,
            FloatTy::F64 => 64,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IntVarValue {
    IntType(IntTy),
    UintType(UintTy),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FloatVarValue(pub FloatTy);

rustc_index::newtype_index! {
    /// A **ty**pe **v**ariable **ID**.
    #[debug_format = "?{}t"]
    pub struct TyVid {}
}

rustc_index::newtype_index! {
    /// An **int**egral (`u32`, `i32`, `usize`, etc.) type **v**ariable **ID**.
    #[debug_format = "?{}i"]
    pub struct IntVid {}
}

rustc_index::newtype_index! {
    /// A **float**ing-point (`f32` or `f64`) type **v**ariable **ID**.
    #[debug_format = "?{}f"]
    pub struct FloatVid {}
}

/// A placeholder for a type that hasn't been inferred yet.
///
/// E.g., if we have an empty array (`[]`), then we create a fresh
/// type variable for the element type since we won't know until it's
/// used what the element type is supposed to be.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Encodable, Decodable)]
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

impl EqUnifyValue for IntVarValue {}

impl UnifyKey for IntVid {
    type Value = Option<IntVarValue>;
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

impl EqUnifyValue for FloatVarValue {}

impl UnifyKey for FloatVid {
    type Value = Option<FloatVarValue>;
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

impl<CTX> HashStable<CTX> for InferTy {
    fn hash_stable(&self, ctx: &mut CTX, hasher: &mut StableHasher) {
        use InferTy::*;
        discriminant(self).hash_stable(ctx, hasher);
        match self {
            TyVar(_) | IntVar(_) | FloatVar(_) => {
                panic!("type variables should not be hashed: {self:?}")
            }
            FreshTy(v) | FreshIntTy(v) | FreshFloatTy(v) => v.hash_stable(ctx, hasher),
        }
    }
}

impl fmt::Debug for IntVarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            IntVarValue::IntType(ref v) => v.fmt(f),
            IntVarValue::UintType(ref v) => v.fmt(f),
        }
    }
}

impl fmt::Debug for FloatVarValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
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

impl<I: Interner<InferTy = InferTy>> DebugWithInfcx<I> for InferTy {
    fn fmt<Infcx: InferCtxtLike<Interner = I>>(
        this: WithInfcx<'_, Infcx, &Self>,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        use InferTy::*;
        match this.infcx.universe_of_ty(*this.data) {
            None => write!(f, "{:?}", this.data),
            Some(universe) => match *this.data {
                TyVar(ty_vid) => write!(f, "?{}_{}t", ty_vid.index(), universe.index()),
                IntVar(_) | FloatVar(_) | FreshTy(_) | FreshIntTy(_) | FreshFloatTy(_) => {
                    unreachable!()
                }
            },
        }
    }
}
