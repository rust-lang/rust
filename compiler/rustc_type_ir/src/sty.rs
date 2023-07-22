#![allow(rustc::usage_of_ty_tykind)]

use std::cmp::Ordering;
use std::{fmt, hash};

use crate::FloatTy;
use crate::HashStableContext;
use crate::IntTy;
use crate::Interner;
use crate::TyDecoder;
use crate::TyEncoder;
use crate::UintTy;
use crate::{DebruijnIndex, DebugWithInfcx, InferCtxtLike, OptWithInfcx};

use self::RegionKind::*;
use self::TyKind::*;

use rustc_data_structures::stable_hasher::HashStable;
use rustc_serialize::{Decodable, Decoder, Encodable};

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
    Adt(I::AdtDef, I::GenericArgsRef),

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
    Ref(I::Region, I::Ty, I::Mutability),

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
    FnDef(I::DefId, I::GenericArgsRef),

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
    Dynamic(I::ListBinderExistentialPredicate, I::Region, DynKind),

    /// The anonymous type of a closure. Used to represent the type of `|a| a`.
    ///
    /// Closure args contain both the - potentially substituted - generic parameters
    /// of its parent and some synthetic parameters. See the documentation for
    /// `ClosureArgs` for more details.
    Closure(I::DefId, I::GenericArgsRef),

    /// The anonymous type of a generator. Used to represent the type of
    /// `|a| yield a`.
    ///
    /// For more info about generator args, visit the documentation for
    /// `GeneratorArgs`.
    Generator(I::DefId, I::GenericArgsRef, I::Movability),

    /// A type representing the types stored inside a generator.
    /// This should only appear as part of the `GeneratorArgs`.
    ///
    /// Note that the captured variables for generators are stored separately
    /// using a tuple in the same way as for closures.
    ///
    /// Unlike upvars, the witness can reference lifetimes from
    /// inside of the generator itself. To deal with them in
    /// the type of the generator, we convert them to higher ranked
    /// lifetimes bound by the witness itself.
    ///
    /// Looking at the following example, the witness for this generator
    /// may end up as something like `for<'a> [Vec<i32>, &'a Vec<i32>]`:
    ///
    /// ```ignore UNSOLVED (ask @compiler-errors, should this error? can we just swap the yields?)
    /// #![feature(generators)]
    /// |a| {
    ///     let x = &vec![3];
    ///     yield a;
    ///     yield x[0];
    /// }
    /// # ;
    /// ```
    GeneratorWitness(I::BinderListTy),

    /// A type representing the types stored inside a generator.
    /// This should only appear as part of the `GeneratorArgs`.
    ///
    /// Unlike upvars, the witness can reference lifetimes from
    /// inside of the generator itself. To deal with them in
    /// the type of the generator, we convert them to higher ranked
    /// lifetimes bound by the witness itself.
    ///
    /// This variant is only using when `drop_tracking_mir` is set.
    /// This contains the `DefId` and the `GenericArgsRef` of the generator.
    /// The actual witness types are computed on MIR by the `mir_generator_witnesses` query.
    ///
    /// Looking at the following example, the witness for this generator
    /// may end up as something like `for<'a> [Vec<i32>, &'a Vec<i32>]`:
    ///
    /// ```ignore UNSOLVED (ask @compiler-errors, should this error? can we just swap the yields?)
    /// #![feature(generators)]
    /// |a| {
    ///     let x = &vec![3];
    ///     yield a;
    ///     yield x[0];
    /// }
    /// # ;
    /// ```
    GeneratorWitnessMIR(I::DefId, I::GenericArgsRef),

    /// The never type `!`.
    Never,

    /// A tuple type. For example, `(i32, bool)`.
    Tuple(I::ListTy),

    /// A projection or opaque type. Both of these types
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
    Placeholder(I::PlaceholderType),

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
        Generator(_, _, _) => 16,
        GeneratorWitness(_) => 17,
        Never => 18,
        Tuple(_) => 19,
        Alias(_, _) => 20,
        Param(_) => 21,
        Bound(_, _) => 22,
        Placeholder(_) => 23,
        Infer(_) => 24,
        Error(_) => 25,
        GeneratorWitnessMIR(_, _) => 26,
    }
}

// This is manually implemented because a derive would require `I: Clone`
impl<I: Interner> Clone for TyKind<I> {
    fn clone(&self) -> Self {
        match self {
            Bool => Bool,
            Char => Char,
            Int(i) => Int(*i),
            Uint(u) => Uint(*u),
            Float(f) => Float(*f),
            Adt(d, s) => Adt(d.clone(), s.clone()),
            Foreign(d) => Foreign(d.clone()),
            Str => Str,
            Array(t, c) => Array(t.clone(), c.clone()),
            Slice(t) => Slice(t.clone()),
            RawPtr(p) => RawPtr(p.clone()),
            Ref(r, t, m) => Ref(r.clone(), t.clone(), m.clone()),
            FnDef(d, s) => FnDef(d.clone(), s.clone()),
            FnPtr(s) => FnPtr(s.clone()),
            Dynamic(p, r, repr) => Dynamic(p.clone(), r.clone(), *repr),
            Closure(d, s) => Closure(d.clone(), s.clone()),
            Generator(d, s, m) => Generator(d.clone(), s.clone(), m.clone()),
            GeneratorWitness(g) => GeneratorWitness(g.clone()),
            GeneratorWitnessMIR(d, s) => GeneratorWitnessMIR(d.clone(), s.clone()),
            Never => Never,
            Tuple(t) => Tuple(t.clone()),
            Alias(k, p) => Alias(*k, p.clone()),
            Param(p) => Param(p.clone()),
            Bound(d, b) => Bound(*d, b.clone()),
            Placeholder(p) => Placeholder(p.clone()),
            Infer(t) => Infer(t.clone()),
            Error(e) => Error(e.clone()),
        }
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
            (Generator(a_d, a_s, a_m), Generator(b_d, b_s, b_m)) => {
                a_d == b_d && a_s == b_s && a_m == b_m
            }
            (GeneratorWitness(a_g), GeneratorWitness(b_g)) => a_g == b_g,
            (GeneratorWitnessMIR(a_d, a_s), GeneratorWitnessMIR(b_d, b_s)) => {
                a_d == b_d && a_s == b_s
            }
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

// This is manually implemented because a derive would require `I: PartialOrd`
impl<I: Interner> PartialOrd for TyKind<I> {
    #[inline]
    fn partial_cmp(&self, other: &TyKind<I>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// This is manually implemented because a derive would require `I: Ord`
impl<I: Interner> Ord for TyKind<I> {
    #[inline]
    fn cmp(&self, other: &TyKind<I>) -> Ordering {
        tykind_discriminant(self).cmp(&tykind_discriminant(other)).then_with(|| {
            match (self, other) {
                (Int(a_i), Int(b_i)) => a_i.cmp(b_i),
                (Uint(a_u), Uint(b_u)) => a_u.cmp(b_u),
                (Float(a_f), Float(b_f)) => a_f.cmp(b_f),
                (Adt(a_d, a_s), Adt(b_d, b_s)) => a_d.cmp(b_d).then_with(|| a_s.cmp(b_s)),
                (Foreign(a_d), Foreign(b_d)) => a_d.cmp(b_d),
                (Array(a_t, a_c), Array(b_t, b_c)) => a_t.cmp(b_t).then_with(|| a_c.cmp(b_c)),
                (Slice(a_t), Slice(b_t)) => a_t.cmp(b_t),
                (RawPtr(a_t), RawPtr(b_t)) => a_t.cmp(b_t),
                (Ref(a_r, a_t, a_m), Ref(b_r, b_t, b_m)) => {
                    a_r.cmp(b_r).then_with(|| a_t.cmp(b_t).then_with(|| a_m.cmp(b_m)))
                }
                (FnDef(a_d, a_s), FnDef(b_d, b_s)) => a_d.cmp(b_d).then_with(|| a_s.cmp(b_s)),
                (FnPtr(a_s), FnPtr(b_s)) => a_s.cmp(b_s),
                (Dynamic(a_p, a_r, a_repr), Dynamic(b_p, b_r, b_repr)) => {
                    a_p.cmp(b_p).then_with(|| a_r.cmp(b_r).then_with(|| a_repr.cmp(b_repr)))
                }
                (Closure(a_p, a_s), Closure(b_p, b_s)) => a_p.cmp(b_p).then_with(|| a_s.cmp(b_s)),
                (Generator(a_d, a_s, a_m), Generator(b_d, b_s, b_m)) => {
                    a_d.cmp(b_d).then_with(|| a_s.cmp(b_s).then_with(|| a_m.cmp(b_m)))
                }
                (GeneratorWitness(a_g), GeneratorWitness(b_g)) => a_g.cmp(b_g),
                (
                    GeneratorWitnessMIR(a_d, a_s),
                    GeneratorWitnessMIR(b_d, b_s),
                ) => match Ord::cmp(a_d, b_d) {
                    Ordering::Equal => Ord::cmp(a_s, b_s),
                    cmp => cmp,
                },
                (Tuple(a_t), Tuple(b_t)) => a_t.cmp(b_t),
                (Alias(a_i, a_p), Alias(b_i, b_p)) => a_i.cmp(b_i).then_with(|| a_p.cmp(b_p)),
                (Param(a_p), Param(b_p)) => a_p.cmp(b_p),
                (Bound(a_d, a_b), Bound(b_d, b_b)) => a_d.cmp(b_d).then_with(|| a_b.cmp(b_b)),
                (Placeholder(a_p), Placeholder(b_p)) => a_p.cmp(b_p),
                (Infer(a_t), Infer(b_t)) => a_t.cmp(b_t),
                (Error(a_e), Error(b_e)) => a_e.cmp(b_e),
                (Bool, Bool) | (Char, Char) | (Str, Str) | (Never, Never) => Ordering::Equal,
                _ => {
                    debug_assert!(false, "This branch must be unreachable, maybe the match is missing an arm? self = {self:?}, other = {other:?}");
                    Ordering::Equal
                }
            }
        })
    }
}

// This is manually implemented because a derive would require `I: Hash`
impl<I: Interner> hash::Hash for TyKind<I> {
    fn hash<__H: hash::Hasher>(&self, state: &mut __H) -> () {
        tykind_discriminant(self).hash(state);
        match self {
            Int(i) => i.hash(state),
            Uint(u) => u.hash(state),
            Float(f) => f.hash(state),
            Adt(d, s) => {
                d.hash(state);
                s.hash(state)
            }
            Foreign(d) => d.hash(state),
            Array(t, c) => {
                t.hash(state);
                c.hash(state)
            }
            Slice(t) => t.hash(state),
            RawPtr(t) => t.hash(state),
            Ref(r, t, m) => {
                r.hash(state);
                t.hash(state);
                m.hash(state)
            }
            FnDef(d, s) => {
                d.hash(state);
                s.hash(state)
            }
            FnPtr(s) => s.hash(state),
            Dynamic(p, r, repr) => {
                p.hash(state);
                r.hash(state);
                repr.hash(state)
            }
            Closure(d, s) => {
                d.hash(state);
                s.hash(state)
            }
            Generator(d, s, m) => {
                d.hash(state);
                s.hash(state);
                m.hash(state)
            }
            GeneratorWitness(g) => g.hash(state),
            GeneratorWitnessMIR(d, s) => {
                d.hash(state);
                s.hash(state);
            }
            Tuple(t) => t.hash(state),
            Alias(i, p) => {
                i.hash(state);
                p.hash(state);
            }
            Param(p) => p.hash(state),
            Bound(d, b) => {
                d.hash(state);
                b.hash(state)
            }
            Placeholder(p) => p.hash(state),
            Infer(t) => t.hash(state),
            Error(e) => e.hash(state),
            Bool | Char | Str | Never => (),
        }
    }
}

impl<I: Interner> DebugWithInfcx<I> for TyKind<I> {
    fn fmt<InfCtx: InferCtxtLike<I>>(
        this: OptWithInfcx<'_, I, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> fmt::Result {
        match this.data {
            Bool => write!(f, "bool"),
            Char => write!(f, "char"),
            Int(i) => write!(f, "{i:?}"),
            Uint(u) => write!(f, "{u:?}"),
            Float(float) => write!(f, "{float:?}"),
            Adt(d, s) => f.debug_tuple_field2_finish("Adt", d, &this.wrap(s)),
            Foreign(d) => f.debug_tuple_field1_finish("Foreign", d),
            Str => write!(f, "str"),
            Array(t, c) => write!(f, "[{:?}; {:?}]", &this.wrap(t), &this.wrap(c)),
            Slice(t) => write!(f, "[{:?}]", &this.wrap(t)),
            RawPtr(p) => {
                let (ty, mutbl) = I::ty_and_mut_to_parts(p.clone());
                match I::mutability_is_mut(mutbl) {
                    true => write!(f, "*mut "),
                    false => write!(f, "*const "),
                }?;
                write!(f, "{:?}", &this.wrap(ty))
            }
            Ref(r, t, m) => match I::mutability_is_mut(m.clone()) {
                true => write!(f, "&{:?} mut {:?}", &this.wrap(r), &this.wrap(t)),
                false => write!(f, "&{:?} {:?}", &this.wrap(r), &this.wrap(t)),
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
            Generator(d, s, m) => f.debug_tuple_field3_finish("Generator", d, &this.wrap(s), m),
            GeneratorWitness(g) => f.debug_tuple_field1_finish("GeneratorWitness", &this.wrap(g)),
            GeneratorWitnessMIR(d, s) => {
                f.debug_tuple_field2_finish("GeneratorWitnessMIR", d, &this.wrap(s))
            }
            Never => write!(f, "!"),
            Tuple(t) => {
                let mut iter = t.clone().into_iter();

                write!(f, "(")?;

                match iter.next() {
                    None => return write!(f, ")"),
                    Some(ty) => write!(f, "{:?}", &this.wrap(ty))?,
                };

                match iter.next() {
                    None => return write!(f, ",)"),
                    Some(ty) => write!(f, "{:?})", &this.wrap(ty))?,
                }

                for ty in iter {
                    write!(f, ", {:?}", &this.wrap(ty))?;
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
        OptWithInfcx::new_no_ctx(self).fmt(f)
    }
}

// This is manually implemented because a derive would require `I: Encodable`
impl<I: Interner, E: TyEncoder> Encodable<E> for TyKind<I>
where
    I::ErrorGuaranteed: Encodable<E>,
    I::AdtDef: Encodable<E>,
    I::GenericArgsRef: Encodable<E>,
    I::DefId: Encodable<E>,
    I::Ty: Encodable<E>,
    I::Const: Encodable<E>,
    I::Region: Encodable<E>,
    I::TypeAndMut: Encodable<E>,
    I::Mutability: Encodable<E>,
    I::Movability: Encodable<E>,
    I::PolyFnSig: Encodable<E>,
    I::ListBinderExistentialPredicate: Encodable<E>,
    I::BinderListTy: Encodable<E>,
    I::ListTy: Encodable<E>,
    I::AliasTy: Encodable<E>,
    I::ParamTy: Encodable<E>,
    I::BoundTy: Encodable<E>,
    I::PlaceholderType: Encodable<E>,
    I::InferTy: Encodable<E>,
    I::PredicateKind: Encodable<E>,
    I::AllocId: Encodable<E>,
{
    fn encode(&self, e: &mut E) {
        let disc = tykind_discriminant(self);
        match self {
            Bool => e.emit_enum_variant(disc, |_| {}),
            Char => e.emit_enum_variant(disc, |_| {}),
            Int(i) => e.emit_enum_variant(disc, |e| {
                i.encode(e);
            }),
            Uint(u) => e.emit_enum_variant(disc, |e| {
                u.encode(e);
            }),
            Float(f) => e.emit_enum_variant(disc, |e| {
                f.encode(e);
            }),
            Adt(adt, args) => e.emit_enum_variant(disc, |e| {
                adt.encode(e);
                args.encode(e);
            }),
            Foreign(def_id) => e.emit_enum_variant(disc, |e| {
                def_id.encode(e);
            }),
            Str => e.emit_enum_variant(disc, |_| {}),
            Array(t, c) => e.emit_enum_variant(disc, |e| {
                t.encode(e);
                c.encode(e);
            }),
            Slice(t) => e.emit_enum_variant(disc, |e| {
                t.encode(e);
            }),
            RawPtr(tam) => e.emit_enum_variant(disc, |e| {
                tam.encode(e);
            }),
            Ref(r, t, m) => e.emit_enum_variant(disc, |e| {
                r.encode(e);
                t.encode(e);
                m.encode(e);
            }),
            FnDef(def_id, args) => e.emit_enum_variant(disc, |e| {
                def_id.encode(e);
                args.encode(e);
            }),
            FnPtr(polyfnsig) => e.emit_enum_variant(disc, |e| {
                polyfnsig.encode(e);
            }),
            Dynamic(l, r, repr) => e.emit_enum_variant(disc, |e| {
                l.encode(e);
                r.encode(e);
                repr.encode(e);
            }),
            Closure(def_id, args) => e.emit_enum_variant(disc, |e| {
                def_id.encode(e);
                args.encode(e);
            }),
            Generator(def_id, args, m) => e.emit_enum_variant(disc, |e| {
                def_id.encode(e);
                args.encode(e);
                m.encode(e);
            }),
            GeneratorWitness(b) => e.emit_enum_variant(disc, |e| {
                b.encode(e);
            }),
            GeneratorWitnessMIR(def_id, args) => e.emit_enum_variant(disc, |e| {
                def_id.encode(e);
                args.encode(e);
            }),
            Never => e.emit_enum_variant(disc, |_| {}),
            Tuple(args) => e.emit_enum_variant(disc, |e| {
                args.encode(e);
            }),
            Alias(k, p) => e.emit_enum_variant(disc, |e| {
                k.encode(e);
                p.encode(e);
            }),
            Param(p) => e.emit_enum_variant(disc, |e| {
                p.encode(e);
            }),
            Bound(d, b) => e.emit_enum_variant(disc, |e| {
                d.encode(e);
                b.encode(e);
            }),
            Placeholder(p) => e.emit_enum_variant(disc, |e| {
                p.encode(e);
            }),
            Infer(i) => e.emit_enum_variant(disc, |e| {
                i.encode(e);
            }),
            Error(d) => e.emit_enum_variant(disc, |e| {
                d.encode(e);
            }),
        }
    }
}

// This is manually implemented because a derive would require `I: Decodable`
impl<I: Interner, D: TyDecoder<I = I>> Decodable<D> for TyKind<I>
where
    I::ErrorGuaranteed: Decodable<D>,
    I::AdtDef: Decodable<D>,
    I::GenericArgsRef: Decodable<D>,
    I::DefId: Decodable<D>,
    I::Ty: Decodable<D>,
    I::Const: Decodable<D>,
    I::Region: Decodable<D>,
    I::TypeAndMut: Decodable<D>,
    I::Mutability: Decodable<D>,
    I::Movability: Decodable<D>,
    I::PolyFnSig: Decodable<D>,
    I::ListBinderExistentialPredicate: Decodable<D>,
    I::BinderListTy: Decodable<D>,
    I::ListTy: Decodable<D>,
    I::AliasTy: Decodable<D>,
    I::ParamTy: Decodable<D>,
    I::AliasTy: Decodable<D>,
    I::BoundTy: Decodable<D>,
    I::PlaceholderType: Decodable<D>,
    I::InferTy: Decodable<D>,
    I::PredicateKind: Decodable<D>,
    I::AllocId: Decodable<D>,
{
    fn decode(d: &mut D) -> Self {
        match Decoder::read_usize(d) {
            0 => Bool,
            1 => Char,
            2 => Int(Decodable::decode(d)),
            3 => Uint(Decodable::decode(d)),
            4 => Float(Decodable::decode(d)),
            5 => Adt(Decodable::decode(d), Decodable::decode(d)),
            6 => Foreign(Decodable::decode(d)),
            7 => Str,
            8 => Array(Decodable::decode(d), Decodable::decode(d)),
            9 => Slice(Decodable::decode(d)),
            10 => RawPtr(Decodable::decode(d)),
            11 => Ref(Decodable::decode(d), Decodable::decode(d), Decodable::decode(d)),
            12 => FnDef(Decodable::decode(d), Decodable::decode(d)),
            13 => FnPtr(Decodable::decode(d)),
            14 => Dynamic(Decodable::decode(d), Decodable::decode(d), Decodable::decode(d)),
            15 => Closure(Decodable::decode(d), Decodable::decode(d)),
            16 => Generator(Decodable::decode(d), Decodable::decode(d), Decodable::decode(d)),
            17 => GeneratorWitness(Decodable::decode(d)),
            18 => Never,
            19 => Tuple(Decodable::decode(d)),
            20 => Alias(Decodable::decode(d), Decodable::decode(d)),
            21 => Param(Decodable::decode(d)),
            22 => Bound(Decodable::decode(d), Decodable::decode(d)),
            23 => Placeholder(Decodable::decode(d)),
            24 => Infer(Decodable::decode(d)),
            25 => Error(Decodable::decode(d)),
            26 => GeneratorWitnessMIR(Decodable::decode(d), Decodable::decode(d)),
            _ => panic!(
                "{}",
                format!(
                    "invalid enum variant tag while decoding `{}`, expected 0..{}",
                    "TyKind", 27,
                )
            ),
        }
    }
}

// This is not a derived impl because a derive would require `I: HashStable`
#[allow(rustc::usage_of_ty_tykind)]
impl<CTX: HashStableContext, I: Interner> HashStable<CTX> for TyKind<I>
where
    I::AdtDef: HashStable<CTX>,
    I::DefId: HashStable<CTX>,
    I::GenericArgsRef: HashStable<CTX>,
    I::Ty: HashStable<CTX>,
    I::Const: HashStable<CTX>,
    I::TypeAndMut: HashStable<CTX>,
    I::PolyFnSig: HashStable<CTX>,
    I::ListBinderExistentialPredicate: HashStable<CTX>,
    I::Region: HashStable<CTX>,
    I::Movability: HashStable<CTX>,
    I::Mutability: HashStable<CTX>,
    I::BinderListTy: HashStable<CTX>,
    I::ListTy: HashStable<CTX>,
    I::AliasTy: HashStable<CTX>,
    I::BoundTy: HashStable<CTX>,
    I::ParamTy: HashStable<CTX>,
    I::PlaceholderType: HashStable<CTX>,
    I::InferTy: HashStable<CTX>,
    I::ErrorGuaranteed: HashStable<CTX>,
{
    #[inline]
    fn hash_stable(
        &self,
        __hcx: &mut CTX,
        __hasher: &mut rustc_data_structures::stable_hasher::StableHasher,
    ) {
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
            Generator(def_id, args, m) => {
                def_id.hash_stable(__hcx, __hasher);
                args.hash_stable(__hcx, __hasher);
                m.hash_stable(__hcx, __hasher);
            }
            GeneratorWitness(b) => {
                b.hash_stable(__hcx, __hasher);
            }
            GeneratorWitnessMIR(def_id, args) => {
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

/// Represents a constant in Rust.
// #[derive(derive_more::From)]
pub enum ConstKind<I: Interner> {
    /// A const generic parameter.
    Param(I::ParamConst),

    /// Infer the value of the const.
    Infer(I::InferConst),

    /// Bound const variable, used only when preparing a trait query.
    Bound(DebruijnIndex, I::BoundConst),

    /// A placeholder const - universally quantified higher-ranked const.
    Placeholder(I::PlaceholderConst),

    /// An unnormalized const item such as an anon const or assoc const or free const item.
    /// Right now anything other than anon consts does not actually work properly but this
    /// should
    Unevaluated(I::AliasConst),

    /// Used to hold computed value.
    Value(I::ValueConst),

    /// A placeholder for a const which could not be computed; this is
    /// propagated to avoid useless error messages.
    Error(I::ErrorGuaranteed),

    /// Unevaluated non-const-item, used by `feature(generic_const_exprs)` to represent
    /// const arguments such as `N + 1` or `foo(N)`
    Expr(I::ExprConst),
}

const fn const_kind_discriminant<I: Interner>(value: &ConstKind<I>) -> usize {
    match value {
        ConstKind::Param(_) => 0,
        ConstKind::Infer(_) => 1,
        ConstKind::Bound(_, _) => 2,
        ConstKind::Placeholder(_) => 3,
        ConstKind::Unevaluated(_) => 4,
        ConstKind::Value(_) => 5,
        ConstKind::Error(_) => 6,
        ConstKind::Expr(_) => 7,
    }
}

impl<I: Interner> hash::Hash for ConstKind<I> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        const_kind_discriminant(self).hash(state);
        match self {
            ConstKind::Param(p) => p.hash(state),
            ConstKind::Infer(i) => i.hash(state),
            ConstKind::Bound(d, b) => {
                d.hash(state);
                b.hash(state);
            }
            ConstKind::Placeholder(p) => p.hash(state),
            ConstKind::Unevaluated(u) => u.hash(state),
            ConstKind::Value(v) => v.hash(state),
            ConstKind::Error(e) => e.hash(state),
            ConstKind::Expr(e) => e.hash(state),
        }
    }
}

impl<CTX: HashStableContext, I: Interner> HashStable<CTX> for ConstKind<I>
where
    I::ParamConst: HashStable<CTX>,
    I::InferConst: HashStable<CTX>,
    I::BoundConst: HashStable<CTX>,
    I::PlaceholderConst: HashStable<CTX>,
    I::AliasConst: HashStable<CTX>,
    I::ValueConst: HashStable<CTX>,
    I::ErrorGuaranteed: HashStable<CTX>,
    I::ExprConst: HashStable<CTX>,
{
    fn hash_stable(
        &self,
        hcx: &mut CTX,
        hasher: &mut rustc_data_structures::stable_hasher::StableHasher,
    ) {
        const_kind_discriminant(self).hash_stable(hcx, hasher);
        match self {
            ConstKind::Param(p) => p.hash_stable(hcx, hasher),
            ConstKind::Infer(i) => i.hash_stable(hcx, hasher),
            ConstKind::Bound(d, b) => {
                d.hash_stable(hcx, hasher);
                b.hash_stable(hcx, hasher);
            }
            ConstKind::Placeholder(p) => p.hash_stable(hcx, hasher),
            ConstKind::Unevaluated(u) => u.hash_stable(hcx, hasher),
            ConstKind::Value(v) => v.hash_stable(hcx, hasher),
            ConstKind::Error(e) => e.hash_stable(hcx, hasher),
            ConstKind::Expr(e) => e.hash_stable(hcx, hasher),
        }
    }
}

impl<I: Interner, D: TyDecoder<I = I>> Decodable<D> for ConstKind<I>
where
    I::ParamConst: Decodable<D>,
    I::InferConst: Decodable<D>,
    I::BoundConst: Decodable<D>,
    I::PlaceholderConst: Decodable<D>,
    I::AliasConst: Decodable<D>,
    I::ValueConst: Decodable<D>,
    I::ErrorGuaranteed: Decodable<D>,
    I::ExprConst: Decodable<D>,
{
    fn decode(d: &mut D) -> Self {
        match Decoder::read_usize(d) {
            0 => ConstKind::Param(Decodable::decode(d)),
            1 => ConstKind::Infer(Decodable::decode(d)),
            2 => ConstKind::Bound(Decodable::decode(d), Decodable::decode(d)),
            3 => ConstKind::Placeholder(Decodable::decode(d)),
            4 => ConstKind::Unevaluated(Decodable::decode(d)),
            5 => ConstKind::Value(Decodable::decode(d)),
            6 => ConstKind::Error(Decodable::decode(d)),
            7 => ConstKind::Expr(Decodable::decode(d)),
            _ => panic!(
                "{}",
                format!(
                    "invalid enum variant tag while decoding `{}`, expected 0..{}",
                    "ConstKind", 8,
                )
            ),
        }
    }
}

impl<I: Interner, E: TyEncoder<I = I>> Encodable<E> for ConstKind<I>
where
    I::ParamConst: Encodable<E>,
    I::InferConst: Encodable<E>,
    I::BoundConst: Encodable<E>,
    I::PlaceholderConst: Encodable<E>,
    I::AliasConst: Encodable<E>,
    I::ValueConst: Encodable<E>,
    I::ErrorGuaranteed: Encodable<E>,
    I::ExprConst: Encodable<E>,
{
    fn encode(&self, e: &mut E) {
        let disc = const_kind_discriminant(self);
        match self {
            ConstKind::Param(p) => e.emit_enum_variant(disc, |e| p.encode(e)),
            ConstKind::Infer(i) => e.emit_enum_variant(disc, |e| i.encode(e)),
            ConstKind::Bound(d, b) => e.emit_enum_variant(disc, |e| {
                d.encode(e);
                b.encode(e);
            }),
            ConstKind::Placeholder(p) => e.emit_enum_variant(disc, |e| p.encode(e)),
            ConstKind::Unevaluated(u) => e.emit_enum_variant(disc, |e| u.encode(e)),
            ConstKind::Value(v) => e.emit_enum_variant(disc, |e| v.encode(e)),
            ConstKind::Error(er) => e.emit_enum_variant(disc, |e| er.encode(e)),
            ConstKind::Expr(ex) => e.emit_enum_variant(disc, |e| ex.encode(e)),
        }
    }
}

impl<I: Interner> PartialOrd for ConstKind<I> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<I: Interner> Ord for ConstKind<I> {
    fn cmp(&self, other: &Self) -> Ordering {
        const_kind_discriminant(self)
            .cmp(&const_kind_discriminant(other))
            .then_with(|| match (self, other) {
                (ConstKind::Param(p1), ConstKind::Param(p2)) => p1.cmp(p2),
                (ConstKind::Infer(i1), ConstKind::Infer(i2)) => i1.cmp(i2),
                (ConstKind::Bound(d1, b1), ConstKind::Bound(d2, b2)) => d1.cmp(d2).then_with(|| b1.cmp(b2)),
                (ConstKind::Placeholder(p1), ConstKind::Placeholder(p2)) => p1.cmp(p2),
                (ConstKind::Unevaluated(u1), ConstKind::Unevaluated(u2)) => u1.cmp(u2),
                (ConstKind::Value(v1), ConstKind::Value(v2)) => v1.cmp(v2),
                (ConstKind::Error(e1), ConstKind::Error(e2)) => e1.cmp(e2),
                (ConstKind::Expr(e1), ConstKind::Expr(e2)) => e1.cmp(e2),
                _ => {
                    debug_assert!(false, "This branch must be unreachable, maybe the match is missing an arm? self = {self:?}, other = {other:?}");
                    Ordering::Equal
                }
            })
    }
}

impl<I: Interner> PartialEq for ConstKind<I> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Param(l0), Self::Param(r0)) => l0 == r0,
            (Self::Infer(l0), Self::Infer(r0)) => l0 == r0,
            (Self::Bound(l0, l1), Self::Bound(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::Placeholder(l0), Self::Placeholder(r0)) => l0 == r0,
            (Self::Unevaluated(l0), Self::Unevaluated(r0)) => l0 == r0,
            (Self::Value(l0), Self::Value(r0)) => l0 == r0,
            (Self::Error(l0), Self::Error(r0)) => l0 == r0,
            (Self::Expr(l0), Self::Expr(r0)) => l0 == r0,
            _ => false,
        }
    }
}

impl<I: Interner> Eq for ConstKind<I> {}

impl<I: Interner> Clone for ConstKind<I> {
    fn clone(&self) -> Self {
        match self {
            Self::Param(arg0) => Self::Param(arg0.clone()),
            Self::Infer(arg0) => Self::Infer(arg0.clone()),
            Self::Bound(arg0, arg1) => Self::Bound(arg0.clone(), arg1.clone()),
            Self::Placeholder(arg0) => Self::Placeholder(arg0.clone()),
            Self::Unevaluated(arg0) => Self::Unevaluated(arg0.clone()),
            Self::Value(arg0) => Self::Value(arg0.clone()),
            Self::Error(arg0) => Self::Error(arg0.clone()),
            Self::Expr(arg0) => Self::Expr(arg0.clone()),
        }
    }
}

/// Representation of regions. Note that the NLL checker uses a distinct
/// representation of regions. For this reason, it internally replaces all the
/// regions with inference variables -- the index of the variable is then used
/// to index into internal NLL data structures. See `rustc_const_eval::borrow_check`
/// module for more information.
///
/// Note: operations are on the wrapper `Region` type, which is interned,
/// rather than this type.
///
/// ## The Region lattice within a given function
///
/// In general, the region lattice looks like
///
/// ```text
/// static ----------+-----...------+       (greatest)
/// |                |              |
/// early-bound and  |              |
/// free regions     |              |
/// |                |              |
/// |                |              |
/// empty(root)   placeholder(U1)   |
/// |            /                  |
/// |           /         placeholder(Un)
/// empty(U1) --         /
/// |                   /
/// ...                /
/// |                 /
/// empty(Un) --------                      (smallest)
/// ```
///
/// Early-bound/free regions are the named lifetimes in scope from the
/// function declaration. They have relationships to one another
/// determined based on the declared relationships from the
/// function.
///
/// Note that inference variables and bound regions are not included
/// in this diagram. In the case of inference variables, they should
/// be inferred to some other region from the diagram. In the case of
/// bound regions, they are excluded because they don't make sense to
/// include -- the diagram indicates the relationship between free
/// regions.
///
/// ## Inference variables
///
/// During region inference, we sometimes create inference variables,
/// represented as `ReVar`. These will be inferred by the code in
/// `infer::lexical_region_resolve` to some free region from the
/// lattice above (the minimal region that meets the
/// constraints).
///
/// During NLL checking, where regions are defined differently, we
/// also use `ReVar` -- in that case, the index is used to index into
/// the NLL region checker's data structures. The variable may in fact
/// represent either a free region or an inference variable, in that
/// case.
///
/// ## Bound Regions
///
/// These are regions that are stored behind a binder and must be substituted
/// with some concrete region before being used. There are two kind of
/// bound regions: early-bound, which are bound in an item's `Generics`,
/// and are substituted by an `GenericArgs`, and late-bound, which are part of
/// higher-ranked types (e.g., `for<'a> fn(&'a ())`), and are substituted by
/// the likes of `liberate_late_bound_regions`. The distinction exists
/// because higher-ranked lifetimes aren't supported in all places. See [1][2].
///
/// Unlike `Param`s, bound regions are not supposed to exist "in the wild"
/// outside their binder, e.g., in types passed to type inference, and
/// should first be substituted (by placeholder regions, free regions,
/// or region variables).
///
/// ## Placeholder and Free Regions
///
/// One often wants to work with bound regions without knowing their precise
/// identity. For example, when checking a function, the lifetime of a borrow
/// can end up being assigned to some region parameter. In these cases,
/// it must be ensured that bounds on the region can't be accidentally
/// assumed without being checked.
///
/// To do this, we replace the bound regions with placeholder markers,
/// which don't satisfy any relation not explicitly provided.
///
/// There are two kinds of placeholder regions in rustc: `ReFree` and
/// `RePlaceholder`. When checking an item's body, `ReFree` is supposed
/// to be used. These also support explicit bounds: both the internally-stored
/// *scope*, which the region is assumed to outlive, as well as other
/// relations stored in the `FreeRegionMap`. Note that these relations
/// aren't checked when you `make_subregion` (or `eq_types`), only by
/// `resolve_regions_and_report_errors`.
///
/// When working with higher-ranked types, some region relations aren't
/// yet known, so you can't just call `resolve_regions_and_report_errors`.
/// `RePlaceholder` is designed for this purpose. In these contexts,
/// there's also the risk that some inference variable laying around will
/// get unified with your placeholder region: if you want to check whether
/// `for<'a> Foo<'_>: 'a`, and you substitute your bound region `'a`
/// with a placeholder region `'%a`, the variable `'_` would just be
/// instantiated to the placeholder region `'%a`, which is wrong because
/// the inference variable is supposed to satisfy the relation
/// *for every value of the placeholder region*. To ensure that doesn't
/// happen, you can use `leak_check`. This is more clearly explained
/// by the [rustc dev guide].
///
/// [1]: https://smallcultfollowing.com/babysteps/blog/2013/10/29/intermingled-parameter-lists/
/// [2]: https://smallcultfollowing.com/babysteps/blog/2013/11/04/intermingled-parameter-lists/
/// [rustc dev guide]: https://rustc-dev-guide.rust-lang.org/traits/hrtb.html
pub enum RegionKind<I: Interner> {
    /// Region bound in a type or fn declaration which will be
    /// substituted 'early' -- that is, at the same time when type
    /// parameters are substituted.
    ReEarlyBound(I::EarlyBoundRegion),

    /// Region bound in a function scope, which will be substituted when the
    /// function is called.
    ReLateBound(DebruijnIndex, I::BoundRegion),

    /// When checking a function body, the types of all arguments and so forth
    /// that refer to bound region parameters are modified to refer to free
    /// region parameters.
    ReFree(I::FreeRegion),

    /// Static data that has an "infinite" lifetime. Top in the region lattice.
    ReStatic,

    /// A region variable. Should not exist outside of type inference.
    ReVar(I::RegionVid),

    /// A placeholder region -- basically, the higher-ranked version of `ReFree`.
    /// Should not exist outside of type inference.
    RePlaceholder(I::PlaceholderRegion),

    /// Erased region, used by trait selection, in MIR and during codegen.
    ReErased,

    /// A region that resulted from some other error. Used exclusively for diagnostics.
    ReError(I::ErrorGuaranteed),
}

// This is manually implemented for `RegionKind` because `std::mem::discriminant`
// returns an opaque value that is `PartialEq` but not `PartialOrd`
#[inline]
const fn regionkind_discriminant<I: Interner>(value: &RegionKind<I>) -> usize {
    match value {
        ReEarlyBound(_) => 0,
        ReLateBound(_, _) => 1,
        ReFree(_) => 2,
        ReStatic => 3,
        ReVar(_) => 4,
        RePlaceholder(_) => 5,
        ReErased => 6,
        ReError(_) => 7,
    }
}

// This is manually implemented because a derive would require `I: Copy`
impl<I: Interner> Copy for RegionKind<I>
where
    I::EarlyBoundRegion: Copy,
    I::BoundRegion: Copy,
    I::FreeRegion: Copy,
    I::RegionVid: Copy,
    I::PlaceholderRegion: Copy,
    I::ErrorGuaranteed: Copy,
{
}

// This is manually implemented because a derive would require `I: Clone`
impl<I: Interner> Clone for RegionKind<I> {
    fn clone(&self) -> Self {
        match self {
            ReEarlyBound(r) => ReEarlyBound(r.clone()),
            ReLateBound(d, r) => ReLateBound(*d, r.clone()),
            ReFree(r) => ReFree(r.clone()),
            ReStatic => ReStatic,
            ReVar(r) => ReVar(r.clone()),
            RePlaceholder(r) => RePlaceholder(r.clone()),
            ReErased => ReErased,
            ReError(r) => ReError(r.clone()),
        }
    }
}

// This is manually implemented because a derive would require `I: PartialEq`
impl<I: Interner> PartialEq for RegionKind<I> {
    #[inline]
    fn eq(&self, other: &RegionKind<I>) -> bool {
        regionkind_discriminant(self) == regionkind_discriminant(other)
            && match (self, other) {
                (ReEarlyBound(a_r), ReEarlyBound(b_r)) => a_r == b_r,
                (ReLateBound(a_d, a_r), ReLateBound(b_d, b_r)) => a_d == b_d && a_r == b_r,
                (ReFree(a_r), ReFree(b_r)) => a_r == b_r,
                (ReStatic, ReStatic) => true,
                (ReVar(a_r), ReVar(b_r)) => a_r == b_r,
                (RePlaceholder(a_r), RePlaceholder(b_r)) => a_r == b_r,
                (ReErased, ReErased) => true,
                (ReError(_), ReError(_)) => true,
                _ => {
                    debug_assert!(
                        false,
                        "This branch must be unreachable, maybe the match is missing an arm? self = {self:?}, other = {other:?}"
                    );
                    true
                }
            }
    }
}

// This is manually implemented because a derive would require `I: Eq`
impl<I: Interner> Eq for RegionKind<I> {}

// This is manually implemented because a derive would require `I: PartialOrd`
impl<I: Interner> PartialOrd for RegionKind<I> {
    #[inline]
    fn partial_cmp(&self, other: &RegionKind<I>) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// This is manually implemented because a derive would require `I: Ord`
impl<I: Interner> Ord for RegionKind<I> {
    #[inline]
    fn cmp(&self, other: &RegionKind<I>) -> Ordering {
        regionkind_discriminant(self).cmp(&regionkind_discriminant(other)).then_with(|| {
            match (self, other) {
                (ReEarlyBound(a_r), ReEarlyBound(b_r)) => a_r.cmp(b_r),
                (ReLateBound(a_d, a_r), ReLateBound(b_d, b_r)) => {
                    a_d.cmp(b_d).then_with(|| a_r.cmp(b_r))
                }
                (ReFree(a_r), ReFree(b_r)) => a_r.cmp(b_r),
                (ReStatic, ReStatic) => Ordering::Equal,
                (ReVar(a_r), ReVar(b_r)) => a_r.cmp(b_r),
                (RePlaceholder(a_r), RePlaceholder(b_r)) => a_r.cmp(b_r),
                (ReErased, ReErased) => Ordering::Equal,
                _ => {
                    debug_assert!(false, "This branch must be unreachable, maybe the match is missing an arm? self = self = {self:?}, other = {other:?}");
                    Ordering::Equal
                }
            }
        })
    }
}

// This is manually implemented because a derive would require `I: Hash`
impl<I: Interner> hash::Hash for RegionKind<I> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) -> () {
        regionkind_discriminant(self).hash(state);
        match self {
            ReEarlyBound(r) => r.hash(state),
            ReLateBound(d, r) => {
                d.hash(state);
                r.hash(state)
            }
            ReFree(r) => r.hash(state),
            ReStatic => (),
            ReVar(r) => r.hash(state),
            RePlaceholder(r) => r.hash(state),
            ReErased => (),
            ReError(_) => (),
        }
    }
}

impl<I: Interner> DebugWithInfcx<I> for RegionKind<I> {
    fn fmt<InfCtx: InferCtxtLike<I>>(
        this: OptWithInfcx<'_, I, InfCtx, &Self>,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        match this.data {
            ReEarlyBound(data) => write!(f, "ReEarlyBound({data:?})"),

            ReLateBound(binder_id, bound_region) => {
                write!(f, "ReLateBound({binder_id:?}, {bound_region:?})")
            }

            ReFree(fr) => write!(f, "{fr:?}"),

            ReStatic => f.write_str("ReStatic"),

            ReVar(vid) => write!(f, "{:?}", &this.wrap(vid)),

            RePlaceholder(placeholder) => write!(f, "RePlaceholder({placeholder:?})"),

            ReErased => f.write_str("ReErased"),

            ReError(_) => f.write_str("ReError"),
        }
    }
}
impl<I: Interner> fmt::Debug for RegionKind<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        OptWithInfcx::new_no_ctx(self).fmt(f)
    }
}

// This is manually implemented because a derive would require `I: Encodable`
impl<I: Interner, E: TyEncoder> Encodable<E> for RegionKind<I>
where
    I::EarlyBoundRegion: Encodable<E>,
    I::BoundRegion: Encodable<E>,
    I::FreeRegion: Encodable<E>,
    I::RegionVid: Encodable<E>,
    I::PlaceholderRegion: Encodable<E>,
{
    fn encode(&self, e: &mut E) {
        let disc = regionkind_discriminant(self);
        match self {
            ReEarlyBound(a) => e.emit_enum_variant(disc, |e| {
                a.encode(e);
            }),
            ReLateBound(a, b) => e.emit_enum_variant(disc, |e| {
                a.encode(e);
                b.encode(e);
            }),
            ReFree(a) => e.emit_enum_variant(disc, |e| {
                a.encode(e);
            }),
            ReStatic => e.emit_enum_variant(disc, |_| {}),
            ReVar(a) => e.emit_enum_variant(disc, |e| {
                a.encode(e);
            }),
            RePlaceholder(a) => e.emit_enum_variant(disc, |e| {
                a.encode(e);
            }),
            ReErased => e.emit_enum_variant(disc, |_| {}),
            ReError(_) => e.emit_enum_variant(disc, |_| {}),
        }
    }
}

// This is manually implemented because a derive would require `I: Decodable`
impl<I: Interner, D: TyDecoder<I = I>> Decodable<D> for RegionKind<I>
where
    I::EarlyBoundRegion: Decodable<D>,
    I::BoundRegion: Decodable<D>,
    I::FreeRegion: Decodable<D>,
    I::RegionVid: Decodable<D>,
    I::PlaceholderRegion: Decodable<D>,
    I::ErrorGuaranteed: Decodable<D>,
{
    fn decode(d: &mut D) -> Self {
        match Decoder::read_usize(d) {
            0 => ReEarlyBound(Decodable::decode(d)),
            1 => ReLateBound(Decodable::decode(d), Decodable::decode(d)),
            2 => ReFree(Decodable::decode(d)),
            3 => ReStatic,
            4 => ReVar(Decodable::decode(d)),
            5 => RePlaceholder(Decodable::decode(d)),
            6 => ReErased,
            7 => ReError(Decodable::decode(d)),
            _ => panic!(
                "{}",
                format!(
                    "invalid enum variant tag while decoding `{}`, expected 0..{}",
                    "RegionKind", 8,
                )
            ),
        }
    }
}

// This is not a derived impl because a derive would require `I: HashStable`
impl<CTX: HashStableContext, I: Interner> HashStable<CTX> for RegionKind<I>
where
    I::EarlyBoundRegion: HashStable<CTX>,
    I::BoundRegion: HashStable<CTX>,
    I::FreeRegion: HashStable<CTX>,
    I::RegionVid: HashStable<CTX>,
    I::PlaceholderRegion: HashStable<CTX>,
{
    #[inline]
    fn hash_stable(
        &self,
        hcx: &mut CTX,
        hasher: &mut rustc_data_structures::stable_hasher::StableHasher,
    ) {
        std::mem::discriminant(self).hash_stable(hcx, hasher);
        match self {
            ReErased | ReStatic | ReError(_) => {
                // No variant fields to hash for these ...
            }
            ReLateBound(d, r) => {
                d.hash_stable(hcx, hasher);
                r.hash_stable(hcx, hasher);
            }
            ReEarlyBound(r) => {
                r.hash_stable(hcx, hasher);
            }
            ReFree(r) => {
                r.hash_stable(hcx, hasher);
            }
            RePlaceholder(r) => {
                r.hash_stable(hcx, hasher);
            }
            ReVar(_) => {
                panic!("region variables should not be hashed: {self:?}")
            }
        }
    }
}
