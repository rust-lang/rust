#![allow(rustc::usage_of_ty_tykind)]

use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::{fmt, hash};

use crate::DebruijnIndex;
use crate::FloatTy;
use crate::HashStableContext;
use crate::IntTy;
use crate::Interner;
use crate::TyDecoder;
use crate::TyEncoder;
use crate::UintTy;

use self::RegionKind::*;
use self::TyKind::*;

use rustc_data_structures::stable_hasher::HashStable;
use rustc_serialize::{Decodable, Decoder, Encodable};

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
    /// for `struct List<T>` and the substs `[i32]`.
    ///
    /// Note that generic parameters in fields only get lazily substituted
    /// by using something like `adt_def.all_fields().map(|field| field.ty(tcx, substs))`.
    Adt(I::AdtDef, I::SubstsRef),

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
    FnDef(I::DefId, I::SubstsRef),

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
    Dynamic(I::ListBinderExistentialPredicate, I::Region),

    /// The anonymous type of a closure. Used to represent the type of `|a| a`.
    ///
    /// Closure substs contain both the - potentially substituted - generic parameters
    /// of its parent and some synthetic parameters. See the documentation for
    /// `ClosureSubsts` for more details.
    Closure(I::DefId, I::SubstsRef),

    /// The anonymous type of a generator. Used to represent the type of
    /// `|a| yield a`.
    ///
    /// For more info about generator substs, visit the documentation for
    /// `GeneratorSubsts`.
    Generator(I::DefId, I::SubstsRef, I::Movability),

    /// A type representing the types stored inside a generator.
    /// This should only appear as part of the `GeneratorSubsts`.
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

    /// The never type `!`.
    Never,

    /// A tuple type. For example, `(i32, bool)`.
    Tuple(I::ListTy),

    /// The projection of an associated type. For example,
    /// `<T as Trait<..>>::N`.
    Projection(I::ProjectionTy),

    /// Opaque (`impl Trait`) type found in a return type.
    ///
    /// The `DefId` comes either from
    /// * the `impl Trait` ast::Ty node,
    /// * or the `type Foo = impl Trait` declaration
    ///
    /// For RPIT the substitutions are for the generics of the function,
    /// while for TAIT it is used for the generic parameters of the alias.
    ///
    /// During codegen, `tcx.type_of(def_id)` can be used to get the underlying type.
    Opaque(I::DefId, I::SubstsRef),

    /// A type parameter; for example, `T` in `fn f<T>(x: T) {}`.
    Param(I::ParamTy),

    /// Bound type variable, used to represent the `'a` in `for<'a> fn(&'a ())`.
    ///
    /// For canonical queries, we replace inference variables with bound variables,
    /// so e.g. when checking whether `&'_ (): Trait<_>` holds, we canonicalize that to
    /// `for<'a, T> &'a (): Trait<T>` and then convert the introduced bound variables
    /// back to inference variables in a new inference context when inside of the query.
    ///
    /// See the `rustc-dev-guide` for more details about
    /// [higher-ranked trait bounds][1] and [canonical queries][2].
    ///
    /// [1]: https://rustc-dev-guide.rust-lang.org/traits/hrtb.html
    /// [2]: https://rustc-dev-guide.rust-lang.org/traits/canonical-queries.html
    Bound(DebruijnIndex, I::BoundTy),

    /// A placeholder type, used during higher ranked subtyping to instantiate
    /// bound variables.
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
    Error(I::DelaySpanBugEmitted),
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
        Dynamic(_, _) => 14,
        Closure(_, _) => 15,
        Generator(_, _, _) => 16,
        GeneratorWitness(_) => 17,
        Never => 18,
        Tuple(_) => 19,
        Projection(_) => 20,
        Opaque(_, _) => 21,
        Param(_) => 22,
        Bound(_, _) => 23,
        Placeholder(_) => 24,
        Infer(_) => 25,
        Error(_) => 26,
    }
}

// This is manually implemented because a derive would require `I: Clone`
impl<I: Interner> Clone for TyKind<I> {
    fn clone(&self) -> Self {
        match self {
            Bool => Bool,
            Char => Char,
            Int(i) => Int(i.clone()),
            Uint(u) => Uint(u.clone()),
            Float(f) => Float(f.clone()),
            Adt(d, s) => Adt(d.clone(), s.clone()),
            Foreign(d) => Foreign(d.clone()),
            Str => Str,
            Array(t, c) => Array(t.clone(), c.clone()),
            Slice(t) => Slice(t.clone()),
            RawPtr(t) => RawPtr(t.clone()),
            Ref(r, t, m) => Ref(r.clone(), t.clone(), m.clone()),
            FnDef(d, s) => FnDef(d.clone(), s.clone()),
            FnPtr(s) => FnPtr(s.clone()),
            Dynamic(p, r) => Dynamic(p.clone(), r.clone()),
            Closure(d, s) => Closure(d.clone(), s.clone()),
            Generator(d, s, m) => Generator(d.clone(), s.clone(), m.clone()),
            GeneratorWitness(g) => GeneratorWitness(g.clone()),
            Never => Never,
            Tuple(t) => Tuple(t.clone()),
            Projection(p) => Projection(p.clone()),
            Opaque(d, s) => Opaque(d.clone(), s.clone()),
            Param(p) => Param(p.clone()),
            Bound(d, b) => Bound(d.clone(), b.clone()),
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
        let __self_vi = tykind_discriminant(self);
        let __arg_1_vi = tykind_discriminant(other);
        if __self_vi == __arg_1_vi {
            match (&*self, &*other) {
                (&Int(ref __self_0), &Int(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (&Uint(ref __self_0), &Uint(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (&Float(ref __self_0), &Float(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (&Adt(ref __self_0, ref __self_1), &Adt(ref __arg_1_0, ref __arg_1_1)) => {
                    __self_0 == __arg_1_0 && __self_1 == __arg_1_1
                }
                (&Foreign(ref __self_0), &Foreign(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (&Array(ref __self_0, ref __self_1), &Array(ref __arg_1_0, ref __arg_1_1)) => {
                    __self_0 == __arg_1_0 && __self_1 == __arg_1_1
                }
                (&Slice(ref __self_0), &Slice(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (&RawPtr(ref __self_0), &RawPtr(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (
                    &Ref(ref __self_0, ref __self_1, ref __self_2),
                    &Ref(ref __arg_1_0, ref __arg_1_1, ref __arg_1_2),
                ) => __self_0 == __arg_1_0 && __self_1 == __arg_1_1 && __self_2 == __arg_1_2,
                (&FnDef(ref __self_0, ref __self_1), &FnDef(ref __arg_1_0, ref __arg_1_1)) => {
                    __self_0 == __arg_1_0 && __self_1 == __arg_1_1
                }
                (&FnPtr(ref __self_0), &FnPtr(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (&Dynamic(ref __self_0, ref __self_1), &Dynamic(ref __arg_1_0, ref __arg_1_1)) => {
                    __self_0 == __arg_1_0 && __self_1 == __arg_1_1
                }
                (&Closure(ref __self_0, ref __self_1), &Closure(ref __arg_1_0, ref __arg_1_1)) => {
                    __self_0 == __arg_1_0 && __self_1 == __arg_1_1
                }
                (
                    &Generator(ref __self_0, ref __self_1, ref __self_2),
                    &Generator(ref __arg_1_0, ref __arg_1_1, ref __arg_1_2),
                ) => __self_0 == __arg_1_0 && __self_1 == __arg_1_1 && __self_2 == __arg_1_2,
                (&GeneratorWitness(ref __self_0), &GeneratorWitness(ref __arg_1_0)) => {
                    __self_0 == __arg_1_0
                }
                (&Tuple(ref __self_0), &Tuple(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (&Projection(ref __self_0), &Projection(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (&Opaque(ref __self_0, ref __self_1), &Opaque(ref __arg_1_0, ref __arg_1_1)) => {
                    __self_0 == __arg_1_0 && __self_1 == __arg_1_1
                }
                (&Param(ref __self_0), &Param(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (&Bound(ref __self_0, ref __self_1), &Bound(ref __arg_1_0, ref __arg_1_1)) => {
                    __self_0 == __arg_1_0 && __self_1 == __arg_1_1
                }
                (&Placeholder(ref __self_0), &Placeholder(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (&Infer(ref __self_0), &Infer(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (&Error(ref __self_0), &Error(ref __arg_1_0)) => __self_0 == __arg_1_0,
                _ => true,
            }
        } else {
            false
        }
    }
}

// This is manually implemented because a derive would require `I: Eq`
impl<I: Interner> Eq for TyKind<I> {}

// This is manually implemented because a derive would require `I: PartialOrd`
impl<I: Interner> PartialOrd for TyKind<I> {
    #[inline]
    fn partial_cmp(&self, other: &TyKind<I>) -> Option<Ordering> {
        Some(Ord::cmp(self, other))
    }
}

// This is manually implemented because a derive would require `I: Ord`
impl<I: Interner> Ord for TyKind<I> {
    #[inline]
    fn cmp(&self, other: &TyKind<I>) -> Ordering {
        let __self_vi = tykind_discriminant(self);
        let __arg_1_vi = tykind_discriminant(other);
        if __self_vi == __arg_1_vi {
            match (&*self, &*other) {
                (&Int(ref __self_0), &Int(ref __arg_1_0)) => Ord::cmp(__self_0, __arg_1_0),
                (&Uint(ref __self_0), &Uint(ref __arg_1_0)) => Ord::cmp(__self_0, __arg_1_0),
                (&Float(ref __self_0), &Float(ref __arg_1_0)) => Ord::cmp(__self_0, __arg_1_0),
                (&Adt(ref __self_0, ref __self_1), &Adt(ref __arg_1_0, ref __arg_1_1)) => {
                    match Ord::cmp(__self_0, __arg_1_0) {
                        Ordering::Equal => Ord::cmp(__self_1, __arg_1_1),
                        cmp => cmp,
                    }
                }
                (&Foreign(ref __self_0), &Foreign(ref __arg_1_0)) => Ord::cmp(__self_0, __arg_1_0),
                (&Array(ref __self_0, ref __self_1), &Array(ref __arg_1_0, ref __arg_1_1)) => {
                    match Ord::cmp(__self_0, __arg_1_0) {
                        Ordering::Equal => Ord::cmp(__self_1, __arg_1_1),
                        cmp => cmp,
                    }
                }
                (&Slice(ref __self_0), &Slice(ref __arg_1_0)) => Ord::cmp(__self_0, __arg_1_0),
                (&RawPtr(ref __self_0), &RawPtr(ref __arg_1_0)) => Ord::cmp(__self_0, __arg_1_0),
                (
                    &Ref(ref __self_0, ref __self_1, ref __self_2),
                    &Ref(ref __arg_1_0, ref __arg_1_1, ref __arg_1_2),
                ) => match Ord::cmp(__self_0, __arg_1_0) {
                    Ordering::Equal => match Ord::cmp(__self_1, __arg_1_1) {
                        Ordering::Equal => Ord::cmp(__self_2, __arg_1_2),
                        cmp => cmp,
                    },
                    cmp => cmp,
                },
                (&FnDef(ref __self_0, ref __self_1), &FnDef(ref __arg_1_0, ref __arg_1_1)) => {
                    match Ord::cmp(__self_0, __arg_1_0) {
                        Ordering::Equal => Ord::cmp(__self_1, __arg_1_1),
                        cmp => cmp,
                    }
                }
                (&FnPtr(ref __self_0), &FnPtr(ref __arg_1_0)) => Ord::cmp(__self_0, __arg_1_0),
                (&Dynamic(ref __self_0, ref __self_1), &Dynamic(ref __arg_1_0, ref __arg_1_1)) => {
                    match Ord::cmp(__self_0, __arg_1_0) {
                        Ordering::Equal => Ord::cmp(__self_1, __arg_1_1),
                        cmp => cmp,
                    }
                }
                (&Closure(ref __self_0, ref __self_1), &Closure(ref __arg_1_0, ref __arg_1_1)) => {
                    match Ord::cmp(__self_0, __arg_1_0) {
                        Ordering::Equal => Ord::cmp(__self_1, __arg_1_1),
                        cmp => cmp,
                    }
                }
                (
                    &Generator(ref __self_0, ref __self_1, ref __self_2),
                    &Generator(ref __arg_1_0, ref __arg_1_1, ref __arg_1_2),
                ) => match Ord::cmp(__self_0, __arg_1_0) {
                    Ordering::Equal => match Ord::cmp(__self_1, __arg_1_1) {
                        Ordering::Equal => Ord::cmp(__self_2, __arg_1_2),
                        cmp => cmp,
                    },
                    cmp => cmp,
                },
                (&GeneratorWitness(ref __self_0), &GeneratorWitness(ref __arg_1_0)) => {
                    Ord::cmp(__self_0, __arg_1_0)
                }
                (&Tuple(ref __self_0), &Tuple(ref __arg_1_0)) => Ord::cmp(__self_0, __arg_1_0),
                (&Projection(ref __self_0), &Projection(ref __arg_1_0)) => {
                    Ord::cmp(__self_0, __arg_1_0)
                }
                (&Opaque(ref __self_0, ref __self_1), &Opaque(ref __arg_1_0, ref __arg_1_1)) => {
                    match Ord::cmp(__self_0, __arg_1_0) {
                        Ordering::Equal => Ord::cmp(__self_1, __arg_1_1),
                        cmp => cmp,
                    }
                }
                (&Param(ref __self_0), &Param(ref __arg_1_0)) => Ord::cmp(__self_0, __arg_1_0),
                (&Bound(ref __self_0, ref __self_1), &Bound(ref __arg_1_0, ref __arg_1_1)) => {
                    match Ord::cmp(__self_0, __arg_1_0) {
                        Ordering::Equal => Ord::cmp(__self_1, __arg_1_1),
                        cmp => cmp,
                    }
                }
                (&Placeholder(ref __self_0), &Placeholder(ref __arg_1_0)) => {
                    Ord::cmp(__self_0, __arg_1_0)
                }
                (&Infer(ref __self_0), &Infer(ref __arg_1_0)) => Ord::cmp(__self_0, __arg_1_0),
                (&Error(ref __self_0), &Error(ref __arg_1_0)) => Ord::cmp(__self_0, __arg_1_0),
                _ => Ordering::Equal,
            }
        } else {
            Ord::cmp(&__self_vi, &__arg_1_vi)
        }
    }
}

// This is manually implemented because a derive would require `I: Hash`
impl<I: Interner> hash::Hash for TyKind<I> {
    fn hash<__H: hash::Hasher>(&self, state: &mut __H) -> () {
        match (&*self,) {
            (&Int(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Uint(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Float(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Adt(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&Foreign(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Array(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&Slice(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&RawPtr(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Ref(ref __self_0, ref __self_1, ref __self_2),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state);
                hash::Hash::hash(__self_2, state)
            }
            (&FnDef(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&FnPtr(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Dynamic(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&Closure(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&Generator(ref __self_0, ref __self_1, ref __self_2),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state);
                hash::Hash::hash(__self_2, state)
            }
            (&GeneratorWitness(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Tuple(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Projection(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Opaque(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&Param(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Bound(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&Placeholder(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Infer(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Error(ref __self_0),) => {
                hash::Hash::hash(&tykind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            _ => hash::Hash::hash(&tykind_discriminant(self), state),
        }
    }
}

// This is manually implemented because a derive would require `I: Debug`
impl<I: Interner> fmt::Debug for TyKind<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use std::fmt::*;
        match self {
            Bool => Formatter::write_str(f, "Bool"),
            Char => Formatter::write_str(f, "Char"),
            Int(f0) => Formatter::debug_tuple_field1_finish(f, "Int", f0),
            Uint(f0) => Formatter::debug_tuple_field1_finish(f, "Uint", f0),
            Float(f0) => Formatter::debug_tuple_field1_finish(f, "Float", f0),
            Adt(f0, f1) => Formatter::debug_tuple_field2_finish(f, "Adt", f0, f1),
            Foreign(f0) => Formatter::debug_tuple_field1_finish(f, "Foreign", f0),
            Str => Formatter::write_str(f, "Str"),
            Array(f0, f1) => Formatter::debug_tuple_field2_finish(f, "Array", f0, f1),
            Slice(f0) => Formatter::debug_tuple_field1_finish(f, "Slice", f0),
            RawPtr(f0) => Formatter::debug_tuple_field1_finish(f, "RawPtr", f0),
            Ref(f0, f1, f2) => Formatter::debug_tuple_field3_finish(f, "Ref", f0, f1, f2),
            FnDef(f0, f1) => Formatter::debug_tuple_field2_finish(f, "FnDef", f0, f1),
            FnPtr(f0) => Formatter::debug_tuple_field1_finish(f, "FnPtr", f0),
            Dynamic(f0, f1) => Formatter::debug_tuple_field2_finish(f, "Dynamic", f0, f1),
            Closure(f0, f1) => Formatter::debug_tuple_field2_finish(f, "Closure", f0, f1),
            Generator(f0, f1, f2) => {
                Formatter::debug_tuple_field3_finish(f, "Generator", f0, f1, f2)
            }
            GeneratorWitness(f0) => Formatter::debug_tuple_field1_finish(f, "GeneratorWitness", f0),
            Never => Formatter::write_str(f, "Never"),
            Tuple(f0) => Formatter::debug_tuple_field1_finish(f, "Tuple", f0),
            Projection(f0) => Formatter::debug_tuple_field1_finish(f, "Projection", f0),
            Opaque(f0, f1) => Formatter::debug_tuple_field2_finish(f, "Opaque", f0, f1),
            Param(f0) => Formatter::debug_tuple_field1_finish(f, "Param", f0),
            Bound(f0, f1) => Formatter::debug_tuple_field2_finish(f, "Bound", f0, f1),
            Placeholder(f0) => Formatter::debug_tuple_field1_finish(f, "Placeholder", f0),
            Infer(f0) => Formatter::debug_tuple_field1_finish(f, "Infer", f0),
            TyKind::Error(f0) => Formatter::debug_tuple_field1_finish(f, "Error", f0),
        }
    }
}

// This is manually implemented because a derive would require `I: Encodable`
impl<I: Interner, E: TyEncoder> Encodable<E> for TyKind<I>
where
    I::DelaySpanBugEmitted: Encodable<E>,
    I::AdtDef: Encodable<E>,
    I::SubstsRef: Encodable<E>,
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
    I::ProjectionTy: Encodable<E>,
    I::ParamTy: Encodable<E>,
    I::BoundTy: Encodable<E>,
    I::PlaceholderType: Encodable<E>,
    I::InferTy: Encodable<E>,
    I::DelaySpanBugEmitted: Encodable<E>,
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
            Adt(adt, substs) => e.emit_enum_variant(disc, |e| {
                adt.encode(e);
                substs.encode(e);
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
            FnDef(def_id, substs) => e.emit_enum_variant(disc, |e| {
                def_id.encode(e);
                substs.encode(e);
            }),
            FnPtr(polyfnsig) => e.emit_enum_variant(disc, |e| {
                polyfnsig.encode(e);
            }),
            Dynamic(l, r) => e.emit_enum_variant(disc, |e| {
                l.encode(e);
                r.encode(e);
            }),
            Closure(def_id, substs) => e.emit_enum_variant(disc, |e| {
                def_id.encode(e);
                substs.encode(e);
            }),
            Generator(def_id, substs, m) => e.emit_enum_variant(disc, |e| {
                def_id.encode(e);
                substs.encode(e);
                m.encode(e);
            }),
            GeneratorWitness(b) => e.emit_enum_variant(disc, |e| {
                b.encode(e);
            }),
            Never => e.emit_enum_variant(disc, |_| {}),
            Tuple(substs) => e.emit_enum_variant(disc, |e| {
                substs.encode(e);
            }),
            Projection(p) => e.emit_enum_variant(disc, |e| {
                p.encode(e);
            }),
            Opaque(def_id, substs) => e.emit_enum_variant(disc, |e| {
                def_id.encode(e);
                substs.encode(e);
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
    I::DelaySpanBugEmitted: Decodable<D>,
    I::AdtDef: Decodable<D>,
    I::SubstsRef: Decodable<D>,
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
    I::ProjectionTy: Decodable<D>,
    I::ParamTy: Decodable<D>,
    I::BoundTy: Decodable<D>,
    I::PlaceholderType: Decodable<D>,
    I::InferTy: Decodable<D>,
    I::DelaySpanBugEmitted: Decodable<D>,
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
            14 => Dynamic(Decodable::decode(d), Decodable::decode(d)),
            15 => Closure(Decodable::decode(d), Decodable::decode(d)),
            16 => Generator(Decodable::decode(d), Decodable::decode(d), Decodable::decode(d)),
            17 => GeneratorWitness(Decodable::decode(d)),
            18 => Never,
            19 => Tuple(Decodable::decode(d)),
            20 => Projection(Decodable::decode(d)),
            21 => Opaque(Decodable::decode(d), Decodable::decode(d)),
            22 => Param(Decodable::decode(d)),
            23 => Bound(Decodable::decode(d), Decodable::decode(d)),
            24 => Placeholder(Decodable::decode(d)),
            25 => Infer(Decodable::decode(d)),
            26 => Error(Decodable::decode(d)),
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
    I::SubstsRef: HashStable<CTX>,
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
    I::ProjectionTy: HashStable<CTX>,
    I::BoundTy: HashStable<CTX>,
    I::ParamTy: HashStable<CTX>,
    I::PlaceholderType: HashStable<CTX>,
    I::InferTy: HashStable<CTX>,
    I::DelaySpanBugEmitted: HashStable<CTX>,
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
            Adt(adt, substs) => {
                adt.hash_stable(__hcx, __hasher);
                substs.hash_stable(__hcx, __hasher);
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
            FnDef(def_id, substs) => {
                def_id.hash_stable(__hcx, __hasher);
                substs.hash_stable(__hcx, __hasher);
            }
            FnPtr(polyfnsig) => {
                polyfnsig.hash_stable(__hcx, __hasher);
            }
            Dynamic(l, r) => {
                l.hash_stable(__hcx, __hasher);
                r.hash_stable(__hcx, __hasher);
            }
            Closure(def_id, substs) => {
                def_id.hash_stable(__hcx, __hasher);
                substs.hash_stable(__hcx, __hasher);
            }
            Generator(def_id, substs, m) => {
                def_id.hash_stable(__hcx, __hasher);
                substs.hash_stable(__hcx, __hasher);
                m.hash_stable(__hcx, __hasher);
            }
            GeneratorWitness(b) => {
                b.hash_stable(__hcx, __hasher);
            }
            Never => {}
            Tuple(substs) => {
                substs.hash_stable(__hcx, __hasher);
            }
            Projection(p) => {
                p.hash_stable(__hcx, __hasher);
            }
            Opaque(def_id, substs) => {
                def_id.hash_stable(__hcx, __hasher);
                substs.hash_stable(__hcx, __hasher);
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
/// be inferred to some other region from the diagram.  In the case of
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
/// and are substituted by an `InternalSubsts`, and late-bound, which are part of
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
{
}

// This is manually implemented because a derive would require `I: Clone`
impl<I: Interner> Clone for RegionKind<I> {
    fn clone(&self) -> Self {
        match self {
            ReEarlyBound(a) => ReEarlyBound(a.clone()),
            ReLateBound(a, b) => ReLateBound(a.clone(), b.clone()),
            ReFree(a) => ReFree(a.clone()),
            ReStatic => ReStatic,
            ReVar(a) => ReVar(a.clone()),
            RePlaceholder(a) => RePlaceholder(a.clone()),
            ReErased => ReErased,
        }
    }
}

// This is manually implemented because a derive would require `I: PartialEq`
impl<I: Interner> PartialEq for RegionKind<I> {
    #[inline]
    fn eq(&self, other: &RegionKind<I>) -> bool {
        let __self_vi = regionkind_discriminant(self);
        let __arg_1_vi = regionkind_discriminant(other);
        if __self_vi == __arg_1_vi {
            match (&*self, &*other) {
                (&ReEarlyBound(ref __self_0), &ReEarlyBound(ref __arg_1_0)) => {
                    __self_0 == __arg_1_0
                }
                (
                    &ReLateBound(ref __self_0, ref __self_1),
                    &ReLateBound(ref __arg_1_0, ref __arg_1_1),
                ) => __self_0 == __arg_1_0 && __self_1 == __arg_1_1,
                (&ReFree(ref __self_0), &ReFree(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (&ReStatic, &ReStatic) => true,
                (&ReVar(ref __self_0), &ReVar(ref __arg_1_0)) => __self_0 == __arg_1_0,
                (&RePlaceholder(ref __self_0), &RePlaceholder(ref __arg_1_0)) => {
                    __self_0 == __arg_1_0
                }
                (&ReErased, &ReErased) => true,
                _ => true,
            }
        } else {
            false
        }
    }
}

// This is manually implemented because a derive would require `I: Eq`
impl<I: Interner> Eq for RegionKind<I> {}

// This is manually implemented because a derive would require `I: PartialOrd`
impl<I: Interner> PartialOrd for RegionKind<I> {
    #[inline]
    fn partial_cmp(&self, other: &RegionKind<I>) -> Option<Ordering> {
        Some(Ord::cmp(self, other))
    }
}

// This is manually implemented because a derive would require `I: Ord`
impl<I: Interner> Ord for RegionKind<I> {
    #[inline]
    fn cmp(&self, other: &RegionKind<I>) -> Ordering {
        let __self_vi = regionkind_discriminant(self);
        let __arg_1_vi = regionkind_discriminant(other);
        if __self_vi == __arg_1_vi {
            match (&*self, &*other) {
                (&ReEarlyBound(ref __self_0), &ReEarlyBound(ref __arg_1_0)) => {
                    Ord::cmp(__self_0, __arg_1_0)
                }
                (
                    &ReLateBound(ref __self_0, ref __self_1),
                    &ReLateBound(ref __arg_1_0, ref __arg_1_1),
                ) => match Ord::cmp(__self_0, __arg_1_0) {
                    Ordering::Equal => Ord::cmp(__self_1, __arg_1_1),
                    cmp => cmp,
                },
                (&ReFree(ref __self_0), &ReFree(ref __arg_1_0)) => Ord::cmp(__self_0, __arg_1_0),
                (&ReStatic, &ReStatic) => Ordering::Equal,
                (&ReVar(ref __self_0), &ReVar(ref __arg_1_0)) => Ord::cmp(__self_0, __arg_1_0),
                (&RePlaceholder(ref __self_0), &RePlaceholder(ref __arg_1_0)) => {
                    Ord::cmp(__self_0, __arg_1_0)
                }
                (&ReErased, &ReErased) => Ordering::Equal,
                _ => Ordering::Equal,
            }
        } else {
            Ord::cmp(&__self_vi, &__arg_1_vi)
        }
    }
}

// This is manually implemented because a derive would require `I: Hash`
impl<I: Interner> hash::Hash for RegionKind<I> {
    fn hash<__H: hash::Hasher>(&self, state: &mut __H) -> () {
        match (&*self,) {
            (&ReEarlyBound(ref __self_0),) => {
                hash::Hash::hash(&regionkind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&ReLateBound(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&regionkind_discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&ReFree(ref __self_0),) => {
                hash::Hash::hash(&regionkind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&ReStatic,) => {
                hash::Hash::hash(&regionkind_discriminant(self), state);
            }
            (&ReVar(ref __self_0),) => {
                hash::Hash::hash(&regionkind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&RePlaceholder(ref __self_0),) => {
                hash::Hash::hash(&regionkind_discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&ReErased,) => {
                hash::Hash::hash(&regionkind_discriminant(self), state);
            }
        }
    }
}

// This is manually implemented because a derive would require `I: Debug`
impl<I: Interner> fmt::Debug for RegionKind<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReEarlyBound(ref data) => write!(f, "ReEarlyBound({:?})", data),

            ReLateBound(binder_id, ref bound_region) => {
                write!(f, "ReLateBound({:?}, {:?})", binder_id, bound_region)
            }

            ReFree(ref fr) => fr.fmt(f),

            ReStatic => write!(f, "ReStatic"),

            ReVar(ref vid) => vid.fmt(f),

            RePlaceholder(placeholder) => write!(f, "RePlaceholder({:?})", placeholder),

            ReErased => write!(f, "ReErased"),
        }
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
            ReErased | ReStatic => {
                // No variant fields to hash for these ...
            }
            ReLateBound(db, br) => {
                db.hash_stable(hcx, hasher);
                br.hash_stable(hcx, hasher);
            }
            ReEarlyBound(eb) => {
                eb.hash_stable(hcx, hasher);
            }
            ReFree(ref free_region) => {
                free_region.hash_stable(hcx, hasher);
            }
            RePlaceholder(p) => {
                p.hash_stable(hcx, hasher);
            }
            ReVar(reg) => {
                reg.hash_stable(hcx, hasher);
            }
        }
    }
}
