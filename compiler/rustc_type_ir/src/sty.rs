#![allow(rustc::usage_of_ty_tykind)]

use std::cmp::{Eq, Ord, Ordering, PartialEq, PartialOrd};
use std::{fmt, hash};

use crate::DebruijnIndex;
use crate::FloatTy;
use crate::IntTy;
use crate::Interner;
use crate::TyDecoder;
use crate::TyEncoder;
use crate::UintTy;

use self::TyKind::*;

use rustc_data_structures::stable_hasher::HashStable;
use rustc_serialize::{Decodable, Encodable};

/// Defines the kinds of types used by the type system.
///
/// Types written by the user start out as `hir::TyKind` and get
/// converted to this representation using `AstConv::ast_ty_to_ty`.
///
/// The `HashStable` implementation for this type is defined in `rustc_query_system::ich`.
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
const fn discriminant<I: Interner>(value: &TyKind<I>) -> usize {
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
        let __self_vi = discriminant(self);
        let __arg_1_vi = discriminant(other);
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
        let __self_vi = discriminant(self);
        let __arg_1_vi = discriminant(other);
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
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Uint(ref __self_0),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Float(ref __self_0),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Adt(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&Foreign(ref __self_0),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Array(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&Slice(ref __self_0),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&RawPtr(ref __self_0),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Ref(ref __self_0, ref __self_1, ref __self_2),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state);
                hash::Hash::hash(__self_2, state)
            }
            (&FnDef(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&FnPtr(ref __self_0),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Dynamic(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&Closure(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&Generator(ref __self_0, ref __self_1, ref __self_2),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state);
                hash::Hash::hash(__self_2, state)
            }
            (&GeneratorWitness(ref __self_0),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Tuple(ref __self_0),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Projection(ref __self_0),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Opaque(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&Param(ref __self_0),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Bound(ref __self_0, ref __self_1),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state);
                hash::Hash::hash(__self_1, state)
            }
            (&Placeholder(ref __self_0),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Infer(ref __self_0),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            (&Error(ref __self_0),) => {
                hash::Hash::hash(&discriminant(self), state);
                hash::Hash::hash(__self_0, state)
            }
            _ => hash::Hash::hash(&discriminant(self), state),
        }
    }
}

// This is manually implemented because a derive would require `I: Debug`
impl<I: Interner> fmt::Debug for TyKind<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (&*self,) {
            (&Bool,) => fmt::Formatter::write_str(f, "Bool"),
            (&Char,) => fmt::Formatter::write_str(f, "Char"),
            (&Int(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Int");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Uint(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Uint");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Float(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Float");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Adt(ref __self_0, ref __self_1),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Adt");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_1);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Foreign(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Foreign");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Str,) => fmt::Formatter::write_str(f, "Str"),
            (&Array(ref __self_0, ref __self_1),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Array");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_1);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Slice(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Slice");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&RawPtr(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "RawPtr");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Ref(ref __self_0, ref __self_1, ref __self_2),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Ref");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_1);
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_2);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&FnDef(ref __self_0, ref __self_1),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "FnDef");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_1);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&FnPtr(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "FnPtr");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Dynamic(ref __self_0, ref __self_1),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Dynamic");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_1);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Closure(ref __self_0, ref __self_1),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Closure");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_1);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Generator(ref __self_0, ref __self_1, ref __self_2),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Generator");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_1);
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_2);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&GeneratorWitness(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "GeneratorWitness");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Never,) => fmt::Formatter::write_str(f, "Never"),
            (&Tuple(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Tuple");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Projection(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Projection");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Opaque(ref __self_0, ref __self_1),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Opaque");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_1);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Param(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Param");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Bound(ref __self_0, ref __self_1),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Bound");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_1);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Placeholder(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Placeholder");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Infer(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Infer");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
            (&Error(ref __self_0),) => {
                let debug_trait_builder = &mut fmt::Formatter::debug_tuple(f, "Error");
                let _ = fmt::DebugTuple::field(debug_trait_builder, &__self_0);
                fmt::DebugTuple::finish(debug_trait_builder)
            }
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
    fn encode(&self, e: &mut E) -> Result<(), <E as rustc_serialize::Encoder>::Error> {
        rustc_serialize::Encoder::emit_enum(e, |e| {
            let disc = discriminant(self);
            match self {
                Bool => e.emit_enum_variant("Bool", disc, 0, |_| Ok(())),
                Char => e.emit_enum_variant("Char", disc, 0, |_| Ok(())),
                Int(i) => e.emit_enum_variant("Int", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| i.encode(e))?;
                    Ok(())
                }),
                Uint(u) => e.emit_enum_variant("Uint", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| u.encode(e))?;
                    Ok(())
                }),
                Float(f) => e.emit_enum_variant("Float", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| f.encode(e))?;
                    Ok(())
                }),
                Adt(adt, substs) => e.emit_enum_variant("Adt", disc, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| adt.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| substs.encode(e))?;
                    Ok(())
                }),
                Foreign(def_id) => e.emit_enum_variant("Foreign", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| def_id.encode(e))?;
                    Ok(())
                }),
                Str => e.emit_enum_variant("Str", disc, 0, |_| Ok(())),
                Array(t, c) => e.emit_enum_variant("Array", disc, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| t.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| c.encode(e))?;
                    Ok(())
                }),
                Slice(t) => e.emit_enum_variant("Slice", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| t.encode(e))?;
                    Ok(())
                }),
                RawPtr(tam) => e.emit_enum_variant("RawPtr", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| tam.encode(e))?;
                    Ok(())
                }),
                Ref(r, t, m) => e.emit_enum_variant("Ref", disc, 3, |e| {
                    e.emit_enum_variant_arg(true, |e| r.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| t.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| m.encode(e))?;
                    Ok(())
                }),
                FnDef(def_id, substs) => e.emit_enum_variant("FnDef", disc, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| def_id.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| substs.encode(e))?;
                    Ok(())
                }),
                FnPtr(polyfnsig) => e.emit_enum_variant("FnPtr", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| polyfnsig.encode(e))?;
                    Ok(())
                }),
                Dynamic(l, r) => e.emit_enum_variant("Dynamic", disc, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| l.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| r.encode(e))?;
                    Ok(())
                }),
                Closure(def_id, substs) => e.emit_enum_variant("Closure", disc, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| def_id.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| substs.encode(e))?;
                    Ok(())
                }),
                Generator(def_id, substs, m) => e.emit_enum_variant("Generator", disc, 3, |e| {
                    e.emit_enum_variant_arg(true, |e| def_id.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| substs.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| m.encode(e))?;
                    Ok(())
                }),
                GeneratorWitness(b) => e.emit_enum_variant("GeneratorWitness", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| b.encode(e))?;
                    Ok(())
                }),
                Never => e.emit_enum_variant("Never", disc, 0, |_| Ok(())),
                Tuple(substs) => e.emit_enum_variant("Tuple", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| substs.encode(e))?;
                    Ok(())
                }),
                Projection(p) => e.emit_enum_variant("Projection", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| p.encode(e))?;
                    Ok(())
                }),
                Opaque(def_id, substs) => e.emit_enum_variant("Opaque", disc, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| def_id.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| substs.encode(e))?;
                    Ok(())
                }),
                Param(p) => e.emit_enum_variant("Param", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| p.encode(e))?;
                    Ok(())
                }),
                Bound(d, b) => e.emit_enum_variant("Bound", disc, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| d.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| b.encode(e))?;
                    Ok(())
                }),
                Placeholder(p) => e.emit_enum_variant("Placeholder", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| p.encode(e))?;
                    Ok(())
                }),
                Infer(i) => e.emit_enum_variant("Infer", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| i.encode(e))?;
                    Ok(())
                }),
                Error(d) => e.emit_enum_variant("Error", disc, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| d.encode(e))?;
                    Ok(())
                }),
            }
        })
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
        match rustc_serialize::Decoder::read_usize(d) {
            0 => Bool,
            1 => Char,
            2 => Int(rustc_serialize::Decodable::decode(d)),
            3 => Uint(rustc_serialize::Decodable::decode(d)),
            4 => Float(rustc_serialize::Decodable::decode(d)),
            5 => Adt(rustc_serialize::Decodable::decode(d), rustc_serialize::Decodable::decode(d)),
            6 => Foreign(rustc_serialize::Decodable::decode(d)),
            7 => Str,
            8 => {
                Array(rustc_serialize::Decodable::decode(d), rustc_serialize::Decodable::decode(d))
            }
            9 => Slice(rustc_serialize::Decodable::decode(d)),
            10 => RawPtr(rustc_serialize::Decodable::decode(d)),
            11 => Ref(
                rustc_serialize::Decodable::decode(d),
                rustc_serialize::Decodable::decode(d),
                rustc_serialize::Decodable::decode(d),
            ),
            12 => {
                FnDef(rustc_serialize::Decodable::decode(d), rustc_serialize::Decodable::decode(d))
            }
            13 => FnPtr(rustc_serialize::Decodable::decode(d)),
            14 => Dynamic(
                rustc_serialize::Decodable::decode(d),
                rustc_serialize::Decodable::decode(d),
            ),
            15 => Closure(
                rustc_serialize::Decodable::decode(d),
                rustc_serialize::Decodable::decode(d),
            ),
            16 => Generator(
                rustc_serialize::Decodable::decode(d),
                rustc_serialize::Decodable::decode(d),
                rustc_serialize::Decodable::decode(d),
            ),
            17 => GeneratorWitness(rustc_serialize::Decodable::decode(d)),
            18 => Never,
            19 => Tuple(rustc_serialize::Decodable::decode(d)),
            20 => Projection(rustc_serialize::Decodable::decode(d)),
            21 => {
                Opaque(rustc_serialize::Decodable::decode(d), rustc_serialize::Decodable::decode(d))
            }
            22 => Param(rustc_serialize::Decodable::decode(d)),
            23 => {
                Bound(rustc_serialize::Decodable::decode(d), rustc_serialize::Decodable::decode(d))
            }
            24 => Placeholder(rustc_serialize::Decodable::decode(d)),
            25 => Infer(rustc_serialize::Decodable::decode(d)),
            26 => Error(rustc_serialize::Decodable::decode(d)),
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
impl<CTX, I: Interner> HashStable<CTX> for TyKind<I>
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
