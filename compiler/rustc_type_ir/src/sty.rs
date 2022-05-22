use crate::DebruijnIndex;
use crate::FloatTy;
use crate::IntTy;
use crate::Interner;
use crate::TyDecoder;
use crate::TyEncoder;
use crate::UintTy;

use rustc_serialize::{Decodable, Encodable};

/// Defines the kinds of types used by the type system.
///
/// Types written by the user start out as [hir::TyKind](rustc_hir::TyKind) and get
/// converted to this representation using `AstConv::ast_ty_to_ty`.
#[allow(rustc::usage_of_ty_tykind)]
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
//#[derive(TyEncodable, TyDecodable)]
//#[derive(HashStable)]
#[rustc_diagnostic_item = "TyKind"]
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
    /// [ClosureSubsts] for more details.
    Closure(I::DefId, I::SubstsRef),

    /// The anonymous type of a generator. Used to represent the type of
    /// `|a| yield a`.
    ///
    /// For more info about generator substs, visit the documentation for
    /// [GeneratorSubsts].
    Generator(I::DefId, I::SubstsRef, I::Movability),

    /// A type representing the types stored inside a generator.
    /// This should only appear as part of the [GeneratorSubsts].
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

#[allow(rustc::usage_of_ty_tykind)]
impl<I: Interner> Clone for TyKind<I> {
    fn clone(&self) -> Self {
        use crate::TyKind::*;
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

#[allow(rustc::usage_of_ty_tykind)]
impl<I: Interner> TyKind<I> {
    #[inline]
    pub fn is_primitive(&self) -> bool {
        use crate::TyKind::*;
        matches!(self, Bool | Char | Int(_) | Uint(_) | Float(_))
    }
}

#[allow(rustc::usage_of_ty_tykind)]
impl<__I: Interner, __E: TyEncoder> Encodable<__E> for TyKind<__I>
where
    __I::DelaySpanBugEmitted: Encodable<__E>,
    __I::AdtDef: Encodable<__E>,
    __I::SubstsRef: Encodable<__E>,
    __I::DefId: Encodable<__E>,
    __I::Ty: Encodable<__E>,
    __I::Const: Encodable<__E>,
    __I::Region: Encodable<__E>,
    __I::TypeAndMut: Encodable<__E>,
    __I::Mutability: Encodable<__E>,
    __I::Movability: Encodable<__E>,
    __I::PolyFnSig: Encodable<__E>,
    __I::ListBinderExistentialPredicate: Encodable<__E>,
    __I::BinderListTy: Encodable<__E>,
    __I::ListTy: Encodable<__E>,
    __I::ProjectionTy: Encodable<__E>,
    __I::ParamTy: Encodable<__E>,
    __I::BoundTy: Encodable<__E>,
    __I::PlaceholderType: Encodable<__E>,
    __I::InferTy: Encodable<__E>,
    __I::DelaySpanBugEmitted: Encodable<__E>,
    __I::PredicateKind: Encodable<__E>,
    __I::AllocId: Encodable<__E>,
{
    fn encode(&self, e: &mut __E) -> Result<(), <__E as rustc_serialize::Encoder>::Error> {
        rustc_serialize::Encoder::emit_enum(e, |e| {
            use rustc_type_ir::TyKind::*;
            match self {
                Bool => e.emit_enum_variant("Bool", 0, 0, |_| Ok(())),
                Char => e.emit_enum_variant("Char", 1, 0, |_| Ok(())),
                Int(i) => e.emit_enum_variant("Int", 2, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| i.encode(e))?;
                    Ok(())
                }),
                Uint(u) => e.emit_enum_variant("Uint", 3, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| u.encode(e))?;
                    Ok(())
                }),
                Float(f) => e.emit_enum_variant("Float", 4, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| f.encode(e))?;
                    Ok(())
                }),
                Adt(adt, substs) => e.emit_enum_variant("Adt", 5, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| adt.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| substs.encode(e))?;
                    Ok(())
                }),
                Foreign(def_id) => e.emit_enum_variant("Foreign", 6, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| def_id.encode(e))?;
                    Ok(())
                }),
                Str => e.emit_enum_variant("Str", 7, 0, |_| Ok(())),
                Array(t, c) => e.emit_enum_variant("Array", 8, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| t.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| c.encode(e))?;
                    Ok(())
                }),
                Slice(t) => e.emit_enum_variant("Slice", 9, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| t.encode(e))?;
                    Ok(())
                }),
                RawPtr(tam) => e.emit_enum_variant("RawPtr", 10, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| tam.encode(e))?;
                    Ok(())
                }),
                Ref(r, t, m) => e.emit_enum_variant("Ref", 11, 3, |e| {
                    e.emit_enum_variant_arg(true, |e| r.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| t.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| m.encode(e))?;
                    Ok(())
                }),
                FnDef(def_id, substs) => e.emit_enum_variant("FnDef", 12, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| def_id.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| substs.encode(e))?;
                    Ok(())
                }),
                FnPtr(polyfnsig) => e.emit_enum_variant("FnPtr", 13, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| polyfnsig.encode(e))?;
                    Ok(())
                }),
                Dynamic(l, r) => e.emit_enum_variant("Dynamic", 14, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| l.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| r.encode(e))?;
                    Ok(())
                }),
                Closure(def_id, substs) => e.emit_enum_variant("Closure", 15, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| def_id.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| substs.encode(e))?;
                    Ok(())
                }),
                Generator(def_id, substs, m) => e.emit_enum_variant("Generator", 16, 3, |e| {
                    e.emit_enum_variant_arg(true, |e| def_id.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| substs.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| m.encode(e))?;
                    Ok(())
                }),
                GeneratorWitness(b) => e.emit_enum_variant("GeneratorWitness", 17, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| b.encode(e))?;
                    Ok(())
                }),
                Never => e.emit_enum_variant("Never", 18, 0, |_| Ok(())),
                Tuple(substs) => e.emit_enum_variant("Tuple", 19, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| substs.encode(e))?;
                    Ok(())
                }),
                Projection(p) => e.emit_enum_variant("Projection", 20, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| p.encode(e))?;
                    Ok(())
                }),
                Opaque(def_id, substs) => e.emit_enum_variant("Opaque", 21, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| def_id.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| substs.encode(e))?;
                    Ok(())
                }),
                Param(p) => e.emit_enum_variant("Param", 22, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| p.encode(e))?;
                    Ok(())
                }),
                Bound(d, b) => e.emit_enum_variant("Bound", 23, 2, |e| {
                    e.emit_enum_variant_arg(true, |e| d.encode(e))?;
                    e.emit_enum_variant_arg(false, |e| b.encode(e))?;
                    Ok(())
                }),
                Placeholder(p) => e.emit_enum_variant("Placeholder", 24, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| p.encode(e))?;
                    Ok(())
                }),
                Infer(i) => e.emit_enum_variant("Infer", 25, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| i.encode(e))?;
                    Ok(())
                }),
                Error(d) => e.emit_enum_variant("Error", 26, 1, |e| {
                    e.emit_enum_variant_arg(true, |e| d.encode(e))?;
                    Ok(())
                }),
            }
        })
    }
}

#[allow(rustc::usage_of_ty_tykind)]
impl<__I: Interner, __D: TyDecoder<I = __I>> Decodable<__D> for TyKind<__I>
where
    __I::DelaySpanBugEmitted: Decodable<__D>,
    __I::AdtDef: Decodable<__D>,
    __I::SubstsRef: Decodable<__D>,
    __I::DefId: Decodable<__D>,
    __I::Ty: Decodable<__D>,
    __I::Const: Decodable<__D>,
    __I::Region: Decodable<__D>,
    __I::TypeAndMut: Decodable<__D>,
    __I::Mutability: Decodable<__D>,
    __I::Movability: Decodable<__D>,
    __I::PolyFnSig: Decodable<__D>,
    __I::ListBinderExistentialPredicate: Decodable<__D>,
    __I::BinderListTy: Decodable<__D>,
    __I::ListTy: Decodable<__D>,
    __I::ProjectionTy: Decodable<__D>,
    __I::ParamTy: Decodable<__D>,
    __I::BoundTy: Decodable<__D>,
    __I::PlaceholderType: Decodable<__D>,
    __I::InferTy: Decodable<__D>,
    __I::DelaySpanBugEmitted: Decodable<__D>,
    __I::PredicateKind: Decodable<__D>,
    __I::AllocId: Decodable<__D>,
{
    fn decode(__decoder: &mut __D) -> Self {
        use TyKind::*;

        match rustc_serialize::Decoder::read_usize(__decoder) {
            0 => Bool,
            1 => Char,
            2 => Int(rustc_serialize::Decodable::decode(__decoder)),
            3 => Uint(rustc_serialize::Decodable::decode(__decoder)),
            4 => Float(rustc_serialize::Decodable::decode(__decoder)),
            5 => Adt(
                rustc_serialize::Decodable::decode(__decoder),
                rustc_serialize::Decodable::decode(__decoder),
            ),
            6 => Foreign(rustc_serialize::Decodable::decode(__decoder)),
            7 => Str,
            8 => Array(
                rustc_serialize::Decodable::decode(__decoder),
                rustc_serialize::Decodable::decode(__decoder),
            ),
            9 => Slice(rustc_serialize::Decodable::decode(__decoder)),
            10 => RawPtr(rustc_serialize::Decodable::decode(__decoder)),
            11 => Ref(
                rustc_serialize::Decodable::decode(__decoder),
                rustc_serialize::Decodable::decode(__decoder),
                rustc_serialize::Decodable::decode(__decoder),
            ),
            12 => FnDef(
                rustc_serialize::Decodable::decode(__decoder),
                rustc_serialize::Decodable::decode(__decoder),
            ),
            13 => FnPtr(rustc_serialize::Decodable::decode(__decoder)),
            14 => Dynamic(
                rustc_serialize::Decodable::decode(__decoder),
                rustc_serialize::Decodable::decode(__decoder),
            ),
            15 => Closure(
                rustc_serialize::Decodable::decode(__decoder),
                rustc_serialize::Decodable::decode(__decoder),
            ),
            16 => Generator(
                rustc_serialize::Decodable::decode(__decoder),
                rustc_serialize::Decodable::decode(__decoder),
                rustc_serialize::Decodable::decode(__decoder),
            ),
            17 => GeneratorWitness(rustc_serialize::Decodable::decode(__decoder)),
            18 => Never,
            19 => Tuple(rustc_serialize::Decodable::decode(__decoder)),
            20 => Projection(rustc_serialize::Decodable::decode(__decoder)),
            21 => Opaque(
                rustc_serialize::Decodable::decode(__decoder),
                rustc_serialize::Decodable::decode(__decoder),
            ),
            22 => Param(rustc_serialize::Decodable::decode(__decoder)),
            23 => Bound(
                rustc_serialize::Decodable::decode(__decoder),
                rustc_serialize::Decodable::decode(__decoder),
            ),
            24 => Placeholder(rustc_serialize::Decodable::decode(__decoder)),
            25 => Infer(rustc_serialize::Decodable::decode(__decoder)),
            26 => Error(rustc_serialize::Decodable::decode(__decoder)),
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
