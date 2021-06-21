use crate::DebruijnIndex;
use crate::FloatTy;
use crate::IntTy;
use crate::UintTy;
use crate::Interner;
use crate::TyDecoder;
use crate::TyEncoder;

use rustc_serialize::{Decodable, Encodable};

/// Defines the kinds of types.
///
/// N.B., if you change this, you'll probably want to change the corresponding
/// AST structure in `rustc_ast/src/ast.rs` as well.
#[allow(rustc::usage_of_ty_tykind)]
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
//#[derive(TyEncodable, TyDecodable)]
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
    /// InternalSubsts here, possibly against intuition, *may* contain `Param`s.
    /// That is, even after substitution it is possible that there are type
    /// variables. This happens when the `Adt` corresponds to an ADT
    /// definition and not a concrete use of it.
    Adt(I::AdtDef, I::SubstsRef),

    /// An unsized FFI type that is opaque to Rust. Written as `extern type T`.
    Foreign(I::DefId),

    /// The pointee of a string slice. Written as `str`.
    Str,

    /// An array with the given length. Written as `[T; n]`.
    Array(I::Ty, I::Const),

    /// The pointee of an array slice. Written as `[T]`.
    Slice(I::Ty),

    /// A raw pointer. Written as `*mut T` or `*const T`
    RawPtr(I::TypeAndMut),

    /// A reference; a pointer with an associated lifetime. Written as
    /// `&'a mut T` or `&'a T`.
    Ref(I::Region, I::Ty, I::Mutability),

    /// The anonymous type of a function declaration/definition. Each
    /// function has a unique type, which is output (for a function
    /// named `foo` returning an `i32`) as `fn() -> i32 {foo}`.
    ///
    /// For example the type of `bar` here:
    ///
    /// ```rust
    /// fn foo() -> i32 { 1 }
    /// let bar = foo; // bar: fn() -> i32 {foo}
    /// ```
    FnDef(I::DefId, I::SubstsRef),

    /// A pointer to a function. Written as `fn() -> i32`.
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

    /// The anonymous type of a closure. Used to represent the type of
    /// `|a| a`.
    Closure(I::DefId, I::SubstsRef),

    /// The anonymous type of a generator. Used to represent the type of
    /// `|a| yield a`.
    Generator(I::DefId, I::SubstsRef, I::Movability),

    /// A type representing the types stored inside a generator.
    /// This should only appear in GeneratorInteriors.
    GeneratorWitness(I::BinderListTy),

    /// The never type `!`.
    Never,

    /// A tuple type. For example, `(i32, bool)`.
    /// Use `TyS::tuple_fields` to iterate over the field types.
    Tuple(I::SubstsRef),

    /// The projection of an associated type. For example,
    /// `<T as Trait<..>>::N`.
    Projection(I::ProjectionTy),

    /// Opaque (`impl Trait`) type found in a return type.
    /// The `DefId` comes either from
    /// * the `impl Trait` ast::Ty node,
    /// * or the `type Foo = impl Trait` declaration
    /// The substitutions are for the generics of the function in question.
    /// After typeck, the concrete type can be found in the `types` map.
    Opaque(I::DefId, I::SubstsRef),

    /// A type parameter; for example, `T` in `fn f<T>(x: T) {}`.
    Param(I::ParamTy),

    /// Bound type variable, used only when preparing a trait query.
    Bound(DebruijnIndex, I::BoundTy),

    /// A placeholder type - universally quantified higher-ranked type.
    Placeholder(I::PlaceholderType),

    /// A type variable used during type checking.
    Infer(I::InferTy),

    /// A placeholder for a type which could not be computed; this is
    /// propagated to avoid useless error messages.
    Error(I::DelaySpanBugEmitted),
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
        __I::ProjectionTy: Decodable<__D>,
        __I::ParamTy: Decodable<__D>,
        __I::BoundTy: Decodable<__D>,
        __I::PlaceholderType: Decodable<__D>,
        __I::InferTy: Decodable<__D>,
        __I::DelaySpanBugEmitted: Decodable<__D>,
        __I::PredicateKind: Decodable<__D>,
        __I::AllocId: Decodable<__D>,
{
    fn decode(
        __decoder: &mut __D,
    ) -> Result<Self, <__D as rustc_serialize::Decoder>::Error> {
        __decoder.read_enum(
            |__decoder| {
                __decoder.read_enum_variant(
                    &[
                        "Bool",
                        "Char",
                        "Int",
                        "Uint",
                        "Float",
                        "Adt",
                        "Foreign",
                        "Str",
                        "Array",
                        "Slice",
                        "RawPtr",
                        "Ref",
                        "FnDef",
                        "FnPtr",
                        "Dynamic",
                        "Closure",
                        "Generator",
                        "GeneratorWitness",
                        "Never",
                        "Tuple",
                        "Projection",
                        "Opaque",
                        "Param",
                        "Bound",
                        "Placeholder",
                        "Infer",
                        "Error",
                    ],
                    |__decoder, __variant_idx| {
                        use TyKind::*;
                        Ok(match __variant_idx {
                            0 => Bool,
                            1 => Char,
                            2 => Int(__decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?),
                            3 => Uint(__decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?),
                            4 => Float(__decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?),
                            5 => Adt(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            6 => Foreign(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            7 => Str,
                            8 => Array(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            9 => Slice(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            10 => RawPtr(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            11 => Ref(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            12 => FnDef(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            13 => FnPtr(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            14 => Dynamic(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            15 => Closure(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            16 => Generator(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            17 => GeneratorWitness(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            18 => Never,
                            19 => Tuple(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            20 => Projection(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            21 => Opaque(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            22 => Param(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            23 => Bound(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            24 => Placeholder(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            25 => Infer(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            26 => Error(
                                __decoder.read_enum_variant_arg(rustc_serialize::Decodable::decode)?,
                            ),
                            _ => return Err(rustc_serialize::Decoder::error(__decoder, &format!(
                                "invalid enum variant tag while decoding `{}`, expected 0..{}",
                                "TyKind",
                                27,
                            ))),
                        })
                    })
            }
        )
    }
}
