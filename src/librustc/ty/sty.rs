// Copyright 2012-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module contains TypeVariants and its major components

use hir::def_id::DefId;

use middle::const_val::ConstVal;
use middle::region;
use rustc_data_structures::indexed_vec::Idx;
use ty::subst::{Substs, Subst};
use ty::{self, AdtDef, TypeFlags, Ty, TyCtxt, TypeFoldable};
use ty::{Slice, TyS};
use ty::subst::Kind;

use std::iter;
use std::cmp::Ordering;
use syntax::abi;
use syntax::ast::{self, Name};
use syntax::symbol::keywords;

use serialize;

use hir;

use self::InferTy::*;
use self::TypeVariants::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct TypeAndMut<'tcx> {
    pub ty: Ty<'tcx>,
    pub mutbl: hir::Mutability,
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash,
         RustcEncodable, RustcDecodable, Copy)]
/// A "free" region `fr` can be interpreted as "some region
/// at least as big as the scope `fr.scope`".
pub struct FreeRegion {
    pub scope: DefId,
    pub bound_region: BoundRegion,
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash,
         RustcEncodable, RustcDecodable, Copy)]
pub enum BoundRegion {
    /// An anonymous region parameter for a given fn (&T)
    BrAnon(u32),

    /// Named region parameters for functions (a in &'a T)
    ///
    /// The def-id is needed to distinguish free regions in
    /// the event of shadowing.
    BrNamed(DefId, Name),

    /// Fresh bound identifiers created during GLB computations.
    BrFresh(u32),

    /// Anonymous region for the implicit env pointer parameter
    /// to a closure
    BrEnv,
}

impl BoundRegion {
    pub fn is_named(&self) -> bool {
        match *self {
            BoundRegion::BrNamed(..) => true,
            _ => false,
        }
    }
}

/// NB: If you change this, you'll probably want to change the corresponding
/// AST structure in libsyntax/ast.rs as well.
#[derive(Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub enum TypeVariants<'tcx> {
    /// The primitive boolean type. Written as `bool`.
    TyBool,

    /// The primitive character type; holds a Unicode scalar value
    /// (a non-surrogate code point).  Written as `char`.
    TyChar,

    /// A primitive signed integer type. For example, `i32`.
    TyInt(ast::IntTy),

    /// A primitive unsigned integer type. For example, `u32`.
    TyUint(ast::UintTy),

    /// A primitive floating-point type. For example, `f64`.
    TyFloat(ast::FloatTy),

    /// Structures, enumerations and unions.
    ///
    /// Substs here, possibly against intuition, *may* contain `TyParam`s.
    /// That is, even after substitution it is possible that there are type
    /// variables. This happens when the `TyAdt` corresponds to an ADT
    /// definition and not a concrete use of it.
    TyAdt(&'tcx AdtDef, &'tcx Substs<'tcx>),

    TyForeign(DefId),

    /// The pointee of a string slice. Written as `str`.
    TyStr,

    /// An array with the given length. Written as `[T; n]`.
    TyArray(Ty<'tcx>, &'tcx ty::Const<'tcx>),

    /// The pointee of an array slice.  Written as `[T]`.
    TySlice(Ty<'tcx>),

    /// A raw pointer. Written as `*mut T` or `*const T`
    TyRawPtr(TypeAndMut<'tcx>),

    /// A reference; a pointer with an associated lifetime. Written as
    /// `&'a mut T` or `&'a T`.
    TyRef(Region<'tcx>, TypeAndMut<'tcx>),

    /// The anonymous type of a function declaration/definition. Each
    /// function has a unique type.
    TyFnDef(DefId, &'tcx Substs<'tcx>),

    /// A pointer to a function.  Written as `fn() -> i32`.
    TyFnPtr(PolyFnSig<'tcx>),

    /// A trait, defined with `trait`.
    TyDynamic(Binder<&'tcx Slice<ExistentialPredicate<'tcx>>>, ty::Region<'tcx>),

    /// The anonymous type of a closure. Used to represent the type of
    /// `|a| a`.
    TyClosure(DefId, ClosureSubsts<'tcx>),

    /// The anonymous type of a generator. Used to represent the type of
    /// `|a| yield a`.
    TyGenerator(DefId, ClosureSubsts<'tcx>, GeneratorInterior<'tcx>),

    /// The never type `!`
    TyNever,

    /// A tuple type.  For example, `(i32, bool)`.
    /// The bool indicates whether this is a unit tuple and was created by
    /// defaulting a diverging type variable with feature(never_type) disabled.
    /// It's only purpose is for raising future-compatibility warnings for when
    /// diverging type variables start defaulting to ! instead of ().
    TyTuple(&'tcx Slice<Ty<'tcx>>, bool),

    /// The projection of an associated type.  For example,
    /// `<T as Trait<..>>::N`.
    TyProjection(ProjectionTy<'tcx>),

    /// Anonymized (`impl Trait`) type found in a return type.
    /// The DefId comes from the `impl Trait` ast::Ty node, and the
    /// substitutions are for the generics of the function in question.
    /// After typeck, the concrete type can be found in the `types` map.
    TyAnon(DefId, &'tcx Substs<'tcx>),

    /// A type parameter; for example, `T` in `fn f<T>(x: T) {}
    TyParam(ParamTy),

    /// A type variable used during type-checking.
    TyInfer(InferTy),

    /// A placeholder for a type which could not be computed; this is
    /// propagated to avoid useless error messages.
    TyError,
}

/// A closure can be modeled as a struct that looks like:
///
///     struct Closure<'l0...'li, T0...Tj, CK, CS, U0...Uk> {
///         upvar0: U0,
///         ...
///         upvark: Uk
///     }
///
/// where:
///
/// - 'l0...'li and T0...Tj are the lifetime and type parameters
///   in scope on the function that defined the closure,
/// - CK represents the *closure kind* (Fn vs FnMut vs FnOnce). This
///   is rather hackily encoded via a scalar type. See
///   `TyS::to_opt_closure_kind` for details.
/// - CS represents the *closure signature*, representing as a `fn()`
///   type. For example, `fn(u32, u32) -> u32` would mean that the closure
///   implements `CK<(u32, u32), Output = u32>`, where `CK` is the trait
///   specified above.
/// - U0...Uk are type parameters representing the types of its upvars
///   (borrowed, if appropriate; that is, if Ui represents a by-ref upvar,
///    and the up-var has the type `Foo`, then `Ui = &Foo`).
///
/// So, for example, given this function:
///
///     fn foo<'a, T>(data: &'a mut T) {
///          do(|| data.count += 1)
///     }
///
/// the type of the closure would be something like:
///
///     struct Closure<'a, T, U0> {
///         data: U0
///     }
///
/// Note that the type of the upvar is not specified in the struct.
/// You may wonder how the impl would then be able to use the upvar,
/// if it doesn't know it's type? The answer is that the impl is
/// (conceptually) not fully generic over Closure but rather tied to
/// instances with the expected upvar types:
///
///     impl<'b, 'a, T> FnMut() for Closure<'a, T, &'b mut &'a mut T> {
///         ...
///     }
///
/// You can see that the *impl* fully specified the type of the upvar
/// and thus knows full well that `data` has type `&'b mut &'a mut T`.
/// (Here, I am assuming that `data` is mut-borrowed.)
///
/// Now, the last question you may ask is: Why include the upvar types
/// as extra type parameters? The reason for this design is that the
/// upvar types can reference lifetimes that are internal to the
/// creating function. In my example above, for example, the lifetime
/// `'b` represents the scope of the closure itself; this is some
/// subset of `foo`, probably just the scope of the call to the to
/// `do()`. If we just had the lifetime/type parameters from the
/// enclosing function, we couldn't name this lifetime `'b`. Note that
/// there can also be lifetimes in the types of the upvars themselves,
/// if one of them happens to be a reference to something that the
/// creating fn owns.
///
/// OK, you say, so why not create a more minimal set of parameters
/// that just includes the extra lifetime parameters? The answer is
/// primarily that it would be hard --- we don't know at the time when
/// we create the closure type what the full types of the upvars are,
/// nor do we know which are borrowed and which are not. In this
/// design, we can just supply a fresh type parameter and figure that
/// out later.
///
/// All right, you say, but why include the type parameters from the
/// original function then? The answer is that trans may need them
/// when monomorphizing, and they may not appear in the upvars.  A
/// closure could capture no variables but still make use of some
/// in-scope type parameter with a bound (e.g., if our example above
/// had an extra `U: Default`, and the closure called `U::default()`).
///
/// There is another reason. This design (implicitly) prohibits
/// closures from capturing themselves (except via a trait
/// object). This simplifies closure inference considerably, since it
/// means that when we infer the kind of a closure or its upvars, we
/// don't have to handle cycles where the decisions we make for
/// closure C wind up influencing the decisions we ought to make for
/// closure C (which would then require fixed point iteration to
/// handle). Plus it fixes an ICE. :P
///
/// ## Generators
///
/// Perhaps surprisingly, `ClosureSubsts` are also used for
/// generators.  In that case, what is written above is only half-true
/// -- the set of type parameters is similar, but the role of CK and
/// CS are different.  CK represents the "yield type" and CS
/// represents the "return type" of the generator.
///
/// It'd be nice to split this struct into ClosureSubsts and
/// GeneratorSubsts, I believe. -nmatsakis
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct ClosureSubsts<'tcx> {
    /// Lifetime and type parameters from the enclosing function,
    /// concatenated with the types of the upvars.
    ///
    /// These are separated out because trans wants to pass them around
    /// when monomorphizing.
    pub substs: &'tcx Substs<'tcx>,
}

/// Struct returned by `split()`. Note that these are subslices of the
/// parent slice and not canonical substs themselves.
struct SplitClosureSubsts<'tcx> {
    closure_kind_ty: Ty<'tcx>,
    closure_sig_ty: Ty<'tcx>,
    upvar_kinds: &'tcx [Kind<'tcx>],
}

impl<'tcx> ClosureSubsts<'tcx> {
    /// Divides the closure substs into their respective
    /// components. Single source of truth with respect to the
    /// ordering.
    fn split(self, def_id: DefId, tcx: TyCtxt<'_, '_, '_>) -> SplitClosureSubsts<'tcx> {
        let generics = tcx.generics_of(def_id);
        let parent_len = generics.parent_count();
        SplitClosureSubsts {
            closure_kind_ty: self.substs[parent_len].as_type().expect("CK should be a type"),
            closure_sig_ty: self.substs[parent_len + 1].as_type().expect("CS should be a type"),
            upvar_kinds: &self.substs[parent_len + 2..],
        }
    }

    #[inline]
    pub fn upvar_tys(self, def_id: DefId, tcx: TyCtxt<'_, '_, '_>) ->
        impl Iterator<Item=Ty<'tcx>> + 'tcx
    {
        let SplitClosureSubsts { upvar_kinds, .. } = self.split(def_id, tcx);
        upvar_kinds.iter().map(|t| t.as_type().expect("upvar should be type"))
    }

    /// Returns the closure kind for this closure; may return a type
    /// variable during inference. To get the closure kind during
    /// inference, use `infcx.closure_kind(def_id, substs)`.
    pub fn closure_kind_ty(self, def_id: DefId, tcx: TyCtxt<'_, '_, '_>) -> Ty<'tcx> {
        self.split(def_id, tcx).closure_kind_ty
    }

    /// Returns the type representing the closure signature for this
    /// closure; may contain type variables during inference. To get
    /// the closure signature during inference, use
    /// `infcx.fn_sig(def_id)`.
    pub fn closure_sig_ty(self, def_id: DefId, tcx: TyCtxt<'_, '_, '_>) -> Ty<'tcx> {
        self.split(def_id, tcx).closure_sig_ty
    }

    /// Returns the type representing the yield type of the generator.
    pub fn generator_yield_ty(self, def_id: DefId, tcx: TyCtxt<'_, '_, '_>) -> Ty<'tcx> {
        self.closure_kind_ty(def_id, tcx)
    }

    /// Returns the type representing the return type of the generator.
    pub fn generator_return_ty(self, def_id: DefId, tcx: TyCtxt<'_, '_, '_>) -> Ty<'tcx> {
        self.closure_sig_ty(def_id, tcx)
    }

    /// Return the "generator signature", which consists of its yield
    /// and return types.
    ///
    /// NB. Some bits of the code prefers to see this wrapped in a
    /// binder, but it never contains bound regions. Probably this
    /// function should be removed.
    pub fn generator_poly_sig(self, def_id: DefId, tcx: TyCtxt<'_, '_, '_>) -> PolyGenSig<'tcx> {
        ty::Binder(self.generator_sig(def_id, tcx))
    }

    /// Return the "generator signature", which consists of its yield
    /// and return types.
    pub fn generator_sig(self, def_id: DefId, tcx: TyCtxt<'_, '_, '_>) -> GenSig<'tcx> {
        ty::GenSig {
            yield_ty: self.generator_yield_ty(def_id, tcx),
            return_ty: self.generator_return_ty(def_id, tcx),
        }
    }
}

impl<'tcx> ClosureSubsts<'tcx> {
    /// Returns the closure kind for this closure; only usable outside
    /// of an inference context, because in that context we know that
    /// there are no type variables.
    ///
    /// If you have an inference context, use `infcx.closure_kind()`.
    pub fn closure_kind(self, def_id: DefId, tcx: TyCtxt<'_, 'tcx, 'tcx>) -> ty::ClosureKind {
        self.split(def_id, tcx).closure_kind_ty.to_opt_closure_kind().unwrap()
    }

    /// Extracts the signature from the closure; only usable outside
    /// of an inference context, because in that context we know that
    /// there are no type variables.
    ///
    /// If you have an inference context, use `infcx.closure_sig()`.
    pub fn closure_sig(self, def_id: DefId, tcx: TyCtxt<'_, 'tcx, 'tcx>) -> ty::PolyFnSig<'tcx> {
        match self.closure_sig_ty(def_id, tcx).sty {
            ty::TyFnPtr(sig) => sig,
            ref t => bug!("closure_sig_ty is not a fn-ptr: {:?}", t),
        }
    }
}

impl<'a, 'gcx, 'tcx> ClosureSubsts<'tcx> {
    /// This returns the types of the MIR locals which had to be stored across suspension points.
    /// It is calculated in rustc_mir::transform::generator::StateTransform.
    /// All the types here must be in the tuple in GeneratorInterior.
    pub fn state_tys(self, def_id: DefId, tcx: TyCtxt<'a, 'gcx, 'tcx>) ->
        impl Iterator<Item=Ty<'tcx>> + 'a
    {
        let state = tcx.generator_layout(def_id).fields.iter();
        state.map(move |d| d.ty.subst(tcx, self.substs))
    }

    /// This is the types of all the fields stored in a generator.
    /// It includes the upvars, state types and the state discriminant which is u32.
    pub fn field_tys(self, def_id: DefId, tcx: TyCtxt<'a, 'gcx, 'tcx>) ->
        impl Iterator<Item=Ty<'tcx>> + 'a
    {
        let upvars = self.upvar_tys(def_id, tcx);
        let state = self.state_tys(def_id, tcx);
        upvars.chain(iter::once(tcx.types.u32)).chain(state)
    }
}

/// This describes the types that can be contained in a generator.
/// It will be a type variable initially and unified in the last stages of typeck of a body.
/// It contains a tuple of all the types that could end up on a generator frame.
/// The state transformation MIR pass may only produce layouts which mention types in this tuple.
/// Upvars are not counted here.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct GeneratorInterior<'tcx> {
    pub witness: Ty<'tcx>,
}

impl<'tcx> GeneratorInterior<'tcx> {
    pub fn new(witness: Ty<'tcx>) -> GeneratorInterior<'tcx> {
        GeneratorInterior { witness }
    }

    pub fn as_slice(&self) -> &'tcx Slice<Ty<'tcx>> {
        match self.witness.sty {
            ty::TyTuple(s, _) => s,
            _ => bug!(),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub enum ExistentialPredicate<'tcx> {
    /// e.g. Iterator
    Trait(ExistentialTraitRef<'tcx>),
    /// e.g. Iterator::Item = T
    Projection(ExistentialProjection<'tcx>),
    /// e.g. Send
    AutoTrait(DefId),
}

impl<'a, 'gcx, 'tcx> ExistentialPredicate<'tcx> {
    pub fn cmp(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, other: &Self) -> Ordering {
        use self::ExistentialPredicate::*;
        match (*self, *other) {
            (Trait(_), Trait(_)) => Ordering::Equal,
            (Projection(ref a), Projection(ref b)) =>
                tcx.def_path_hash(a.item_def_id).cmp(&tcx.def_path_hash(b.item_def_id)),
            (AutoTrait(ref a), AutoTrait(ref b)) =>
                tcx.trait_def(*a).def_path_hash.cmp(&tcx.trait_def(*b).def_path_hash),
            (Trait(_), _) => Ordering::Less,
            (Projection(_), Trait(_)) => Ordering::Greater,
            (Projection(_), _) => Ordering::Less,
            (AutoTrait(_), _) => Ordering::Greater,
        }
    }

}

impl<'a, 'gcx, 'tcx> Binder<ExistentialPredicate<'tcx>> {
    pub fn with_self_ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, self_ty: Ty<'tcx>)
        -> ty::Predicate<'tcx> {
        use ty::ToPredicate;
        match *self.skip_binder() {
            ExistentialPredicate::Trait(tr) => Binder(tr).with_self_ty(tcx, self_ty).to_predicate(),
            ExistentialPredicate::Projection(p) =>
                ty::Predicate::Projection(Binder(p.with_self_ty(tcx, self_ty))),
            ExistentialPredicate::AutoTrait(did) => {
                let trait_ref = Binder(ty::TraitRef {
                    def_id: did,
                    substs: tcx.mk_substs_trait(self_ty, &[]),
                });
                trait_ref.to_predicate()
            }
        }
    }
}

impl<'tcx> serialize::UseSpecializedDecodable for &'tcx Slice<ExistentialPredicate<'tcx>> {}

impl<'tcx> Slice<ExistentialPredicate<'tcx>> {
    pub fn principal(&self) -> Option<ExistentialTraitRef<'tcx>> {
        match self.get(0) {
            Some(&ExistentialPredicate::Trait(tr)) => Some(tr),
            _ => None,
        }
    }

    #[inline]
    pub fn projection_bounds<'a>(&'a self) ->
        impl Iterator<Item=ExistentialProjection<'tcx>> + 'a {
        self.iter().filter_map(|predicate| {
            match *predicate {
                ExistentialPredicate::Projection(p) => Some(p),
                _ => None,
            }
        })
    }

    #[inline]
    pub fn auto_traits<'a>(&'a self) -> impl Iterator<Item=DefId> + 'a {
        self.iter().filter_map(|predicate| {
            match *predicate {
                ExistentialPredicate::AutoTrait(d) => Some(d),
                _ => None
            }
        })
    }
}

impl<'tcx> Binder<&'tcx Slice<ExistentialPredicate<'tcx>>> {
    pub fn principal(&self) -> Option<PolyExistentialTraitRef<'tcx>> {
        self.skip_binder().principal().map(Binder)
    }

    #[inline]
    pub fn projection_bounds<'a>(&'a self) ->
        impl Iterator<Item=PolyExistentialProjection<'tcx>> + 'a {
        self.skip_binder().projection_bounds().map(Binder)
    }

    #[inline]
    pub fn auto_traits<'a>(&'a self) -> impl Iterator<Item=DefId> + 'a {
        self.skip_binder().auto_traits()
    }

    pub fn iter<'a>(&'a self)
        -> impl DoubleEndedIterator<Item=Binder<ExistentialPredicate<'tcx>>> + 'tcx {
        self.skip_binder().iter().cloned().map(Binder)
    }
}

/// A complete reference to a trait. These take numerous guises in syntax,
/// but perhaps the most recognizable form is in a where clause:
///
///     T : Foo<U>
///
/// This would be represented by a trait-reference where the def-id is the
/// def-id for the trait `Foo` and the substs define `T` as parameter 0,
/// and `U` as parameter 1.
///
/// Trait references also appear in object types like `Foo<U>`, but in
/// that case the `Self` parameter is absent from the substitutions.
///
/// Note that a `TraitRef` introduces a level of region binding, to
/// account for higher-ranked trait bounds like `T : for<'a> Foo<&'a
/// U>` or higher-ranked object types.
#[derive(Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct TraitRef<'tcx> {
    pub def_id: DefId,
    pub substs: &'tcx Substs<'tcx>,
}

impl<'tcx> TraitRef<'tcx> {
    pub fn new(def_id: DefId, substs: &'tcx Substs<'tcx>) -> TraitRef<'tcx> {
        TraitRef { def_id: def_id, substs: substs }
    }

    pub fn self_ty(&self) -> Ty<'tcx> {
        self.substs.type_at(0)
    }

    pub fn input_types<'a>(&'a self) -> impl DoubleEndedIterator<Item=Ty<'tcx>> + 'a {
        // Select only the "input types" from a trait-reference. For
        // now this is all the types that appear in the
        // trait-reference, but it should eventually exclude
        // associated types.
        self.substs.types()
    }
}

pub type PolyTraitRef<'tcx> = Binder<TraitRef<'tcx>>;

impl<'tcx> PolyTraitRef<'tcx> {
    pub fn self_ty(&self) -> Ty<'tcx> {
        self.0.self_ty()
    }

    pub fn def_id(&self) -> DefId {
        self.0.def_id
    }

    pub fn substs(&self) -> &'tcx Substs<'tcx> {
        // FIXME(#20664) every use of this fn is probably a bug, it should yield Binder<>
        self.0.substs
    }

    pub fn input_types<'a>(&'a self) -> impl DoubleEndedIterator<Item=Ty<'tcx>> + 'a {
        // FIXME(#20664) every use of this fn is probably a bug, it should yield Binder<>
        self.0.input_types()
    }

    pub fn to_poly_trait_predicate(&self) -> ty::PolyTraitPredicate<'tcx> {
        // Note that we preserve binding levels
        Binder(ty::TraitPredicate { trait_ref: self.0.clone() })
    }
}

/// An existential reference to a trait, where `Self` is erased.
/// For example, the trait object `Trait<'a, 'b, X, Y>` is:
///
///     exists T. T: Trait<'a, 'b, X, Y>
///
/// The substitutions don't include the erased `Self`, only trait
/// type and lifetime parameters (`[X, Y]` and `['a, 'b]` above).
#[derive(Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct ExistentialTraitRef<'tcx> {
    pub def_id: DefId,
    pub substs: &'tcx Substs<'tcx>,
}

impl<'a, 'gcx, 'tcx> ExistentialTraitRef<'tcx> {
    pub fn input_types<'b>(&'b self) -> impl DoubleEndedIterator<Item=Ty<'tcx>> + 'b {
        // Select only the "input types" from a trait-reference. For
        // now this is all the types that appear in the
        // trait-reference, but it should eventually exclude
        // associated types.
        self.substs.types()
    }

    /// Object types don't have a self-type specified. Therefore, when
    /// we convert the principal trait-ref into a normal trait-ref,
    /// you must give *some* self-type. A common choice is `mk_err()`
    /// or some skolemized type.
    pub fn with_self_ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, self_ty: Ty<'tcx>)
        -> ty::TraitRef<'tcx>  {
        // otherwise the escaping regions would be captured by the binder
        assert!(!self_ty.has_escaping_regions());

        ty::TraitRef {
            def_id: self.def_id,
            substs: tcx.mk_substs(
                iter::once(Kind::from(self_ty)).chain(self.substs.iter().cloned()))
        }
    }
}

pub type PolyExistentialTraitRef<'tcx> = Binder<ExistentialTraitRef<'tcx>>;

impl<'tcx> PolyExistentialTraitRef<'tcx> {
    pub fn def_id(&self) -> DefId {
        self.0.def_id
    }

    pub fn input_types<'a>(&'a self) -> impl DoubleEndedIterator<Item=Ty<'tcx>> + 'a {
        // FIXME(#20664) every use of this fn is probably a bug, it should yield Binder<>
        self.0.input_types()
    }
}

/// Binder is a binder for higher-ranked lifetimes. It is part of the
/// compiler's representation for things like `for<'a> Fn(&'a isize)`
/// (which would be represented by the type `PolyTraitRef ==
/// Binder<TraitRef>`). Note that when we skolemize, instantiate,
/// erase, or otherwise "discharge" these bound regions, we change the
/// type from `Binder<T>` to just `T` (see
/// e.g. `liberate_late_bound_regions`).
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct Binder<T>(pub T);

impl<T> Binder<T> {
    /// Wraps `value` in a binder, asserting that `value` does not
    /// contain any bound regions that would be bound by the
    /// binder. This is commonly used to 'inject' a value T into a
    /// different binding level.
    pub fn dummy<'tcx>(value: T) -> Binder<T>
        where T: TypeFoldable<'tcx>
    {
        assert!(!value.has_escaping_regions());
        Binder(value)
    }

    /// Skips the binder and returns the "bound" value. This is a
    /// risky thing to do because it's easy to get confused about
    /// debruijn indices and the like. It is usually better to
    /// discharge the binder using `no_late_bound_regions` or
    /// `replace_late_bound_regions` or something like
    /// that. `skip_binder` is only valid when you are either
    /// extracting data that has nothing to do with bound regions, you
    /// are doing some sort of test that does not involve bound
    /// regions, or you are being very careful about your depth
    /// accounting.
    ///
    /// Some examples where `skip_binder` is reasonable:
    ///
    /// - extracting the def-id from a PolyTraitRef;
    /// - comparing the self type of a PolyTraitRef to see if it is equal to
    ///   a type parameter `X`, since the type `X`  does not reference any regions
    pub fn skip_binder(&self) -> &T {
        &self.0
    }

    pub fn as_ref(&self) -> Binder<&T> {
        ty::Binder(&self.0)
    }

    pub fn map_bound_ref<F, U>(&self, f: F) -> Binder<U>
        where F: FnOnce(&T) -> U
    {
        self.as_ref().map_bound(f)
    }

    pub fn map_bound<F, U>(self, f: F) -> Binder<U>
        where F: FnOnce(T) -> U
    {
        ty::Binder(f(self.0))
    }

    /// Unwraps and returns the value within, but only if it contains
    /// no bound regions at all. (In other words, if this binder --
    /// and indeed any enclosing binder -- doesn't bind anything at
    /// all.) Otherwise, returns `None`.
    ///
    /// (One could imagine having a method that just unwraps a single
    /// binder, but permits late-bound regions bound by enclosing
    /// binders, but that would require adjusting the debruijn
    /// indices, and given the shallow binding structure we often use,
    /// would not be that useful.)
    pub fn no_late_bound_regions<'tcx>(self) -> Option<T>
        where T : TypeFoldable<'tcx>
    {
        if self.skip_binder().has_escaping_regions() {
            None
        } else {
            Some(self.skip_binder().clone())
        }
    }

    /// Given two things that have the same binder level,
    /// and an operation that wraps on their contents, execute the operation
    /// and then wrap its result.
    ///
    /// `f` should consider bound regions at depth 1 to be free, and
    /// anything it produces with bound regions at depth 1 will be
    /// bound in the resulting return value.
    pub fn fuse<U,F,R>(self, u: Binder<U>, f: F) -> Binder<R>
        where F: FnOnce(T, U) -> R
    {
        ty::Binder(f(self.0, u.0))
    }

    /// Split the contents into two things that share the same binder
    /// level as the original, returning two distinct binders.
    ///
    /// `f` should consider bound regions at depth 1 to be free, and
    /// anything it produces with bound regions at depth 1 will be
    /// bound in the resulting return values.
    pub fn split<U,V,F>(self, f: F) -> (Binder<U>, Binder<V>)
        where F: FnOnce(T) -> (U, V)
    {
        let (u, v) = f(self.0);
        (ty::Binder(u), ty::Binder(v))
    }
}

/// Represents the projection of an associated type. In explicit UFCS
/// form this would be written `<T as Trait<..>>::N`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct ProjectionTy<'tcx> {
    /// The parameters of the associated item.
    pub substs: &'tcx Substs<'tcx>,

    /// The DefId of the TraitItem for the associated type N.
    ///
    /// Note that this is not the DefId of the TraitRef containing this
    /// associated type, which is in tcx.associated_item(item_def_id).container.
    pub item_def_id: DefId,
}

impl<'a, 'tcx> ProjectionTy<'tcx> {
    /// Construct a ProjectionTy by searching the trait from trait_ref for the
    /// associated item named item_name.
    pub fn from_ref_and_name(
        tcx: TyCtxt, trait_ref: ty::TraitRef<'tcx>, item_name: Name
    ) -> ProjectionTy<'tcx> {
        let item_def_id = tcx.associated_items(trait_ref.def_id).find(|item| {
            item.kind == ty::AssociatedKind::Type &&
            tcx.hygienic_eq(item_name, item.name, trait_ref.def_id)
        }).unwrap().def_id;

        ProjectionTy {
            substs: trait_ref.substs,
            item_def_id,
        }
    }

    /// Extracts the underlying trait reference from this projection.
    /// For example, if this is a projection of `<T as Iterator>::Item`,
    /// then this function would return a `T: Iterator` trait reference.
    pub fn trait_ref(&self, tcx: TyCtxt) -> ty::TraitRef<'tcx> {
        let def_id = tcx.associated_item(self.item_def_id).container.id();
        ty::TraitRef {
            def_id,
            substs: self.substs,
        }
    }

    pub fn self_ty(&self) -> Ty<'tcx> {
        self.substs.type_at(0)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct GenSig<'tcx> {
    pub yield_ty: Ty<'tcx>,
    pub return_ty: Ty<'tcx>,
}

pub type PolyGenSig<'tcx> = Binder<GenSig<'tcx>>;

impl<'tcx> PolyGenSig<'tcx> {
    pub fn yield_ty(&self) -> ty::Binder<Ty<'tcx>> {
        self.map_bound_ref(|sig| sig.yield_ty)
    }
    pub fn return_ty(&self) -> ty::Binder<Ty<'tcx>> {
        self.map_bound_ref(|sig| sig.return_ty)
    }
}

/// Signature of a function type, which I have arbitrarily
/// decided to use to refer to the input/output types.
///
/// - `inputs` is the list of arguments and their modes.
/// - `output` is the return type.
/// - `variadic` indicates whether this is a variadic function. (only true for foreign fns)
#[derive(Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct FnSig<'tcx> {
    pub inputs_and_output: &'tcx Slice<Ty<'tcx>>,
    pub variadic: bool,
    pub unsafety: hir::Unsafety,
    pub abi: abi::Abi,
}

impl<'tcx> FnSig<'tcx> {
    pub fn inputs(&self) -> &'tcx [Ty<'tcx>] {
        &self.inputs_and_output[..self.inputs_and_output.len() - 1]
    }

    pub fn output(&self) -> Ty<'tcx> {
        self.inputs_and_output[self.inputs_and_output.len() - 1]
    }
}

pub type PolyFnSig<'tcx> = Binder<FnSig<'tcx>>;

impl<'tcx> PolyFnSig<'tcx> {
    pub fn inputs(&self) -> Binder<&'tcx [Ty<'tcx>]> {
        Binder(self.skip_binder().inputs())
    }
    pub fn input(&self, index: usize) -> ty::Binder<Ty<'tcx>> {
        self.map_bound_ref(|fn_sig| fn_sig.inputs()[index])
    }
    pub fn inputs_and_output(&self) -> ty::Binder<&'tcx Slice<Ty<'tcx>>> {
        self.map_bound_ref(|fn_sig| fn_sig.inputs_and_output)
    }
    pub fn output(&self) -> ty::Binder<Ty<'tcx>> {
        self.map_bound_ref(|fn_sig| fn_sig.output().clone())
    }
    pub fn variadic(&self) -> bool {
        self.skip_binder().variadic
    }
    pub fn unsafety(&self) -> hir::Unsafety {
        self.skip_binder().unsafety
    }
    pub fn abi(&self) -> abi::Abi {
        self.skip_binder().abi
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct ParamTy {
    pub idx: u32,
    pub name: Name,
}

impl<'a, 'gcx, 'tcx> ParamTy {
    pub fn new(index: u32, name: Name) -> ParamTy {
        ParamTy { idx: index, name: name }
    }

    pub fn for_self() -> ParamTy {
        ParamTy::new(0, keywords::SelfType.name())
    }

    pub fn for_def(def: &ty::TypeParameterDef) -> ParamTy {
        ParamTy::new(def.index, def.name)
    }

    pub fn to_ty(self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        tcx.mk_param(self.idx, self.name)
    }

    pub fn is_self(&self) -> bool {
        if self.name == keywords::SelfType.name() {
            assert_eq!(self.idx, 0);
            true
        } else {
            false
        }
    }
}

/// A [De Bruijn index][dbi] is a standard means of representing
/// regions (and perhaps later types) in a higher-ranked setting. In
/// particular, imagine a type like this:
///
///     for<'a> fn(for<'b> fn(&'b isize, &'a isize), &'a char)
///     ^          ^            |        |         |
///     |          |            |        |         |
///     |          +------------+ 1      |         |
///     |                                |         |
///     +--------------------------------+ 2       |
///     |                                          |
///     +------------------------------------------+ 1
///
/// In this type, there are two binders (the outer fn and the inner
/// fn). We need to be able to determine, for any given region, which
/// fn type it is bound by, the inner or the outer one. There are
/// various ways you can do this, but a De Bruijn index is one of the
/// more convenient and has some nice properties. The basic idea is to
/// count the number of binders, inside out. Some examples should help
/// clarify what I mean.
///
/// Let's start with the reference type `&'b isize` that is the first
/// argument to the inner function. This region `'b` is assigned a De
/// Bruijn index of 1, meaning "the innermost binder" (in this case, a
/// fn). The region `'a` that appears in the second argument type (`&'a
/// isize`) would then be assigned a De Bruijn index of 2, meaning "the
/// second-innermost binder". (These indices are written on the arrays
/// in the diagram).
///
/// What is interesting is that De Bruijn index attached to a particular
/// variable will vary depending on where it appears. For example,
/// the final type `&'a char` also refers to the region `'a` declared on
/// the outermost fn. But this time, this reference is not nested within
/// any other binders (i.e., it is not an argument to the inner fn, but
/// rather the outer one). Therefore, in this case, it is assigned a
/// De Bruijn index of 1, because the innermost binder in that location
/// is the outer fn.
///
/// [dbi]: http://en.wikipedia.org/wiki/De_Bruijn_index
#[derive(Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Debug, Copy, PartialOrd, Ord)]
pub struct DebruijnIndex {
    /// We maintain the invariant that this is never 0. So 1 indicates
    /// the innermost binder. To ensure this, create with `DebruijnIndex::new`.
    pub depth: u32,
}

pub type Region<'tcx> = &'tcx RegionKind;

/// Representation of regions.
///
/// Unlike types, most region variants are "fictitious", not concrete,
/// regions. Among these, `ReStatic`, `ReEmpty` and `ReScope` are the only
/// ones representing concrete regions.
///
/// ## Bound Regions
///
/// These are regions that are stored behind a binder and must be substituted
/// with some concrete region before being used. There are 2 kind of
/// bound regions: early-bound, which are bound in an item's Generics,
/// and are substituted by a Substs,  and late-bound, which are part of
/// higher-ranked types (e.g. `for<'a> fn(&'a ())`) and are substituted by
/// the likes of `liberate_late_bound_regions`. The distinction exists
/// because higher-ranked lifetimes aren't supported in all places. See [1][2].
///
/// Unlike TyParam-s, bound regions are not supposed to exist "in the wild"
/// outside their binder, e.g. in types passed to type inference, and
/// should first be substituted (by skolemized regions, free regions,
/// or region variables).
///
/// ## Skolemized and Free Regions
///
/// One often wants to work with bound regions without knowing their precise
/// identity. For example, when checking a function, the lifetime of a borrow
/// can end up being assigned to some region parameter. In these cases,
/// it must be ensured that bounds on the region can't be accidentally
/// assumed without being checked.
///
/// The process of doing that is called "skolemization". The bound regions
/// are replaced by skolemized markers, which don't satisfy any relation
/// not explicitly provided.
///
/// There are 2 kinds of skolemized regions in rustc: `ReFree` and
/// `ReSkolemized`. When checking an item's body, `ReFree` is supposed
/// to be used. These also support explicit bounds: both the internally-stored
/// *scope*, which the region is assumed to outlive, as well as other
/// relations stored in the `FreeRegionMap`. Note that these relations
/// aren't checked when you `make_subregion` (or `eq_types`), only by
/// `resolve_regions_and_report_errors`.
///
/// When working with higher-ranked types, some region relations aren't
/// yet known, so you can't just call `resolve_regions_and_report_errors`.
/// `ReSkolemized` is designed for this purpose. In these contexts,
/// there's also the risk that some inference variable laying around will
/// get unified with your skolemized region: if you want to check whether
/// `for<'a> Foo<'_>: 'a`, and you substitute your bound region `'a`
/// with a skolemized region `'%a`, the variable `'_` would just be
/// instantiated to the skolemized region `'%a`, which is wrong because
/// the inference variable is supposed to satisfy the relation
/// *for every value of the skolemized region*. To ensure that doesn't
/// happen, you can use `leak_check`. This is more clearly explained
/// by infer/higher_ranked/README.md.
///
/// [1]: http://smallcultfollowing.com/babysteps/blog/2013/10/29/intermingled-parameter-lists/
/// [2]: http://smallcultfollowing.com/babysteps/blog/2013/11/04/intermingled-parameter-lists/
#[derive(Clone, PartialEq, Eq, Hash, Copy, RustcEncodable, RustcDecodable, PartialOrd, Ord)]
pub enum RegionKind {
    // Region bound in a type or fn declaration which will be
    // substituted 'early' -- that is, at the same time when type
    // parameters are substituted.
    ReEarlyBound(EarlyBoundRegion),

    // Region bound in a function scope, which will be substituted when the
    // function is called.
    ReLateBound(DebruijnIndex, BoundRegion),

    /// When checking a function body, the types of all arguments and so forth
    /// that refer to bound region parameters are modified to refer to free
    /// region parameters.
    ReFree(FreeRegion),

    /// A concrete region naming some statically determined scope
    /// (e.g. an expression or sequence of statements) within the
    /// current function.
    ReScope(region::Scope),

    /// Static data that has an "infinite" lifetime. Top in the region lattice.
    ReStatic,

    /// A region variable.  Should not exist after typeck.
    ReVar(RegionVid),

    /// A skolemized region - basically the higher-ranked version of ReFree.
    /// Should not exist after typeck.
    ReSkolemized(SkolemizedRegionVid, BoundRegion),

    /// Empty lifetime is for data that is never accessed.
    /// Bottom in the region lattice. We treat ReEmpty somewhat
    /// specially; at least right now, we do not generate instances of
    /// it during the GLB computations, but rather
    /// generate an error instead. This is to improve error messages.
    /// The only way to get an instance of ReEmpty is to have a region
    /// variable with no constraints.
    ReEmpty,

    /// Erased region, used by trait selection, in MIR and during trans.
    ReErased,

    /// These are regions bound in the "defining type" for a
    /// closure. They are used ONLY as part of the
    /// `ClosureRegionRequirements` that are produced by MIR borrowck.
    /// See `ClosureRegionRequirements` for more details.
    ReClosureBound(RegionVid),
}

impl<'tcx> serialize::UseSpecializedDecodable for Region<'tcx> {}

#[derive(Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Debug, PartialOrd, Ord)]
pub struct EarlyBoundRegion {
    pub def_id: DefId,
    pub index: u32,
    pub name: Name,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct TyVid {
    pub index: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct IntVid {
    pub index: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct FloatVid {
    pub index: u32,
}

newtype_index!(RegionVid
    {
        pub idx
        DEBUG_FORMAT = custom,
    });

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, PartialOrd, Ord)]
pub struct SkolemizedRegionVid {
    pub index: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub enum InferTy {
    TyVar(TyVid),
    IntVar(IntVid),
    FloatVar(FloatVid),

    /// A `FreshTy` is one that is generated as a replacement for an
    /// unbound type variable. This is convenient for caching etc. See
    /// `infer::freshen` for more details.
    FreshTy(u32),
    FreshIntTy(u32),
    FreshFloatTy(u32),
}

/// A `ProjectionPredicate` for an `ExistentialTraitRef`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct ExistentialProjection<'tcx> {
    pub item_def_id: DefId,
    pub substs: &'tcx Substs<'tcx>,
    pub ty: Ty<'tcx>,
}

pub type PolyExistentialProjection<'tcx> = Binder<ExistentialProjection<'tcx>>;

impl<'a, 'tcx, 'gcx> ExistentialProjection<'tcx> {
    /// Extracts the underlying existential trait reference from this projection.
    /// For example, if this is a projection of `exists T. <T as Iterator>::Item == X`,
    /// then this function would return a `exists T. T: Iterator` existential trait
    /// reference.
    pub fn trait_ref(&self, tcx: TyCtxt) -> ty::ExistentialTraitRef<'tcx> {
        let def_id = tcx.associated_item(self.item_def_id).container.id();
        ty::ExistentialTraitRef{
            def_id,
            substs: self.substs,
        }
    }

    pub fn with_self_ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                        self_ty: Ty<'tcx>)
                        -> ty::ProjectionPredicate<'tcx>
    {
        // otherwise the escaping regions would be captured by the binders
        assert!(!self_ty.has_escaping_regions());

        ty::ProjectionPredicate {
            projection_ty: ty::ProjectionTy {
                item_def_id: self.item_def_id,
                substs: tcx.mk_substs(
                iter::once(Kind::from(self_ty)).chain(self.substs.iter().cloned())),
            },
            ty: self.ty,
        }
    }
}

impl<'a, 'tcx, 'gcx> PolyExistentialProjection<'tcx> {
    pub fn with_self_ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, self_ty: Ty<'tcx>)
        -> ty::PolyProjectionPredicate<'tcx> {
        self.map_bound(|p| p.with_self_ty(tcx, self_ty))
    }
}

impl DebruijnIndex {
    pub fn new(depth: u32) -> DebruijnIndex {
        assert!(depth > 0);
        DebruijnIndex { depth: depth }
    }

    pub fn shifted(&self, amount: u32) -> DebruijnIndex {
        DebruijnIndex { depth: self.depth + amount }
    }
}

/// Region utilities
impl RegionKind {
    pub fn is_late_bound(&self) -> bool {
        match *self {
            ty::ReLateBound(..) => true,
            _ => false,
        }
    }

    pub fn needs_infer(&self) -> bool {
        match *self {
            ty::ReVar(..) | ty::ReSkolemized(..) => true,
            _ => false
        }
    }

    pub fn escapes_depth(&self, depth: u32) -> bool {
        match *self {
            ty::ReLateBound(debruijn, _) => debruijn.depth > depth,
            _ => false,
        }
    }

    /// Returns the depth of `self` from the (1-based) binding level `depth`
    pub fn from_depth(&self, depth: u32) -> RegionKind {
        match *self {
            ty::ReLateBound(debruijn, r) => ty::ReLateBound(DebruijnIndex {
                depth: debruijn.depth - (depth - 1)
            }, r),
            r => r
        }
    }

    pub fn type_flags(&self) -> TypeFlags {
        let mut flags = TypeFlags::empty();

        match *self {
            ty::ReVar(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_RE_INFER;
                flags = flags | TypeFlags::KEEP_IN_LOCAL_TCX;
            }
            ty::ReSkolemized(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_RE_INFER;
                flags = flags | TypeFlags::HAS_RE_SKOL;
                flags = flags | TypeFlags::KEEP_IN_LOCAL_TCX;
            }
            ty::ReLateBound(..) => { }
            ty::ReEarlyBound(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_RE_EARLY_BOUND;
            }
            ty::ReEmpty |
            ty::ReStatic |
            ty::ReFree { .. } |
            ty::ReScope { .. } => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
            }
            ty::ReErased => {
            }
            ty::ReClosureBound(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
            }
        }

        match *self {
            ty::ReStatic | ty::ReEmpty | ty::ReErased => (),
            _ => flags = flags | TypeFlags::HAS_LOCAL_NAMES,
        }

        debug!("type_flags({:?}) = {:?}", self, flags);

        flags
    }

    /// Given an early-bound or free region, returns the def-id where it was bound.
    /// For example, consider the regions in this snippet of code:
    ///
    /// ```
    /// impl<'a> Foo {
    ///      ^^ -- early bound, declared on an impl
    ///
    ///     fn bar<'b, 'c>(x: &self, y: &'b u32, z: &'c u64) where 'static: 'c
    ///            ^^  ^^     ^ anonymous, late-bound
    ///            |   early-bound, appears in where-clauses
    ///            late-bound, appears only in fn args
    ///     {..}
    /// }
    /// ```
    ///
    /// Here, `free_region_binding_scope('a)` would return the def-id
    /// of the impl, and for all the other highlighted regions, it
    /// would return the def-id of the function. In other cases (not shown), this
    /// function might return the def-id of a closure.
    pub fn free_region_binding_scope(&self, tcx: TyCtxt<'_, '_, '_>) -> DefId {
        match self {
            ty::ReEarlyBound(br) => {
                tcx.parent_def_id(br.def_id).unwrap()
            }
            ty::ReFree(fr) => fr.scope,
            _ => bug!("free_region_binding_scope invoked on inappropriate region: {:?}", self),
        }
    }
}

/// Type utilities
impl<'a, 'gcx, 'tcx> TyS<'tcx> {
    pub fn is_nil(&self) -> bool {
        match self.sty {
            TyTuple(ref tys, _) => tys.is_empty(),
            _ => false,
        }
    }

    pub fn is_never(&self) -> bool {
        match self.sty {
            TyNever => true,
            _ => false,
        }
    }

    /// Test whether this is a `()` which was produced by defaulting a
    /// diverging type variable with feature(never_type) disabled.
    pub fn is_defaulted_unit(&self) -> bool {
        match self.sty {
            TyTuple(_, true) => true,
            _ => false,
        }
    }

    pub fn is_primitive(&self) -> bool {
        match self.sty {
            TyBool | TyChar | TyInt(_) | TyUint(_) | TyFloat(_) => true,
            _ => false,
        }
    }

    pub fn is_ty_var(&self) -> bool {
        match self.sty {
            TyInfer(TyVar(_)) => true,
            _ => false,
        }
    }

    pub fn is_phantom_data(&self) -> bool {
        if let TyAdt(def, _) = self.sty {
            def.is_phantom_data()
        } else {
            false
        }
    }

    pub fn is_bool(&self) -> bool { self.sty == TyBool }

    pub fn is_param(&self, index: u32) -> bool {
        match self.sty {
            ty::TyParam(ref data) => data.idx == index,
            _ => false,
        }
    }

    pub fn is_self(&self) -> bool {
        match self.sty {
            TyParam(ref p) => p.is_self(),
            _ => false,
        }
    }

    pub fn is_slice(&self) -> bool {
        match self.sty {
            TyRawPtr(mt) | TyRef(_, mt) => match mt.ty.sty {
                TySlice(_) | TyStr => true,
                _ => false,
            },
            _ => false
        }
    }

    #[inline]
    pub fn is_simd(&self) -> bool {
        match self.sty {
            TyAdt(def, _) => def.repr.simd(),
            _ => false,
        }
    }

    pub fn sequence_element_type(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        match self.sty {
            TyArray(ty, _) | TySlice(ty) => ty,
            TyStr => tcx.mk_mach_uint(ast::UintTy::U8),
            _ => bug!("sequence_element_type called on non-sequence value: {}", self),
        }
    }

    pub fn simd_type(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> Ty<'tcx> {
        match self.sty {
            TyAdt(def, substs) => {
                def.non_enum_variant().fields[0].ty(tcx, substs)
            }
            _ => bug!("simd_type called on invalid type")
        }
    }

    pub fn simd_size(&self, _cx: TyCtxt) -> usize {
        match self.sty {
            TyAdt(def, _) => def.non_enum_variant().fields.len(),
            _ => bug!("simd_size called on invalid type")
        }
    }

    pub fn is_region_ptr(&self) -> bool {
        match self.sty {
            TyRef(..) => true,
            _ => false,
        }
    }

    pub fn is_mutable_pointer(&self) -> bool {
        match self.sty {
            TyRawPtr(tnm) | TyRef(_, tnm) => if let hir::Mutability::MutMutable = tnm.mutbl {
                true
            } else {
                false
            },
            _ => false
        }
    }

    pub fn is_unsafe_ptr(&self) -> bool {
        match self.sty {
            TyRawPtr(_) => return true,
            _ => return false,
        }
    }

    pub fn is_box(&self) -> bool {
        match self.sty {
            TyAdt(def, _) => def.is_box(),
            _ => false,
        }
    }

    /// panics if called on any type other than `Box<T>`
    pub fn boxed_ty(&self) -> Ty<'tcx> {
        match self.sty {
            TyAdt(def, substs) if def.is_box() => substs.type_at(0),
            _ => bug!("`boxed_ty` is called on non-box type {:?}", self),
        }
    }

    /// A scalar type is one that denotes an atomic datum, with no sub-components.
    /// (A TyRawPtr is scalar because it represents a non-managed pointer, so its
    /// contents are abstract to rustc.)
    pub fn is_scalar(&self) -> bool {
        match self.sty {
            TyBool | TyChar | TyInt(_) | TyFloat(_) | TyUint(_) |
            TyInfer(IntVar(_)) | TyInfer(FloatVar(_)) |
            TyFnDef(..) | TyFnPtr(_) | TyRawPtr(_) => true,
            _ => false
        }
    }

    /// Returns true if this type is a floating point type and false otherwise.
    pub fn is_floating_point(&self) -> bool {
        match self.sty {
            TyFloat(_) |
            TyInfer(FloatVar(_)) => true,
            _ => false,
        }
    }

    pub fn is_trait(&self) -> bool {
        match self.sty {
            TyDynamic(..) => true,
            _ => false,
        }
    }

    pub fn is_enum(&self) -> bool {
        match self.sty {
            TyAdt(adt_def, _) => {
                adt_def.is_enum()
            }
            _ => false,
        }
    }

    pub fn is_closure(&self) -> bool {
        match self.sty {
            TyClosure(..) => true,
            _ => false,
        }
    }

    pub fn is_generator(&self) -> bool {
        match self.sty {
            TyGenerator(..) => true,
            _ => false,
        }
    }

    pub fn is_integral(&self) -> bool {
        match self.sty {
            TyInfer(IntVar(_)) | TyInt(_) | TyUint(_) => true,
            _ => false
        }
    }

    pub fn is_fresh_ty(&self) -> bool {
        match self.sty {
            TyInfer(FreshTy(_)) => true,
            _ => false,
        }
    }

    pub fn is_fresh(&self) -> bool {
        match self.sty {
            TyInfer(FreshTy(_)) => true,
            TyInfer(FreshIntTy(_)) => true,
            TyInfer(FreshFloatTy(_)) => true,
            _ => false,
        }
    }

    pub fn is_char(&self) -> bool {
        match self.sty {
            TyChar => true,
            _ => false,
        }
    }

    pub fn is_fp(&self) -> bool {
        match self.sty {
            TyInfer(FloatVar(_)) | TyFloat(_) => true,
            _ => false
        }
    }

    pub fn is_numeric(&self) -> bool {
        self.is_integral() || self.is_fp()
    }

    pub fn is_signed(&self) -> bool {
        match self.sty {
            TyInt(_) => true,
            _ => false,
        }
    }

    pub fn is_machine(&self) -> bool {
        match self.sty {
            TyInt(ast::IntTy::Isize) | TyUint(ast::UintTy::Usize) => false,
            TyInt(..) | TyUint(..) | TyFloat(..) => true,
            _ => false,
        }
    }

    pub fn has_concrete_skeleton(&self) -> bool {
        match self.sty {
            TyParam(_) | TyInfer(_) | TyError => false,
            _ => true,
        }
    }

    /// Returns the type and mutability of *ty.
    ///
    /// The parameter `explicit` indicates if this is an *explicit* dereference.
    /// Some types---notably unsafe ptrs---can only be dereferenced explicitly.
    pub fn builtin_deref(&self, explicit: bool, pref: ty::LvaluePreference)
        -> Option<TypeAndMut<'tcx>>
    {
        match self.sty {
            TyAdt(def, _) if def.is_box() => {
                Some(TypeAndMut {
                    ty: self.boxed_ty(),
                    mutbl: if pref == ty::PreferMutLvalue {
                        hir::MutMutable
                    } else {
                        hir::MutImmutable
                    },
                })
            },
            TyRef(_, mt) => Some(mt),
            TyRawPtr(mt) if explicit => Some(mt),
            _ => None,
        }
    }

    /// Returns the type of ty[i]
    pub fn builtin_index(&self) -> Option<Ty<'tcx>> {
        match self.sty {
            TyArray(ty, _) | TySlice(ty) => Some(ty),
            _ => None,
        }
    }

    pub fn fn_sig(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> PolyFnSig<'tcx> {
        match self.sty {
            TyFnDef(def_id, substs) => {
                tcx.fn_sig(def_id).subst(tcx, substs)
            }
            TyFnPtr(f) => f,
            _ => bug!("Ty::fn_sig() called on non-fn type: {:?}", self)
        }
    }

    pub fn is_fn(&self) -> bool {
        match self.sty {
            TyFnDef(..) | TyFnPtr(_) => true,
            _ => false,
        }
    }

    pub fn ty_to_def_id(&self) -> Option<DefId> {
        match self.sty {
            TyDynamic(ref tt, ..) => tt.principal().map(|p| p.def_id()),
            TyAdt(def, _) => Some(def.did),
            TyForeign(did) => Some(did),
            TyClosure(id, _) => Some(id),
            TyFnDef(id, _) => Some(id),
            _ => None,
        }
    }

    pub fn ty_adt_def(&self) -> Option<&'tcx AdtDef> {
        match self.sty {
            TyAdt(adt, _) => Some(adt),
            _ => None,
        }
    }

    /// Returns the regions directly referenced from this type (but
    /// not types reachable from this type via `walk_tys`). This
    /// ignores late-bound regions binders.
    pub fn regions(&self) -> Vec<ty::Region<'tcx>> {
        match self.sty {
            TyRef(region, _) => {
                vec![region]
            }
            TyDynamic(ref obj, region) => {
                let mut v = vec![region];
                if let Some(p) = obj.principal() {
                    v.extend(p.skip_binder().substs.regions());
                }
                v
            }
            TyAdt(_, substs) | TyAnon(_, substs) => {
                substs.regions().collect()
            }
            TyClosure(_, ref substs) | TyGenerator(_, ref substs, _) => {
                substs.substs.regions().collect()
            }
            TyProjection(ref data) => {
                data.substs.regions().collect()
            }
            TyFnDef(..) |
            TyFnPtr(_) |
            TyBool |
            TyChar |
            TyInt(_) |
            TyUint(_) |
            TyFloat(_) |
            TyStr |
            TyArray(..) |
            TySlice(_) |
            TyRawPtr(_) |
            TyNever |
            TyTuple(..) |
            TyForeign(..) |
            TyParam(_) |
            TyInfer(_) |
            TyError => {
                vec![]
            }
        }
    }

    /// When we create a closure, we record its kind (i.e., what trait
    /// it implements) into its `ClosureSubsts` using a type
    /// parameter. This is kind of a phantom type, except that the
    /// most convenient thing for us to are the integral types. This
    /// function converts such a special type into the closure
    /// kind. To go the other way, use
    /// `tcx.closure_kind_ty(closure_kind)`.
    ///
    /// Note that during type checking, we use an inference variable
    /// to represent the closure kind, because it has not yet been
    /// inferred. Once upvar inference (in `src/librustc_typeck/check/upvar.rs`)
    /// is complete, that type variable will be unified.
    pub fn to_opt_closure_kind(&self) -> Option<ty::ClosureKind> {
        match self.sty {
            TyInt(int_ty) => match int_ty {
                ast::IntTy::I8 => Some(ty::ClosureKind::Fn),
                ast::IntTy::I16 => Some(ty::ClosureKind::FnMut),
                ast::IntTy::I32 => Some(ty::ClosureKind::FnOnce),
                _ => bug!("cannot convert type `{:?}` to a closure kind", self),
            },

            TyInfer(_) => None,

            TyError => Some(ty::ClosureKind::Fn),

            _ => bug!("cannot convert type `{:?}` to a closure kind", self),
        }
    }
}

/// Typed constant value.
#[derive(Copy, Clone, Debug, Hash, RustcEncodable, RustcDecodable, Eq, PartialEq)]
pub struct Const<'tcx> {
    pub ty: Ty<'tcx>,

    // FIXME(eddyb) Replace this with a miri value.
    pub val: ConstVal<'tcx>,
}

impl<'tcx> serialize::UseSpecializedDecodable for &'tcx Const<'tcx> {}
