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

use middle::region;
use ty::subst::Substs;
use ty::{self, AdtDef, TypeFlags, Ty, TyCtxt, TypeFoldable};
use ty::{Slice, TyS};
use ty::subst::Kind;

use std::fmt;
use std::iter;
use std::cmp::Ordering;
use syntax::abi;
use syntax::ast::{self, Name};
use syntax::symbol::{keywords, InternedString};
use util::nodemap::FxHashSet;

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
    pub scope: region::CodeExtent,
    pub bound_region: BoundRegion
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
    BrNamed(DefId, Name, Issue32330),

    /// Fresh bound identifiers created during GLB computations.
    BrFresh(u32),

    // Anonymous region for the implicit env pointer parameter
    // to a closure
    BrEnv
}

/// True if this late-bound region is unconstrained, and hence will
/// become early-bound once #32330 is fixed.
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd, Eq, Ord, Hash,
         RustcEncodable, RustcDecodable)]
pub enum Issue32330 {
    WontChange,

    /// this region will change from late-bound to early-bound once
    /// #32330 is fixed.
    WillChange {
        /// fn where is region declared
        fn_def_id: DefId,

        /// name of region; duplicates the info in BrNamed but convenient
        /// to have it here, and this code is only temporary
        region_name: ast::Name,
    }
}

// NB: If you change this, you'll probably want to change the corresponding
// AST structure in libsyntax/ast.rs as well.
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

    /// `Box<T>`; this is nominally a struct in the documentation, but is
    /// special-cased internally. For example, it is possible to implicitly
    /// move the contents of a box out of that box, and methods of any type
    /// can have type `Box<Self>`.
    TyBox(Ty<'tcx>),

    /// The pointee of a string slice. Written as `str`.
    TyStr,

    /// An array with the given length. Written as `[T; n]`.
    TyArray(Ty<'tcx>, usize),

    /// The pointee of an array slice.  Written as `[T]`.
    TySlice(Ty<'tcx>),

    /// A raw pointer. Written as `*mut T` or `*const T`
    TyRawPtr(TypeAndMut<'tcx>),

    /// A reference; a pointer with an associated lifetime. Written as
    /// `&a mut T` or `&'a T`.
    TyRef(&'tcx Region, TypeAndMut<'tcx>),

    /// The anonymous type of a function declaration/definition. Each
    /// function has a unique type.
    TyFnDef(DefId, &'tcx Substs<'tcx>, &'tcx BareFnTy<'tcx>),

    /// A pointer to a function.  Written as `fn() -> i32`.
    /// FIXME: This is currently also used to represent the callee of a method;
    /// see ty::MethodCallee etc.
    TyFnPtr(&'tcx BareFnTy<'tcx>),

    /// A trait, defined with `trait`.
    TyDynamic(Binder<&'tcx Slice<ExistentialPredicate<'tcx>>>, &'tcx ty::Region),

    /// The anonymous type of a closure. Used to represent the type of
    /// `|a| a`.
    TyClosure(DefId, ClosureSubsts<'tcx>),

    /// The never type `!`
    TyNever,

    /// A tuple type.  For example, `(i32, bool)`.
    TyTuple(&'tcx Slice<Ty<'tcx>>),

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
///     struct Closure<'l0...'li, T0...Tj, U0...Uk> {
///         upvar0: U0,
///         ...
///         upvark: Uk
///     }
///
/// where 'l0...'li and T0...Tj are the lifetime and type parameters
/// in scope on the function that defined the closure, and U0...Uk are
/// type parameters representing the types of its upvars (borrowed, if
/// appropriate).
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
/// `'b` represents the extent of the closure itself; this is some
/// subset of `foo`, probably just the extent of the call to the to
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
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct ClosureSubsts<'tcx> {
    /// Lifetime and type parameters from the enclosing function,
    /// concatenated with the types of the upvars.
    ///
    /// These are separated out because trans wants to pass them around
    /// when monomorphizing.
    pub substs: &'tcx Substs<'tcx>,
}

impl<'a, 'gcx, 'acx, 'tcx> ClosureSubsts<'tcx> {
    #[inline]
    pub fn upvar_tys(self, def_id: DefId, tcx: TyCtxt<'a, 'gcx, 'acx>) ->
        impl Iterator<Item=Ty<'tcx>> + 'tcx
    {
        let generics = tcx.item_generics(def_id);
        self.substs[self.substs.len()-generics.own_count()..].iter().map(
            |t| t.as_type().expect("unexpected region in upvars"))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub enum ExistentialPredicate<'tcx> {
    // e.g. Iterator
    Trait(ExistentialTraitRef<'tcx>),
    // e.g. Iterator::Item = T
    Projection(ExistentialProjection<'tcx>),
    // e.g. Send
    AutoTrait(DefId),
}

impl<'a, 'gcx, 'tcx> ExistentialPredicate<'tcx> {
    pub fn cmp(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>, other: &Self) -> Ordering {
        use self::ExistentialPredicate::*;
        match (*self, *other) {
            (Trait(_), Trait(_)) => Ordering::Equal,
            (Projection(ref a), Projection(ref b)) => a.sort_key(tcx).cmp(&b.sort_key(tcx)),
            (AutoTrait(ref a), AutoTrait(ref b)) =>
                tcx.lookup_trait_def(*a).def_path_hash.cmp(&tcx.lookup_trait_def(*b).def_path_hash),
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
            _ => None
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
    /// - extracting the def-id from a PolyTraitRef;
    /// - comparing the self type of a PolyTraitRef to see if it is equal to
    ///   a type parameter `X`, since the type `X`  does not reference any regions
    pub fn skip_binder(&self) -> &T {
        &self.0
    }

    pub fn as_ref(&self) -> Binder<&T> {
        ty::Binder(&self.0)
    }

    pub fn map_bound_ref<F,U>(&self, f: F) -> Binder<U>
        where F: FnOnce(&T) -> U
    {
        self.as_ref().map_bound(f)
    }

    pub fn map_bound<F,U>(self, f: F) -> Binder<U>
        where F: FnOnce(T) -> U
    {
        ty::Binder(f(self.0))
    }
}

impl fmt::Debug for TypeFlags {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:x}", self.bits)
    }
}

/// Represents the projection of an associated type. In explicit UFCS
/// form this would be written `<T as Trait<..>>::N`.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct ProjectionTy<'tcx> {
    /// The trait reference `T as Trait<..>`.
    pub trait_ref: ty::TraitRef<'tcx>,

    /// The name `N` of the associated type.
    pub item_name: Name,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct BareFnTy<'tcx> {
    pub unsafety: hir::Unsafety,
    pub abi: abi::Abi,
    /// Signature (inputs and output) of this function type.
    pub sig: PolyFnSig<'tcx>,
}

impl<'tcx> serialize::UseSpecializedDecodable for &'tcx BareFnTy<'tcx> {}

#[derive(Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct ClosureTy<'tcx> {
    pub unsafety: hir::Unsafety,
    pub abi: abi::Abi,
    pub sig: PolyFnSig<'tcx>,
}

/// Signature of a function type, which I have arbitrarily
/// decided to use to refer to the input/output types.
///
/// - `inputs` is the list of arguments and their modes.
/// - `output` is the return type.
/// - `variadic` indicates whether this is a variadic function. (only true for foreign fns)
#[derive(Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct FnSig<'tcx> {
    pub inputs_and_output: &'tcx Slice<Ty<'tcx>>,
    pub variadic: bool
}

impl<'tcx> FnSig<'tcx> {
    pub fn inputs(&self) -> &[Ty<'tcx>] {
        &self.inputs_and_output[..self.inputs_and_output.len() - 1]
    }

    pub fn output(&self) -> Ty<'tcx> {
        self.inputs_and_output[self.inputs_and_output.len() - 1]
    }
}

pub type PolyFnSig<'tcx> = Binder<FnSig<'tcx>>;

impl<'tcx> PolyFnSig<'tcx> {
    pub fn inputs(&self) -> Binder<&[Ty<'tcx>]> {
        Binder(self.skip_binder().inputs())
    }
    pub fn input(&self, index: usize) -> ty::Binder<Ty<'tcx>> {
        self.map_bound_ref(|fn_sig| fn_sig.inputs()[index])
    }
    pub fn output(&self) -> ty::Binder<Ty<'tcx>> {
        self.map_bound_ref(|fn_sig| fn_sig.output().clone())
    }
    pub fn variadic(&self) -> bool {
        self.skip_binder().variadic
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
#[derive(Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Debug, Copy)]
pub struct DebruijnIndex {
    // We maintain the invariant that this is never 0. So 1 indicates
    // the innermost binder. To ensure this, create with `DebruijnIndex::new`.
    pub depth: u32,
}

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
/// not explicity provided.
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
/// [1] http://smallcultfollowing.com/babysteps/blog/2013/10/29/intermingled-parameter-lists/
/// [2] http://smallcultfollowing.com/babysteps/blog/2013/11/04/intermingled-parameter-lists/
#[derive(Clone, PartialEq, Eq, Hash, Copy, RustcEncodable, RustcDecodable)]
pub enum Region {
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

    /// A concrete region naming some statically determined extent
    /// (e.g. an expression or sequence of statements) within the
    /// current function.
    ReScope(region::CodeExtent),

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
}

impl<'tcx> serialize::UseSpecializedDecodable for &'tcx Region {}

#[derive(Copy, Clone, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable, Debug)]
pub struct EarlyBoundRegion {
    pub index: u32,
    pub name: Name,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct TyVid {
    pub index: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct IntVid {
    pub index: u32
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct FloatVid {
    pub index: u32
}

#[derive(Clone, PartialEq, Eq, RustcEncodable, RustcDecodable, Hash, Copy)]
pub struct RegionVid {
    pub index: u32
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, RustcEncodable, RustcDecodable)]
pub struct SkolemizedRegionVid {
    pub index: u32
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
    FreshFloatTy(u32)
}

/// A `ProjectionPredicate` for an `ExistentialTraitRef`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, RustcEncodable, RustcDecodable)]
pub struct ExistentialProjection<'tcx> {
    pub trait_ref: ExistentialTraitRef<'tcx>,
    pub item_name: Name,
    pub ty: Ty<'tcx>
}

pub type PolyExistentialProjection<'tcx> = Binder<ExistentialProjection<'tcx>>;

impl<'a, 'tcx, 'gcx> ExistentialProjection<'tcx> {
    pub fn item_name(&self) -> Name {
        self.item_name // safe to skip the binder to access a name
    }

    pub fn sort_key(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> (u64, InternedString) {
        // We want something here that is stable across crate boundaries.
        // The DefId isn't but the `deterministic_hash` of the corresponding
        // DefPath is.
        let trait_def = tcx.lookup_trait_def(self.trait_ref.def_id);
        let def_path_hash = trait_def.def_path_hash;

        // An `ast::Name` is also not stable (it's just an index into an
        // interning table), so map to the corresponding `InternedString`.
        let item_name = self.item_name.as_str();
        (def_path_hash, item_name)
    }

    pub fn with_self_ty(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>,
                        self_ty: Ty<'tcx>)
                        -> ty::ProjectionPredicate<'tcx>
    {
        // otherwise the escaping regions would be captured by the binders
        assert!(!self_ty.has_escaping_regions());

        ty::ProjectionPredicate {
            projection_ty: ty::ProjectionTy {
                trait_ref: self.trait_ref.with_self_ty(tcx, self_ty),
                item_name: self.item_name
            },
            ty: self.ty
        }
    }
}

impl<'a, 'tcx, 'gcx> PolyExistentialProjection<'tcx> {
    pub fn item_name(&self) -> Name {
        self.skip_binder().item_name()
    }

    pub fn sort_key(&self, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> (u64, InternedString) {
        self.skip_binder().sort_key(tcx)
    }

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

// Region utilities
impl Region {
    pub fn is_bound(&self) -> bool {
        match *self {
            ty::ReEarlyBound(..) => true,
            ty::ReLateBound(..) => true,
            _ => false
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
    pub fn from_depth(&self, depth: u32) -> Region {
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
                flags = flags | TypeFlags::HAS_RE_INFER;
                flags = flags | TypeFlags::KEEP_IN_LOCAL_TCX;
            }
            ty::ReSkolemized(..) => {
                flags = flags | TypeFlags::HAS_RE_INFER;
                flags = flags | TypeFlags::HAS_RE_SKOL;
                flags = flags | TypeFlags::KEEP_IN_LOCAL_TCX;
            }
            ty::ReLateBound(..) => { }
            ty::ReEarlyBound(..) => { flags = flags | TypeFlags::HAS_RE_EARLY_BOUND; }
            ty::ReStatic | ty::ReErased => { }
            _ => { flags = flags | TypeFlags::HAS_FREE_REGIONS; }
        }

        match *self {
            ty::ReStatic | ty::ReEmpty | ty::ReErased => (),
            _ => flags = flags | TypeFlags::HAS_LOCAL_NAMES,
        }

        debug!("type_flags({:?}) = {:?}", self, flags);

        flags
    }
}

// Type utilities
impl<'a, 'gcx, 'tcx> TyS<'tcx> {
    pub fn as_opt_param_ty(&self) -> Option<ty::ParamTy> {
        match self.sty {
            ty::TyParam(ref d) => Some(d.clone()),
            _ => None,
        }
    }

    pub fn is_nil(&self) -> bool {
        match self.sty {
            TyTuple(ref tys) => tys.is_empty(),
            _ => false
        }
    }

    pub fn is_never(&self) -> bool {
        match self.sty {
            TyNever => true,
            _ => false,
        }
    }

    /// Checks whether a type is visibly uninhabited from a particular module.
    /// # Example
    /// ```rust
    /// enum Void {}
    /// mod a {
    ///     pub mod b {
    ///         pub struct SecretlyUninhabited {
    ///             _priv: !,
    ///         }
    ///     }
    /// }
    ///
    /// mod c {
    ///     pub struct AlsoSecretlyUninhabited {
    ///         _priv: Void,
    ///     }
    ///     mod d {
    ///     }
    /// }
    ///
    /// struct Foo {
    ///     x: a::b::SecretlyUninhabited,
    ///     y: c::AlsoSecretlyUninhabited,
    /// }
    /// ```
    /// In this code, the type `Foo` will only be visibly uninhabited inside the
    /// modules b, c and d. This effects pattern-matching on `Foo` or types that
    /// contain `Foo`.
    ///
    /// # Example
    /// ```rust
    /// let foo_result: Result<T, Foo> = ... ;
    /// let Ok(t) = foo_result;
    /// ```
    /// This code should only compile in modules where the uninhabitedness of Foo is
    /// visible.
    pub fn is_uninhabited_from(&self, module: DefId, tcx: TyCtxt<'a, 'gcx, 'tcx>) -> bool {
        let mut visited = FxHashSet::default();
        let forest = self.uninhabited_from(&mut visited, tcx);

        // To check whether this type is uninhabited at all (not just from the
        // given node) you could check whether the forest is empty.
        // ```
        // forest.is_empty()
        // ```
        forest.contains(tcx, module)
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
            _ => false
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
            _ => false
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

    pub fn is_structural(&self) -> bool {
        match self.sty {
            TyAdt(..) | TyTuple(..) | TyArray(..) | TyClosure(..) => true,
            _ => self.is_slice() | self.is_trait()
        }
    }

    #[inline]
    pub fn is_simd(&self) -> bool {
        match self.sty {
            TyAdt(def, _) => def.is_simd(),
            _ => false
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
                def.struct_variant().fields[0].ty(tcx, substs)
            }
            _ => bug!("simd_type called on invalid type")
        }
    }

    pub fn simd_size(&self, _cx: TyCtxt) -> usize {
        match self.sty {
            TyAdt(def, _) => def.struct_variant().fields.len(),
            _ => bug!("simd_size called on invalid type")
        }
    }

    pub fn is_region_ptr(&self) -> bool {
        match self.sty {
            TyRef(..) => true,
            _ => false
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
            _ => return false
        }
    }

    pub fn is_unique(&self) -> bool {
        match self.sty {
            TyBox(_) => true,
            _ => false
        }
    }

    /*
     A scalar type is one that denotes an atomic datum, with no sub-components.
     (A TyRawPtr is scalar because it represents a non-managed pointer, so its
     contents are abstract to rustc.)
    */
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
            _ => false
        }
    }

    pub fn is_integral(&self) -> bool {
        match self.sty {
            TyInfer(IntVar(_)) | TyInt(_) | TyUint(_) => true,
            _ => false
        }
    }

    pub fn is_fresh(&self) -> bool {
        match self.sty {
            TyInfer(FreshTy(_)) => true,
            TyInfer(FreshIntTy(_)) => true,
            TyInfer(FreshFloatTy(_)) => true,
            _ => false
        }
    }

    pub fn is_uint(&self) -> bool {
        match self.sty {
            TyInfer(IntVar(_)) | TyUint(ast::UintTy::Us) => true,
            _ => false
        }
    }

    pub fn is_char(&self) -> bool {
        match self.sty {
            TyChar => true,
            _ => false
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
            _ => false
        }
    }

    pub fn is_machine(&self) -> bool {
        match self.sty {
            TyInt(ast::IntTy::Is) | TyUint(ast::UintTy::Us) => false,
            TyInt(..) | TyUint(..) | TyFloat(..) => true,
            _ => false
        }
    }

    pub fn has_concrete_skeleton(&self) -> bool {
        match self.sty {
            TyParam(_) | TyInfer(_) | TyError => false,
            _ => true,
        }
    }

    // Returns the type and mutability of *ty.
    //
    // The parameter `explicit` indicates if this is an *explicit* dereference.
    // Some types---notably unsafe ptrs---can only be dereferenced explicitly.
    pub fn builtin_deref(&self, explicit: bool, pref: ty::LvaluePreference)
        -> Option<TypeAndMut<'tcx>>
    {
        match self.sty {
            TyBox(ty) => {
                Some(TypeAndMut {
                    ty: ty,
                    mutbl: if pref == ty::PreferMutLvalue {
                        hir::MutMutable
                    } else {
                        hir::MutImmutable
                    },
                })
            },
            TyRef(_, mt) => Some(mt),
            TyRawPtr(mt) if explicit => Some(mt),
            _ => None
        }
    }

    // Returns the type of ty[i]
    pub fn builtin_index(&self) -> Option<Ty<'tcx>> {
        match self.sty {
            TyArray(ty, _) | TySlice(ty) => Some(ty),
            _ => None
        }
    }

    pub fn fn_sig(&self) -> &'tcx PolyFnSig<'tcx> {
        match self.sty {
            TyFnDef(.., ref f) | TyFnPtr(ref f) => &f.sig,
            _ => bug!("Ty::fn_sig() called on non-fn type: {:?}", self)
        }
    }

    /// Returns the ABI of the given function.
    pub fn fn_abi(&self) -> abi::Abi {
        match self.sty {
            TyFnDef(.., ref f) | TyFnPtr(ref f) => f.abi,
            _ => bug!("Ty::fn_abi() called on non-fn type"),
        }
    }

    // Type accessors for substructures of types
    pub fn fn_args(&self) -> ty::Binder<&[Ty<'tcx>]> {
        self.fn_sig().inputs()
    }

    pub fn fn_ret(&self) -> Binder<Ty<'tcx>> {
        self.fn_sig().output()
    }

    pub fn is_fn(&self) -> bool {
        match self.sty {
            TyFnDef(..) | TyFnPtr(_) => true,
            _ => false
        }
    }

    pub fn ty_to_def_id(&self) -> Option<DefId> {
        match self.sty {
            TyDynamic(ref tt, ..) => tt.principal().map(|p| p.def_id()),
            TyAdt(def, _) => Some(def.did),
            TyClosure(id, _) => Some(id),
            _ => None
        }
    }

    pub fn ty_adt_def(&self) -> Option<&'tcx AdtDef> {
        match self.sty {
            TyAdt(adt, _) => Some(adt),
            _ => None
        }
    }

    /// Returns the regions directly referenced from this type (but
    /// not types reachable from this type via `walk_tys`). This
    /// ignores late-bound regions binders.
    pub fn regions(&self) -> Vec<&'tcx ty::Region> {
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
            TyClosure(_, ref substs) => {
                substs.substs.regions().collect()
            }
            TyProjection(ref data) => {
                data.trait_ref.substs.regions().collect()
            }
            TyFnDef(..) |
            TyFnPtr(_) |
            TyBool |
            TyChar |
            TyInt(_) |
            TyUint(_) |
            TyFloat(_) |
            TyBox(_) |
            TyStr |
            TyArray(..) |
            TySlice(_) |
            TyRawPtr(_) |
            TyNever |
            TyTuple(_) |
            TyParam(_) |
            TyInfer(_) |
            TyError => {
                vec![]
            }
        }
    }
}
