//! This module contains `TyKind` and its major components.

#![allow(rustc::usage_of_ty_tykind)]

use crate::infer::canonical::Canonical;
use crate::ty::subst::{GenericArg, InternalSubsts, Subst, SubstsRef};
use crate::ty::visit::ValidateBoundVars;
use crate::ty::InferTy::*;
use crate::ty::{
    self, AdtDef, DefIdTree, Discr, Term, Ty, TyCtxt, TypeFlags, TypeSuperVisitable, TypeVisitable,
    TypeVisitor,
};
use crate::ty::{List, ParamEnv};
use polonius_engine::Atom;
use rustc_data_structures::captures::Captures;
use rustc_data_structures::intern::Interned;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_index::vec::Idx;
use rustc_macros::HashStable;
use rustc_span::symbol::{kw, Symbol};
use rustc_target::abi::VariantIdx;
use rustc_target::spec::abi;
use std::borrow::Cow;
use std::cmp::Ordering;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{ControlFlow, Deref, Range};
use ty::util::IntTypeExt;

use rustc_type_ir::sty::TyKind::*;
use rustc_type_ir::RegionKind as IrRegionKind;
use rustc_type_ir::TyKind as IrTyKind;

// Re-export the `TyKind` from `rustc_type_ir` here for convenience
#[rustc_diagnostic_item = "TyKind"]
pub type TyKind<'tcx> = IrTyKind<TyCtxt<'tcx>>;
pub type RegionKind<'tcx> = IrRegionKind<TyCtxt<'tcx>>;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable, Lift)]
pub struct TypeAndMut<'tcx> {
    pub ty: Ty<'tcx>,
    pub mutbl: hir::Mutability,
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, TyEncodable, TyDecodable, Copy)]
#[derive(HashStable)]
/// A "free" region `fr` can be interpreted as "some region
/// at least as big as the scope `fr.scope`".
pub struct FreeRegion {
    pub scope: DefId,
    pub bound_region: BoundRegionKind,
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash, TyEncodable, TyDecodable, Copy)]
#[derive(HashStable)]
pub enum BoundRegionKind {
    /// An anonymous region parameter for a given fn (&T)
    BrAnon(u32),

    /// Named region parameters for functions (a in &'a T)
    ///
    /// The `DefId` is needed to distinguish free regions in
    /// the event of shadowing.
    BrNamed(DefId, Symbol),

    /// Anonymous region for the implicit env pointer parameter
    /// to a closure
    BrEnv,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable, Debug, PartialOrd, Ord)]
#[derive(HashStable)]
pub struct BoundRegion {
    pub var: BoundVar,
    pub kind: BoundRegionKind,
}

impl BoundRegionKind {
    pub fn is_named(&self) -> bool {
        match *self {
            BoundRegionKind::BrNamed(_, name) => name != kw::UnderscoreLifetime,
            _ => false,
        }
    }
}

pub trait Article {
    fn article(&self) -> &'static str;
}

impl<'tcx> Article for TyKind<'tcx> {
    /// Get the article ("a" or "an") to use with this type.
    fn article(&self) -> &'static str {
        match self {
            Int(_) | Float(_) | Array(_, _) => "an",
            Adt(def, _) if def.is_enum() => "an",
            // This should never happen, but ICEing and causing the user's code
            // to not compile felt too harsh.
            Error(_) => "a",
            _ => "a",
        }
    }
}

// `TyKind` is used a lot. Make sure it doesn't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
static_assert_size!(TyKind<'_>, 32);

/// A closure can be modeled as a struct that looks like:
/// ```ignore (illustrative)
/// struct Closure<'l0...'li, T0...Tj, CK, CS, U>(...U);
/// ```
/// where:
///
/// - 'l0...'li and T0...Tj are the generic parameters
///   in scope on the function that defined the closure,
/// - CK represents the *closure kind* (Fn vs FnMut vs FnOnce). This
///   is rather hackily encoded via a scalar type. See
///   `Ty::to_opt_closure_kind` for details.
/// - CS represents the *closure signature*, representing as a `fn()`
///   type. For example, `fn(u32, u32) -> u32` would mean that the closure
///   implements `CK<(u32, u32), Output = u32>`, where `CK` is the trait
///   specified above.
/// - U is a type parameter representing the types of its upvars, tupled up
///   (borrowed, if appropriate; that is, if a U field represents a by-ref upvar,
///    and the up-var has the type `Foo`, then that field of U will be `&Foo`).
///
/// So, for example, given this function:
/// ```ignore (illustrative)
/// fn foo<'a, T>(data: &'a mut T) {
///      do(|| data.count += 1)
/// }
/// ```
/// the type of the closure would be something like:
/// ```ignore (illustrative)
/// struct Closure<'a, T, U>(...U);
/// ```
/// Note that the type of the upvar is not specified in the struct.
/// You may wonder how the impl would then be able to use the upvar,
/// if it doesn't know it's type? The answer is that the impl is
/// (conceptually) not fully generic over Closure but rather tied to
/// instances with the expected upvar types:
/// ```ignore (illustrative)
/// impl<'b, 'a, T> FnMut() for Closure<'a, T, (&'b mut &'a mut T,)> {
///     ...
/// }
/// ```
/// You can see that the *impl* fully specified the type of the upvar
/// and thus knows full well that `data` has type `&'b mut &'a mut T`.
/// (Here, I am assuming that `data` is mut-borrowed.)
///
/// Now, the last question you may ask is: Why include the upvar types
/// in an extra type parameter? The reason for this design is that the
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
/// original function then? The answer is that codegen may need them
/// when monomorphizing, and they may not appear in the upvars. A
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
/// Generators are handled similarly in `GeneratorSubsts`.  The set of
/// type parameters is similar, but `CK` and `CS` are replaced by the
/// following type parameters:
///
/// * `GS`: The generator's "resume type", which is the type of the
///   argument passed to `resume`, and the type of `yield` expressions
///   inside the generator.
/// * `GY`: The "yield type", which is the type of values passed to
///   `yield` inside the generator.
/// * `GR`: The "return type", which is the type of value returned upon
///   completion of the generator.
/// * `GW`: The "generator witness".
#[derive(Copy, Clone, Debug, TypeFoldable, TypeVisitable)]
pub struct ClosureSubsts<'tcx> {
    /// Lifetime and type parameters from the enclosing function,
    /// concatenated with a tuple containing the types of the upvars.
    ///
    /// These are separated out because codegen wants to pass them around
    /// when monomorphizing.
    pub substs: SubstsRef<'tcx>,
}

/// Struct returned by `split()`.
pub struct ClosureSubstsParts<'tcx, T> {
    pub parent_substs: &'tcx [GenericArg<'tcx>],
    pub closure_kind_ty: T,
    pub closure_sig_as_fn_ptr_ty: T,
    pub tupled_upvars_ty: T,
}

impl<'tcx> ClosureSubsts<'tcx> {
    /// Construct `ClosureSubsts` from `ClosureSubstsParts`, containing `Substs`
    /// for the closure parent, alongside additional closure-specific components.
    pub fn new(
        tcx: TyCtxt<'tcx>,
        parts: ClosureSubstsParts<'tcx, Ty<'tcx>>,
    ) -> ClosureSubsts<'tcx> {
        ClosureSubsts {
            substs: tcx.mk_substs(
                parts.parent_substs.iter().copied().chain(
                    [parts.closure_kind_ty, parts.closure_sig_as_fn_ptr_ty, parts.tupled_upvars_ty]
                        .iter()
                        .map(|&ty| ty.into()),
                ),
            ),
        }
    }

    /// Divides the closure substs into their respective components.
    /// The ordering assumed here must match that used by `ClosureSubsts::new` above.
    fn split(self) -> ClosureSubstsParts<'tcx, GenericArg<'tcx>> {
        match self.substs[..] {
            [
                ref parent_substs @ ..,
                closure_kind_ty,
                closure_sig_as_fn_ptr_ty,
                tupled_upvars_ty,
            ] => ClosureSubstsParts {
                parent_substs,
                closure_kind_ty,
                closure_sig_as_fn_ptr_ty,
                tupled_upvars_ty,
            },
            _ => bug!("closure substs missing synthetics"),
        }
    }

    /// Returns `true` only if enough of the synthetic types are known to
    /// allow using all of the methods on `ClosureSubsts` without panicking.
    ///
    /// Used primarily by `ty::print::pretty` to be able to handle closure
    /// types that haven't had their synthetic types substituted in.
    pub fn is_valid(self) -> bool {
        self.substs.len() >= 3
            && matches!(self.split().tupled_upvars_ty.expect_ty().kind(), Tuple(_))
    }

    /// Returns the substitutions of the closure's parent.
    pub fn parent_substs(self) -> &'tcx [GenericArg<'tcx>] {
        self.split().parent_substs
    }

    /// Returns an iterator over the list of types of captured paths by the closure.
    /// In case there was a type error in figuring out the types of the captured path, an
    /// empty iterator is returned.
    #[inline]
    pub fn upvar_tys(self) -> impl Iterator<Item = Ty<'tcx>> + 'tcx {
        match self.tupled_upvars_ty().kind() {
            TyKind::Error(_) => None,
            TyKind::Tuple(..) => Some(self.tupled_upvars_ty().tuple_fields()),
            TyKind::Infer(_) => bug!("upvar_tys called before capture types are inferred"),
            ty => bug!("Unexpected representation of upvar types tuple {:?}", ty),
        }
        .into_iter()
        .flatten()
    }

    /// Returns the tuple type representing the upvars for this closure.
    #[inline]
    pub fn tupled_upvars_ty(self) -> Ty<'tcx> {
        self.split().tupled_upvars_ty.expect_ty()
    }

    /// Returns the closure kind for this closure; may return a type
    /// variable during inference. To get the closure kind during
    /// inference, use `infcx.closure_kind(substs)`.
    pub fn kind_ty(self) -> Ty<'tcx> {
        self.split().closure_kind_ty.expect_ty()
    }

    /// Returns the `fn` pointer type representing the closure signature for this
    /// closure.
    // FIXME(eddyb) this should be unnecessary, as the shallowly resolved
    // type is known at the time of the creation of `ClosureSubsts`,
    // see `rustc_typeck::check::closure`.
    pub fn sig_as_fn_ptr_ty(self) -> Ty<'tcx> {
        self.split().closure_sig_as_fn_ptr_ty.expect_ty()
    }

    /// Returns the closure kind for this closure; only usable outside
    /// of an inference context, because in that context we know that
    /// there are no type variables.
    ///
    /// If you have an inference context, use `infcx.closure_kind()`.
    pub fn kind(self) -> ty::ClosureKind {
        self.kind_ty().to_opt_closure_kind().unwrap()
    }

    /// Extracts the signature from the closure.
    pub fn sig(self) -> ty::PolyFnSig<'tcx> {
        let ty = self.sig_as_fn_ptr_ty();
        match ty.kind() {
            ty::FnPtr(sig) => *sig,
            _ => bug!("closure_sig_as_fn_ptr_ty is not a fn-ptr: {:?}", ty.kind()),
        }
    }

    pub fn print_as_impl_trait(self) -> ty::print::PrintClosureAsImpl<'tcx> {
        ty::print::PrintClosureAsImpl { closure: self }
    }
}

/// Similar to `ClosureSubsts`; see the above documentation for more.
#[derive(Copy, Clone, Debug, TypeFoldable, TypeVisitable)]
pub struct GeneratorSubsts<'tcx> {
    pub substs: SubstsRef<'tcx>,
}

pub struct GeneratorSubstsParts<'tcx, T> {
    pub parent_substs: &'tcx [GenericArg<'tcx>],
    pub resume_ty: T,
    pub yield_ty: T,
    pub return_ty: T,
    pub witness: T,
    pub tupled_upvars_ty: T,
}

impl<'tcx> GeneratorSubsts<'tcx> {
    /// Construct `GeneratorSubsts` from `GeneratorSubstsParts`, containing `Substs`
    /// for the generator parent, alongside additional generator-specific components.
    pub fn new(
        tcx: TyCtxt<'tcx>,
        parts: GeneratorSubstsParts<'tcx, Ty<'tcx>>,
    ) -> GeneratorSubsts<'tcx> {
        GeneratorSubsts {
            substs: tcx.mk_substs(
                parts.parent_substs.iter().copied().chain(
                    [
                        parts.resume_ty,
                        parts.yield_ty,
                        parts.return_ty,
                        parts.witness,
                        parts.tupled_upvars_ty,
                    ]
                    .iter()
                    .map(|&ty| ty.into()),
                ),
            ),
        }
    }

    /// Divides the generator substs into their respective components.
    /// The ordering assumed here must match that used by `GeneratorSubsts::new` above.
    fn split(self) -> GeneratorSubstsParts<'tcx, GenericArg<'tcx>> {
        match self.substs[..] {
            [ref parent_substs @ .., resume_ty, yield_ty, return_ty, witness, tupled_upvars_ty] => {
                GeneratorSubstsParts {
                    parent_substs,
                    resume_ty,
                    yield_ty,
                    return_ty,
                    witness,
                    tupled_upvars_ty,
                }
            }
            _ => bug!("generator substs missing synthetics"),
        }
    }

    /// Returns `true` only if enough of the synthetic types are known to
    /// allow using all of the methods on `GeneratorSubsts` without panicking.
    ///
    /// Used primarily by `ty::print::pretty` to be able to handle generator
    /// types that haven't had their synthetic types substituted in.
    pub fn is_valid(self) -> bool {
        self.substs.len() >= 5
            && matches!(self.split().tupled_upvars_ty.expect_ty().kind(), Tuple(_))
    }

    /// Returns the substitutions of the generator's parent.
    pub fn parent_substs(self) -> &'tcx [GenericArg<'tcx>] {
        self.split().parent_substs
    }

    /// This describes the types that can be contained in a generator.
    /// It will be a type variable initially and unified in the last stages of typeck of a body.
    /// It contains a tuple of all the types that could end up on a generator frame.
    /// The state transformation MIR pass may only produce layouts which mention types
    /// in this tuple. Upvars are not counted here.
    pub fn witness(self) -> Ty<'tcx> {
        self.split().witness.expect_ty()
    }

    /// Returns an iterator over the list of types of captured paths by the generator.
    /// In case there was a type error in figuring out the types of the captured path, an
    /// empty iterator is returned.
    #[inline]
    pub fn upvar_tys(self) -> impl Iterator<Item = Ty<'tcx>> + 'tcx {
        match self.tupled_upvars_ty().kind() {
            TyKind::Error(_) => None,
            TyKind::Tuple(..) => Some(self.tupled_upvars_ty().tuple_fields()),
            TyKind::Infer(_) => bug!("upvar_tys called before capture types are inferred"),
            ty => bug!("Unexpected representation of upvar types tuple {:?}", ty),
        }
        .into_iter()
        .flatten()
    }

    /// Returns the tuple type representing the upvars for this generator.
    #[inline]
    pub fn tupled_upvars_ty(self) -> Ty<'tcx> {
        self.split().tupled_upvars_ty.expect_ty()
    }

    /// Returns the type representing the resume type of the generator.
    pub fn resume_ty(self) -> Ty<'tcx> {
        self.split().resume_ty.expect_ty()
    }

    /// Returns the type representing the yield type of the generator.
    pub fn yield_ty(self) -> Ty<'tcx> {
        self.split().yield_ty.expect_ty()
    }

    /// Returns the type representing the return type of the generator.
    pub fn return_ty(self) -> Ty<'tcx> {
        self.split().return_ty.expect_ty()
    }

    /// Returns the "generator signature", which consists of its yield
    /// and return types.
    ///
    /// N.B., some bits of the code prefers to see this wrapped in a
    /// binder, but it never contains bound regions. Probably this
    /// function should be removed.
    pub fn poly_sig(self) -> PolyGenSig<'tcx> {
        ty::Binder::dummy(self.sig())
    }

    /// Returns the "generator signature", which consists of its resume, yield
    /// and return types.
    pub fn sig(self) -> GenSig<'tcx> {
        ty::GenSig {
            resume_ty: self.resume_ty(),
            yield_ty: self.yield_ty(),
            return_ty: self.return_ty(),
        }
    }
}

impl<'tcx> GeneratorSubsts<'tcx> {
    /// Generator has not been resumed yet.
    pub const UNRESUMED: usize = 0;
    /// Generator has returned or is completed.
    pub const RETURNED: usize = 1;
    /// Generator has been poisoned.
    pub const POISONED: usize = 2;

    const UNRESUMED_NAME: &'static str = "Unresumed";
    const RETURNED_NAME: &'static str = "Returned";
    const POISONED_NAME: &'static str = "Panicked";

    /// The valid variant indices of this generator.
    #[inline]
    pub fn variant_range(&self, def_id: DefId, tcx: TyCtxt<'tcx>) -> Range<VariantIdx> {
        // FIXME requires optimized MIR
        let num_variants = tcx.generator_layout(def_id).unwrap().variant_fields.len();
        VariantIdx::new(0)..VariantIdx::new(num_variants)
    }

    /// The discriminant for the given variant. Panics if the `variant_index` is
    /// out of range.
    #[inline]
    pub fn discriminant_for_variant(
        &self,
        def_id: DefId,
        tcx: TyCtxt<'tcx>,
        variant_index: VariantIdx,
    ) -> Discr<'tcx> {
        // Generators don't support explicit discriminant values, so they are
        // the same as the variant index.
        assert!(self.variant_range(def_id, tcx).contains(&variant_index));
        Discr { val: variant_index.as_usize() as u128, ty: self.discr_ty(tcx) }
    }

    /// The set of all discriminants for the generator, enumerated with their
    /// variant indices.
    #[inline]
    pub fn discriminants(
        self,
        def_id: DefId,
        tcx: TyCtxt<'tcx>,
    ) -> impl Iterator<Item = (VariantIdx, Discr<'tcx>)> + Captures<'tcx> {
        self.variant_range(def_id, tcx).map(move |index| {
            (index, Discr { val: index.as_usize() as u128, ty: self.discr_ty(tcx) })
        })
    }

    /// Calls `f` with a reference to the name of the enumerator for the given
    /// variant `v`.
    pub fn variant_name(v: VariantIdx) -> Cow<'static, str> {
        match v.as_usize() {
            Self::UNRESUMED => Cow::from(Self::UNRESUMED_NAME),
            Self::RETURNED => Cow::from(Self::RETURNED_NAME),
            Self::POISONED => Cow::from(Self::POISONED_NAME),
            _ => Cow::from(format!("Suspend{}", v.as_usize() - 3)),
        }
    }

    /// The type of the state discriminant used in the generator type.
    #[inline]
    pub fn discr_ty(&self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        tcx.types.u32
    }

    /// This returns the types of the MIR locals which had to be stored across suspension points.
    /// It is calculated in rustc_mir_transform::generator::StateTransform.
    /// All the types here must be in the tuple in GeneratorInterior.
    ///
    /// The locals are grouped by their variant number. Note that some locals may
    /// be repeated in multiple variants.
    #[inline]
    pub fn state_tys(
        self,
        def_id: DefId,
        tcx: TyCtxt<'tcx>,
    ) -> impl Iterator<Item = impl Iterator<Item = Ty<'tcx>> + Captures<'tcx>> {
        let layout = tcx.generator_layout(def_id).unwrap();
        layout.variant_fields.iter().map(move |variant| {
            variant
                .iter()
                .map(move |field| EarlyBinder(layout.field_tys[*field]).subst(tcx, self.substs))
        })
    }

    /// This is the types of the fields of a generator which are not stored in a
    /// variant.
    #[inline]
    pub fn prefix_tys(self) -> impl Iterator<Item = Ty<'tcx>> {
        self.upvar_tys()
    }
}

#[derive(Debug, Copy, Clone, HashStable)]
pub enum UpvarSubsts<'tcx> {
    Closure(SubstsRef<'tcx>),
    Generator(SubstsRef<'tcx>),
}

impl<'tcx> UpvarSubsts<'tcx> {
    /// Returns an iterator over the list of types of captured paths by the closure/generator.
    /// In case there was a type error in figuring out the types of the captured path, an
    /// empty iterator is returned.
    #[inline]
    pub fn upvar_tys(self) -> impl Iterator<Item = Ty<'tcx>> + 'tcx {
        let tupled_tys = match self {
            UpvarSubsts::Closure(substs) => substs.as_closure().tupled_upvars_ty(),
            UpvarSubsts::Generator(substs) => substs.as_generator().tupled_upvars_ty(),
        };

        match tupled_tys.kind() {
            TyKind::Error(_) => None,
            TyKind::Tuple(..) => Some(self.tupled_upvars_ty().tuple_fields()),
            TyKind::Infer(_) => bug!("upvar_tys called before capture types are inferred"),
            ty => bug!("Unexpected representation of upvar types tuple {:?}", ty),
        }
        .into_iter()
        .flatten()
    }

    #[inline]
    pub fn tupled_upvars_ty(self) -> Ty<'tcx> {
        match self {
            UpvarSubsts::Closure(substs) => substs.as_closure().tupled_upvars_ty(),
            UpvarSubsts::Generator(substs) => substs.as_generator().tupled_upvars_ty(),
        }
    }
}

/// An inline const is modeled like
/// ```ignore (illustrative)
/// const InlineConst<'l0...'li, T0...Tj, R>: R;
/// ```
/// where:
///
/// - 'l0...'li and T0...Tj are the generic parameters
///   inherited from the item that defined the inline const,
/// - R represents the type of the constant.
///
/// When the inline const is instantiated, `R` is substituted as the actual inferred
/// type of the constant. The reason that `R` is represented as an extra type parameter
/// is the same reason that [`ClosureSubsts`] have `CS` and `U` as type parameters:
/// inline const can reference lifetimes that are internal to the creating function.
#[derive(Copy, Clone, Debug, TypeFoldable, TypeVisitable)]
pub struct InlineConstSubsts<'tcx> {
    /// Generic parameters from the enclosing item,
    /// concatenated with the inferred type of the constant.
    pub substs: SubstsRef<'tcx>,
}

/// Struct returned by `split()`.
pub struct InlineConstSubstsParts<'tcx, T> {
    pub parent_substs: &'tcx [GenericArg<'tcx>],
    pub ty: T,
}

impl<'tcx> InlineConstSubsts<'tcx> {
    /// Construct `InlineConstSubsts` from `InlineConstSubstsParts`.
    pub fn new(
        tcx: TyCtxt<'tcx>,
        parts: InlineConstSubstsParts<'tcx, Ty<'tcx>>,
    ) -> InlineConstSubsts<'tcx> {
        InlineConstSubsts {
            substs: tcx.mk_substs(
                parts.parent_substs.iter().copied().chain(std::iter::once(parts.ty.into())),
            ),
        }
    }

    /// Divides the inline const substs into their respective components.
    /// The ordering assumed here must match that used by `InlineConstSubsts::new` above.
    fn split(self) -> InlineConstSubstsParts<'tcx, GenericArg<'tcx>> {
        match self.substs[..] {
            [ref parent_substs @ .., ty] => InlineConstSubstsParts { parent_substs, ty },
            _ => bug!("inline const substs missing synthetics"),
        }
    }

    /// Returns the substitutions of the inline const's parent.
    pub fn parent_substs(self) -> &'tcx [GenericArg<'tcx>] {
        self.split().parent_substs
    }

    /// Returns the type of this inline const.
    pub fn ty(self) -> Ty<'tcx> {
        self.split().ty.expect_ty()
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Ord, Eq, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable)]
pub enum ExistentialPredicate<'tcx> {
    /// E.g., `Iterator`.
    Trait(ExistentialTraitRef<'tcx>),
    /// E.g., `Iterator::Item = T`.
    Projection(ExistentialProjection<'tcx>),
    /// E.g., `Send`.
    AutoTrait(DefId),
}

impl<'tcx> ExistentialPredicate<'tcx> {
    /// Compares via an ordering that will not change if modules are reordered or other changes are
    /// made to the tree. In particular, this ordering is preserved across incremental compilations.
    pub fn stable_cmp(&self, tcx: TyCtxt<'tcx>, other: &Self) -> Ordering {
        use self::ExistentialPredicate::*;
        match (*self, *other) {
            (Trait(_), Trait(_)) => Ordering::Equal,
            (Projection(ref a), Projection(ref b)) => {
                tcx.def_path_hash(a.item_def_id).cmp(&tcx.def_path_hash(b.item_def_id))
            }
            (AutoTrait(ref a), AutoTrait(ref b)) => {
                tcx.def_path_hash(*a).cmp(&tcx.def_path_hash(*b))
            }
            (Trait(_), _) => Ordering::Less,
            (Projection(_), Trait(_)) => Ordering::Greater,
            (Projection(_), _) => Ordering::Less,
            (AutoTrait(_), _) => Ordering::Greater,
        }
    }
}

impl<'tcx> Binder<'tcx, ExistentialPredicate<'tcx>> {
    pub fn with_self_ty(&self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> ty::Predicate<'tcx> {
        use crate::ty::ToPredicate;
        match self.skip_binder() {
            ExistentialPredicate::Trait(tr) => {
                self.rebind(tr).with_self_ty(tcx, self_ty).without_const().to_predicate(tcx)
            }
            ExistentialPredicate::Projection(p) => {
                self.rebind(p.with_self_ty(tcx, self_ty)).to_predicate(tcx)
            }
            ExistentialPredicate::AutoTrait(did) => {
                let trait_ref = self.rebind(ty::TraitRef {
                    def_id: did,
                    substs: tcx.mk_substs_trait(self_ty, &[]),
                });
                trait_ref.without_const().to_predicate(tcx)
            }
        }
    }
}

impl<'tcx> List<ty::Binder<'tcx, ExistentialPredicate<'tcx>>> {
    /// Returns the "principal `DefId`" of this set of existential predicates.
    ///
    /// A Rust trait object type consists (in addition to a lifetime bound)
    /// of a set of trait bounds, which are separated into any number
    /// of auto-trait bounds, and at most one non-auto-trait bound. The
    /// non-auto-trait bound is called the "principal" of the trait
    /// object.
    ///
    /// Only the principal can have methods or type parameters (because
    /// auto traits can have neither of them). This is important, because
    /// it means the auto traits can be treated as an unordered set (methods
    /// would force an order for the vtable, while relating traits with
    /// type parameters without knowing the order to relate them in is
    /// a rather non-trivial task).
    ///
    /// For example, in the trait object `dyn fmt::Debug + Sync`, the
    /// principal bound is `Some(fmt::Debug)`, while the auto-trait bounds
    /// are the set `{Sync}`.
    ///
    /// It is also possible to have a "trivial" trait object that
    /// consists only of auto traits, with no principal - for example,
    /// `dyn Send + Sync`. In that case, the set of auto-trait bounds
    /// is `{Send, Sync}`, while there is no principal. These trait objects
    /// have a "trivial" vtable consisting of just the size, alignment,
    /// and destructor.
    pub fn principal(&self) -> Option<ty::Binder<'tcx, ExistentialTraitRef<'tcx>>> {
        self[0]
            .map_bound(|this| match this {
                ExistentialPredicate::Trait(tr) => Some(tr),
                _ => None,
            })
            .transpose()
    }

    pub fn principal_def_id(&self) -> Option<DefId> {
        self.principal().map(|trait_ref| trait_ref.skip_binder().def_id)
    }

    #[inline]
    pub fn projection_bounds<'a>(
        &'a self,
    ) -> impl Iterator<Item = ty::Binder<'tcx, ExistentialProjection<'tcx>>> + 'a {
        self.iter().filter_map(|predicate| {
            predicate
                .map_bound(|pred| match pred {
                    ExistentialPredicate::Projection(projection) => Some(projection),
                    _ => None,
                })
                .transpose()
        })
    }

    #[inline]
    pub fn auto_traits<'a>(&'a self) -> impl Iterator<Item = DefId> + Captures<'tcx> + 'a {
        self.iter().filter_map(|predicate| match predicate.skip_binder() {
            ExistentialPredicate::AutoTrait(did) => Some(did),
            _ => None,
        })
    }
}

/// A complete reference to a trait. These take numerous guises in syntax,
/// but perhaps the most recognizable form is in a where-clause:
/// ```ignore (illustrative)
/// T: Foo<U>
/// ```
/// This would be represented by a trait-reference where the `DefId` is the
/// `DefId` for the trait `Foo` and the substs define `T` as parameter 0,
/// and `U` as parameter 1.
///
/// Trait references also appear in object types like `Foo<U>`, but in
/// that case the `Self` parameter is absent from the substitutions.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable)]
pub struct TraitRef<'tcx> {
    pub def_id: DefId,
    pub substs: SubstsRef<'tcx>,
}

impl<'tcx> TraitRef<'tcx> {
    pub fn new(def_id: DefId, substs: SubstsRef<'tcx>) -> TraitRef<'tcx> {
        TraitRef { def_id, substs }
    }

    /// Returns a `TraitRef` of the form `P0: Foo<P1..Pn>` where `Pi`
    /// are the parameters defined on trait.
    pub fn identity(tcx: TyCtxt<'tcx>, def_id: DefId) -> Binder<'tcx, TraitRef<'tcx>> {
        ty::Binder::dummy(TraitRef {
            def_id,
            substs: InternalSubsts::identity_for_item(tcx, def_id),
        })
    }

    #[inline]
    pub fn self_ty(&self) -> Ty<'tcx> {
        self.substs.type_at(0)
    }

    pub fn from_method(
        tcx: TyCtxt<'tcx>,
        trait_id: DefId,
        substs: SubstsRef<'tcx>,
    ) -> ty::TraitRef<'tcx> {
        let defs = tcx.generics_of(trait_id);
        ty::TraitRef { def_id: trait_id, substs: tcx.intern_substs(&substs[..defs.params.len()]) }
    }
}

pub type PolyTraitRef<'tcx> = Binder<'tcx, TraitRef<'tcx>>;

impl<'tcx> PolyTraitRef<'tcx> {
    pub fn self_ty(&self) -> Binder<'tcx, Ty<'tcx>> {
        self.map_bound_ref(|tr| tr.self_ty())
    }

    pub fn def_id(&self) -> DefId {
        self.skip_binder().def_id
    }

    pub fn to_poly_trait_predicate(&self) -> ty::PolyTraitPredicate<'tcx> {
        self.map_bound(|trait_ref| ty::TraitPredicate {
            trait_ref,
            constness: ty::BoundConstness::NotConst,
            polarity: ty::ImplPolarity::Positive,
        })
    }

    /// Same as [`PolyTraitRef::to_poly_trait_predicate`] but sets a negative polarity instead.
    pub fn to_poly_trait_predicate_negative_polarity(&self) -> ty::PolyTraitPredicate<'tcx> {
        self.map_bound(|trait_ref| ty::TraitPredicate {
            trait_ref,
            constness: ty::BoundConstness::NotConst,
            polarity: ty::ImplPolarity::Negative,
        })
    }
}

/// An existential reference to a trait, where `Self` is erased.
/// For example, the trait object `Trait<'a, 'b, X, Y>` is:
/// ```ignore (illustrative)
/// exists T. T: Trait<'a, 'b, X, Y>
/// ```
/// The substitutions don't include the erased `Self`, only trait
/// type and lifetime parameters (`[X, Y]` and `['a, 'b]` above).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable)]
pub struct ExistentialTraitRef<'tcx> {
    pub def_id: DefId,
    pub substs: SubstsRef<'tcx>,
}

impl<'tcx> ExistentialTraitRef<'tcx> {
    pub fn erase_self_ty(
        tcx: TyCtxt<'tcx>,
        trait_ref: ty::TraitRef<'tcx>,
    ) -> ty::ExistentialTraitRef<'tcx> {
        // Assert there is a Self.
        trait_ref.substs.type_at(0);

        ty::ExistentialTraitRef {
            def_id: trait_ref.def_id,
            substs: tcx.intern_substs(&trait_ref.substs[1..]),
        }
    }

    /// Object types don't have a self type specified. Therefore, when
    /// we convert the principal trait-ref into a normal trait-ref,
    /// you must give *some* self type. A common choice is `mk_err()`
    /// or some placeholder type.
    pub fn with_self_ty(&self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> ty::TraitRef<'tcx> {
        // otherwise the escaping vars would be captured by the binder
        // debug_assert!(!self_ty.has_escaping_bound_vars());

        ty::TraitRef { def_id: self.def_id, substs: tcx.mk_substs_trait(self_ty, self.substs) }
    }
}

pub type PolyExistentialTraitRef<'tcx> = Binder<'tcx, ExistentialTraitRef<'tcx>>;

impl<'tcx> PolyExistentialTraitRef<'tcx> {
    pub fn def_id(&self) -> DefId {
        self.skip_binder().def_id
    }

    /// Object types don't have a self type specified. Therefore, when
    /// we convert the principal trait-ref into a normal trait-ref,
    /// you must give *some* self type. A common choice is `mk_err()`
    /// or some placeholder type.
    pub fn with_self_ty(&self, tcx: TyCtxt<'tcx>, self_ty: Ty<'tcx>) -> ty::PolyTraitRef<'tcx> {
        self.map_bound(|trait_ref| trait_ref.with_self_ty(tcx, self_ty))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[derive(Encodable, Decodable, HashStable)]
pub struct EarlyBinder<T>(pub T);

impl<T> EarlyBinder<T> {
    pub fn as_ref(&self) -> EarlyBinder<&T> {
        EarlyBinder(&self.0)
    }

    pub fn map_bound_ref<F, U>(&self, f: F) -> EarlyBinder<U>
    where
        F: FnOnce(&T) -> U,
    {
        self.as_ref().map_bound(f)
    }

    pub fn map_bound<F, U>(self, f: F) -> EarlyBinder<U>
    where
        F: FnOnce(T) -> U,
    {
        let value = f(self.0);
        EarlyBinder(value)
    }

    pub fn try_map_bound<F, U, E>(self, f: F) -> Result<EarlyBinder<U>, E>
    where
        F: FnOnce(T) -> Result<U, E>,
    {
        let value = f(self.0)?;
        Ok(EarlyBinder(value))
    }

    pub fn rebind<U>(&self, value: U) -> EarlyBinder<U> {
        EarlyBinder(value)
    }
}

impl<T> EarlyBinder<Option<T>> {
    pub fn transpose(self) -> Option<EarlyBinder<T>> {
        self.0.map(|v| EarlyBinder(v))
    }
}

impl<T, U> EarlyBinder<(T, U)> {
    pub fn transpose_tuple2(self) -> (EarlyBinder<T>, EarlyBinder<U>) {
        (EarlyBinder(self.0.0), EarlyBinder(self.0.1))
    }
}

pub struct EarlyBinderIter<T> {
    t: T,
}

impl<T: IntoIterator> EarlyBinder<T> {
    pub fn transpose_iter(self) -> EarlyBinderIter<T::IntoIter> {
        EarlyBinderIter { t: self.0.into_iter() }
    }
}

impl<T: Iterator> Iterator for EarlyBinderIter<T> {
    type Item = EarlyBinder<T::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.t.next().map(|i| EarlyBinder(i))
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub enum BoundVariableKind {
    Ty(BoundTyKind),
    Region(BoundRegionKind),
    Const,
}

impl BoundVariableKind {
    pub fn expect_region(self) -> BoundRegionKind {
        match self {
            BoundVariableKind::Region(lt) => lt,
            _ => bug!("expected a region, but found another kind"),
        }
    }

    pub fn expect_ty(self) -> BoundTyKind {
        match self {
            BoundVariableKind::Ty(ty) => ty,
            _ => bug!("expected a type, but found another kind"),
        }
    }

    pub fn expect_const(self) {
        match self {
            BoundVariableKind::Const => (),
            _ => bug!("expected a const, but found another kind"),
        }
    }
}

/// Binder is a binder for higher-ranked lifetimes or types. It is part of the
/// compiler's representation for things like `for<'a> Fn(&'a isize)`
/// (which would be represented by the type `PolyTraitRef ==
/// Binder<'tcx, TraitRef>`). Note that when we instantiate,
/// erase, or otherwise "discharge" these bound vars, we change the
/// type from `Binder<'tcx, T>` to just `T` (see
/// e.g., `liberate_late_bound_regions`).
///
/// `Decodable` and `Encodable` are implemented for `Binder<T>` using the `impl_binder_encode_decode!` macro.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[derive(HashStable)]
pub struct Binder<'tcx, T>(T, &'tcx List<BoundVariableKind>);

impl<'tcx, T> Binder<'tcx, T>
where
    T: TypeVisitable<'tcx>,
{
    /// Wraps `value` in a binder, asserting that `value` does not
    /// contain any bound vars that would be bound by the
    /// binder. This is commonly used to 'inject' a value T into a
    /// different binding level.
    pub fn dummy(value: T) -> Binder<'tcx, T> {
        assert!(!value.has_escaping_bound_vars());
        Binder(value, ty::List::empty())
    }

    pub fn bind_with_vars(value: T, vars: &'tcx List<BoundVariableKind>) -> Binder<'tcx, T> {
        if cfg!(debug_assertions) {
            let mut validator = ValidateBoundVars::new(vars);
            value.visit_with(&mut validator);
        }
        Binder(value, vars)
    }
}

impl<'tcx, T> Binder<'tcx, T> {
    /// Skips the binder and returns the "bound" value. This is a
    /// risky thing to do because it's easy to get confused about
    /// De Bruijn indices and the like. It is usually better to
    /// discharge the binder using `no_bound_vars` or
    /// `replace_late_bound_regions` or something like
    /// that. `skip_binder` is only valid when you are either
    /// extracting data that has nothing to do with bound vars, you
    /// are doing some sort of test that does not involve bound
    /// regions, or you are being very careful about your depth
    /// accounting.
    ///
    /// Some examples where `skip_binder` is reasonable:
    ///
    /// - extracting the `DefId` from a PolyTraitRef;
    /// - comparing the self type of a PolyTraitRef to see if it is equal to
    ///   a type parameter `X`, since the type `X` does not reference any regions
    pub fn skip_binder(self) -> T {
        self.0
    }

    pub fn bound_vars(&self) -> &'tcx List<BoundVariableKind> {
        self.1
    }

    pub fn as_ref(&self) -> Binder<'tcx, &T> {
        Binder(&self.0, self.1)
    }

    pub fn as_deref(&self) -> Binder<'tcx, &T::Target>
    where
        T: Deref,
    {
        Binder(&self.0, self.1)
    }

    pub fn map_bound_ref_unchecked<F, U>(&self, f: F) -> Binder<'tcx, U>
    where
        F: FnOnce(&T) -> U,
    {
        let value = f(&self.0);
        Binder(value, self.1)
    }

    pub fn map_bound_ref<F, U: TypeVisitable<'tcx>>(&self, f: F) -> Binder<'tcx, U>
    where
        F: FnOnce(&T) -> U,
    {
        self.as_ref().map_bound(f)
    }

    pub fn map_bound<F, U: TypeVisitable<'tcx>>(self, f: F) -> Binder<'tcx, U>
    where
        F: FnOnce(T) -> U,
    {
        let value = f(self.0);
        if cfg!(debug_assertions) {
            let mut validator = ValidateBoundVars::new(self.1);
            value.visit_with(&mut validator);
        }
        Binder(value, self.1)
    }

    pub fn try_map_bound<F, U: TypeVisitable<'tcx>, E>(self, f: F) -> Result<Binder<'tcx, U>, E>
    where
        F: FnOnce(T) -> Result<U, E>,
    {
        let value = f(self.0)?;
        if cfg!(debug_assertions) {
            let mut validator = ValidateBoundVars::new(self.1);
            value.visit_with(&mut validator);
        }
        Ok(Binder(value, self.1))
    }

    /// Wraps a `value` in a binder, using the same bound variables as the
    /// current `Binder`. This should not be used if the new value *changes*
    /// the bound variables. Note: the (old or new) value itself does not
    /// necessarily need to *name* all the bound variables.
    ///
    /// This currently doesn't do anything different than `bind`, because we
    /// don't actually track bound vars. However, semantically, it is different
    /// because bound vars aren't allowed to change here, whereas they are
    /// in `bind`. This may be (debug) asserted in the future.
    pub fn rebind<U>(&self, value: U) -> Binder<'tcx, U>
    where
        U: TypeVisitable<'tcx>,
    {
        if cfg!(debug_assertions) {
            let mut validator = ValidateBoundVars::new(self.bound_vars());
            value.visit_with(&mut validator);
        }
        Binder(value, self.1)
    }

    /// Unwraps and returns the value within, but only if it contains
    /// no bound vars at all. (In other words, if this binder --
    /// and indeed any enclosing binder -- doesn't bind anything at
    /// all.) Otherwise, returns `None`.
    ///
    /// (One could imagine having a method that just unwraps a single
    /// binder, but permits late-bound vars bound by enclosing
    /// binders, but that would require adjusting the debruijn
    /// indices, and given the shallow binding structure we often use,
    /// would not be that useful.)
    pub fn no_bound_vars(self) -> Option<T>
    where
        T: TypeVisitable<'tcx>,
    {
        if self.0.has_escaping_bound_vars() { None } else { Some(self.skip_binder()) }
    }

    /// Splits the contents into two things that share the same binder
    /// level as the original, returning two distinct binders.
    ///
    /// `f` should consider bound regions at depth 1 to be free, and
    /// anything it produces with bound regions at depth 1 will be
    /// bound in the resulting return values.
    pub fn split<U, V, F>(self, f: F) -> (Binder<'tcx, U>, Binder<'tcx, V>)
    where
        F: FnOnce(T) -> (U, V),
    {
        let (u, v) = f(self.0);
        (Binder(u, self.1), Binder(v, self.1))
    }
}

impl<'tcx, T> Binder<'tcx, Option<T>> {
    pub fn transpose(self) -> Option<Binder<'tcx, T>> {
        let bound_vars = self.1;
        self.0.map(|v| Binder(v, bound_vars))
    }
}

/// Represents the projection of an associated type. In explicit UFCS
/// form this would be written `<T as Trait<..>>::N`.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable)]
pub struct ProjectionTy<'tcx> {
    /// The parameters of the associated item.
    pub substs: SubstsRef<'tcx>,

    /// The `DefId` of the `TraitItem` for the associated type `N`.
    ///
    /// Note that this is not the `DefId` of the `TraitRef` containing this
    /// associated type, which is in `tcx.associated_item(item_def_id).container`,
    /// aka. `tcx.parent(item_def_id).unwrap()`.
    pub item_def_id: DefId,
}

impl<'tcx> ProjectionTy<'tcx> {
    pub fn trait_def_id(&self, tcx: TyCtxt<'tcx>) -> DefId {
        tcx.parent(self.item_def_id)
    }

    /// Extracts the underlying trait reference and own substs from this projection.
    /// For example, if this is a projection of `<T as StreamingIterator>::Item<'a>`,
    /// then this function would return a `T: Iterator` trait reference and `['a]` as the own substs
    pub fn trait_ref_and_own_substs(
        &self,
        tcx: TyCtxt<'tcx>,
    ) -> (ty::TraitRef<'tcx>, &'tcx [ty::GenericArg<'tcx>]) {
        let def_id = tcx.parent(self.item_def_id);
        let trait_generics = tcx.generics_of(def_id);
        (
            ty::TraitRef { def_id, substs: self.substs.truncate_to(tcx, trait_generics) },
            &self.substs[trait_generics.count()..],
        )
    }

    /// Extracts the underlying trait reference from this projection.
    /// For example, if this is a projection of `<T as Iterator>::Item`,
    /// then this function would return a `T: Iterator` trait reference.
    ///
    /// WARNING: This will drop the substs for generic associated types
    /// consider calling [Self::trait_ref_and_own_substs] to get those
    /// as well.
    pub fn trait_ref(&self, tcx: TyCtxt<'tcx>) -> ty::TraitRef<'tcx> {
        let def_id = self.trait_def_id(tcx);
        ty::TraitRef { def_id, substs: self.substs.truncate_to(tcx, tcx.generics_of(def_id)) }
    }

    pub fn self_ty(&self) -> Ty<'tcx> {
        self.substs.type_at(0)
    }
}

#[derive(Copy, Clone, Debug, TypeFoldable, TypeVisitable)]
pub struct GenSig<'tcx> {
    pub resume_ty: Ty<'tcx>,
    pub yield_ty: Ty<'tcx>,
    pub return_ty: Ty<'tcx>,
}

pub type PolyGenSig<'tcx> = Binder<'tcx, GenSig<'tcx>>;

/// Signature of a function type, which we have arbitrarily
/// decided to use to refer to the input/output types.
///
/// - `inputs`: is the list of arguments and their modes.
/// - `output`: is the return type.
/// - `c_variadic`: indicates whether this is a C-variadic function.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable)]
pub struct FnSig<'tcx> {
    pub inputs_and_output: &'tcx List<Ty<'tcx>>,
    pub c_variadic: bool,
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

    // Creates a minimal `FnSig` to be used when encountering a `TyKind::Error` in a fallible
    // method.
    fn fake() -> FnSig<'tcx> {
        FnSig {
            inputs_and_output: List::empty(),
            c_variadic: false,
            unsafety: hir::Unsafety::Normal,
            abi: abi::Abi::Rust,
        }
    }
}

pub type PolyFnSig<'tcx> = Binder<'tcx, FnSig<'tcx>>;

impl<'tcx> PolyFnSig<'tcx> {
    #[inline]
    pub fn inputs(&self) -> Binder<'tcx, &'tcx [Ty<'tcx>]> {
        self.map_bound_ref_unchecked(|fn_sig| fn_sig.inputs())
    }
    #[inline]
    pub fn input(&self, index: usize) -> ty::Binder<'tcx, Ty<'tcx>> {
        self.map_bound_ref(|fn_sig| fn_sig.inputs()[index])
    }
    pub fn inputs_and_output(&self) -> ty::Binder<'tcx, &'tcx List<Ty<'tcx>>> {
        self.map_bound_ref(|fn_sig| fn_sig.inputs_and_output)
    }
    #[inline]
    pub fn output(&self) -> ty::Binder<'tcx, Ty<'tcx>> {
        self.map_bound_ref(|fn_sig| fn_sig.output())
    }
    pub fn c_variadic(&self) -> bool {
        self.skip_binder().c_variadic
    }
    pub fn unsafety(&self) -> hir::Unsafety {
        self.skip_binder().unsafety
    }
    pub fn abi(&self) -> abi::Abi {
        self.skip_binder().abi
    }
}

pub type CanonicalPolyFnSig<'tcx> = Canonical<'tcx, Binder<'tcx, FnSig<'tcx>>>;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub struct ParamTy {
    pub index: u32,
    pub name: Symbol,
}

impl<'tcx> ParamTy {
    pub fn new(index: u32, name: Symbol) -> ParamTy {
        ParamTy { index, name }
    }

    pub fn for_def(def: &ty::GenericParamDef) -> ParamTy {
        ParamTy::new(def.index, def.name)
    }

    #[inline]
    pub fn to_ty(self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        tcx.mk_ty_param(self.index, self.name)
    }
}

#[derive(Copy, Clone, Hash, TyEncodable, TyDecodable, Eq, PartialEq, Ord, PartialOrd)]
#[derive(HashStable)]
pub struct ParamConst {
    pub index: u32,
    pub name: Symbol,
}

impl ParamConst {
    pub fn new(index: u32, name: Symbol) -> ParamConst {
        ParamConst { index, name }
    }

    pub fn for_def(def: &ty::GenericParamDef) -> ParamConst {
        ParamConst::new(def.index, def.name)
    }
}

/// Use this rather than `RegionKind`, whenever possible.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, HashStable)]
#[rustc_pass_by_value]
pub struct Region<'tcx>(pub Interned<'tcx, RegionKind<'tcx>>);

impl<'tcx> Deref for Region<'tcx> {
    type Target = RegionKind<'tcx>;

    #[inline]
    fn deref(&self) -> &RegionKind<'tcx> {
        &self.0.0
    }
}

impl<'tcx> fmt::Debug for Region<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.kind())
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, TyEncodable, TyDecodable, PartialOrd, Ord)]
#[derive(HashStable)]
pub struct EarlyBoundRegion {
    pub def_id: DefId,
    pub index: u32,
    pub name: Symbol,
}

impl fmt::Debug for EarlyBoundRegion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}, {}", self.index, self.name)
    }
}

/// A **`const`** **v**ariable **ID**.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(HashStable, TyEncodable, TyDecodable)]
pub struct ConstVid<'tcx> {
    pub index: u32,
    pub phantom: PhantomData<&'tcx ()>,
}

rustc_index::newtype_index! {
    /// A **region** (lifetime) **v**ariable **ID**.
    #[derive(HashStable)]
    pub struct RegionVid {
        DEBUG_FORMAT = custom,
    }
}

impl Atom for RegionVid {
    fn index(self) -> usize {
        Idx::index(self)
    }
}

rustc_index::newtype_index! {
    #[derive(HashStable)]
    pub struct BoundVar { .. }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub struct BoundTy {
    pub var: BoundVar,
    pub kind: BoundTyKind,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable)]
pub enum BoundTyKind {
    Anon,
    Param(Symbol),
}

impl From<BoundVar> for BoundTy {
    fn from(var: BoundVar) -> Self {
        BoundTy { var, kind: BoundTyKind::Anon }
    }
}

/// A `ProjectionPredicate` for an `ExistentialTraitRef`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, TypeFoldable, TypeVisitable)]
pub struct ExistentialProjection<'tcx> {
    pub item_def_id: DefId,
    pub substs: SubstsRef<'tcx>,
    pub term: Term<'tcx>,
}

pub type PolyExistentialProjection<'tcx> = Binder<'tcx, ExistentialProjection<'tcx>>;

impl<'tcx> ExistentialProjection<'tcx> {
    /// Extracts the underlying existential trait reference from this projection.
    /// For example, if this is a projection of `exists T. <T as Iterator>::Item == X`,
    /// then this function would return an `exists T. T: Iterator` existential trait
    /// reference.
    pub fn trait_ref(&self, tcx: TyCtxt<'tcx>) -> ty::ExistentialTraitRef<'tcx> {
        let def_id = tcx.parent(self.item_def_id);
        let subst_count = tcx.generics_of(def_id).count() - 1;
        let substs = tcx.intern_substs(&self.substs[..subst_count]);
        ty::ExistentialTraitRef { def_id, substs }
    }

    pub fn with_self_ty(
        &self,
        tcx: TyCtxt<'tcx>,
        self_ty: Ty<'tcx>,
    ) -> ty::ProjectionPredicate<'tcx> {
        // otherwise the escaping regions would be captured by the binders
        debug_assert!(!self_ty.has_escaping_bound_vars());

        ty::ProjectionPredicate {
            projection_ty: ty::ProjectionTy {
                item_def_id: self.item_def_id,
                substs: tcx.mk_substs_trait(self_ty, self.substs),
            },
            term: self.term,
        }
    }

    pub fn erase_self_ty(
        tcx: TyCtxt<'tcx>,
        projection_predicate: ty::ProjectionPredicate<'tcx>,
    ) -> Self {
        // Assert there is a Self.
        projection_predicate.projection_ty.substs.type_at(0);

        Self {
            item_def_id: projection_predicate.projection_ty.item_def_id,
            substs: tcx.intern_substs(&projection_predicate.projection_ty.substs[1..]),
            term: projection_predicate.term,
        }
    }
}

impl<'tcx> PolyExistentialProjection<'tcx> {
    pub fn with_self_ty(
        &self,
        tcx: TyCtxt<'tcx>,
        self_ty: Ty<'tcx>,
    ) -> ty::PolyProjectionPredicate<'tcx> {
        self.map_bound(|p| p.with_self_ty(tcx, self_ty))
    }

    pub fn item_def_id(&self) -> DefId {
        self.skip_binder().item_def_id
    }
}

/// Region utilities
impl<'tcx> Region<'tcx> {
    pub fn kind(self) -> RegionKind<'tcx> {
        *self.0.0
    }

    /// Is this region named by the user?
    pub fn has_name(self) -> bool {
        match *self {
            ty::ReEarlyBound(ebr) => ebr.has_name(),
            ty::ReLateBound(_, br) => br.kind.is_named(),
            ty::ReFree(fr) => fr.bound_region.is_named(),
            ty::ReStatic => true,
            ty::ReVar(..) => false,
            ty::RePlaceholder(placeholder) => placeholder.name.is_named(),
            ty::ReEmpty(_) => false,
            ty::ReErased => false,
        }
    }

    #[inline]
    pub fn is_static(self) -> bool {
        matches!(*self, ty::ReStatic)
    }

    #[inline]
    pub fn is_erased(self) -> bool {
        matches!(*self, ty::ReErased)
    }

    #[inline]
    pub fn is_late_bound(self) -> bool {
        matches!(*self, ty::ReLateBound(..))
    }

    #[inline]
    pub fn is_placeholder(self) -> bool {
        matches!(*self, ty::RePlaceholder(..))
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        matches!(*self, ty::ReEmpty(..))
    }

    #[inline]
    pub fn bound_at_or_above_binder(self, index: ty::DebruijnIndex) -> bool {
        match *self {
            ty::ReLateBound(debruijn, _) => debruijn >= index,
            _ => false,
        }
    }

    pub fn type_flags(self) -> TypeFlags {
        let mut flags = TypeFlags::empty();

        match *self {
            ty::ReVar(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags = flags | TypeFlags::HAS_RE_INFER;
            }
            ty::RePlaceholder(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags = flags | TypeFlags::HAS_RE_PLACEHOLDER;
            }
            ty::ReEarlyBound(..) => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
                flags = flags | TypeFlags::HAS_RE_PARAM;
            }
            ty::ReFree { .. } => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
                flags = flags | TypeFlags::HAS_FREE_LOCAL_REGIONS;
            }
            ty::ReEmpty(_) | ty::ReStatic => {
                flags = flags | TypeFlags::HAS_FREE_REGIONS;
            }
            ty::ReLateBound(..) => {
                flags = flags | TypeFlags::HAS_RE_LATE_BOUND;
            }
            ty::ReErased => {
                flags = flags | TypeFlags::HAS_RE_ERASED;
            }
        }

        debug!("type_flags({:?}) = {:?}", self, flags);

        flags
    }

    /// Given an early-bound or free region, returns the `DefId` where it was bound.
    /// For example, consider the regions in this snippet of code:
    ///
    /// ```ignore (illustrative)
    /// impl<'a> Foo {
    /// //   ^^ -- early bound, declared on an impl
    ///
    ///     fn bar<'b, 'c>(x: &self, y: &'b u32, z: &'c u64) where 'static: 'c
    /// //         ^^  ^^     ^ anonymous, late-bound
    /// //         |   early-bound, appears in where-clauses
    /// //         late-bound, appears only in fn args
    ///     {..}
    /// }
    /// ```
    ///
    /// Here, `free_region_binding_scope('a)` would return the `DefId`
    /// of the impl, and for all the other highlighted regions, it
    /// would return the `DefId` of the function. In other cases (not shown), this
    /// function might return the `DefId` of a closure.
    pub fn free_region_binding_scope(self, tcx: TyCtxt<'_>) -> DefId {
        match *self {
            ty::ReEarlyBound(br) => tcx.parent(br.def_id),
            ty::ReFree(fr) => fr.scope,
            _ => bug!("free_region_binding_scope invoked on inappropriate region: {:?}", self),
        }
    }

    /// True for free regions other than `'static`.
    pub fn is_free(self) -> bool {
        matches!(*self, ty::ReEarlyBound(_) | ty::ReFree(_))
    }

    /// True if `self` is a free region or static.
    pub fn is_free_or_static(self) -> bool {
        match *self {
            ty::ReStatic => true,
            _ => self.is_free(),
        }
    }

    pub fn is_var(self) -> bool {
        matches!(self.kind(), ty::ReVar(_))
    }
}

/// Type utilities
impl<'tcx> Ty<'tcx> {
    #[inline(always)]
    pub fn kind(self) -> &'tcx TyKind<'tcx> {
        &self.0.0.kind
    }

    #[inline(always)]
    pub fn flags(self) -> TypeFlags {
        self.0.0.flags
    }

    #[inline]
    pub fn is_unit(self) -> bool {
        match self.kind() {
            Tuple(ref tys) => tys.is_empty(),
            _ => false,
        }
    }

    #[inline]
    pub fn is_never(self) -> bool {
        matches!(self.kind(), Never)
    }

    #[inline]
    pub fn is_primitive(self) -> bool {
        self.kind().is_primitive()
    }

    #[inline]
    pub fn is_adt(self) -> bool {
        matches!(self.kind(), Adt(..))
    }

    #[inline]
    pub fn is_ref(self) -> bool {
        matches!(self.kind(), Ref(..))
    }

    #[inline]
    pub fn is_ty_var(self) -> bool {
        matches!(self.kind(), Infer(TyVar(_)))
    }

    #[inline]
    pub fn ty_vid(self) -> Option<ty::TyVid> {
        match self.kind() {
            &Infer(TyVar(vid)) => Some(vid),
            _ => None,
        }
    }

    #[inline]
    pub fn is_ty_infer(self) -> bool {
        matches!(self.kind(), Infer(_))
    }

    #[inline]
    pub fn is_phantom_data(self) -> bool {
        if let Adt(def, _) = self.kind() { def.is_phantom_data() } else { false }
    }

    #[inline]
    pub fn is_bool(self) -> bool {
        *self.kind() == Bool
    }

    /// Returns `true` if this type is a `str`.
    #[inline]
    pub fn is_str(self) -> bool {
        *self.kind() == Str
    }

    #[inline]
    pub fn is_param(self, index: u32) -> bool {
        match self.kind() {
            ty::Param(ref data) => data.index == index,
            _ => false,
        }
    }

    #[inline]
    pub fn is_slice(self) -> bool {
        matches!(self.kind(), Slice(_))
    }

    #[inline]
    pub fn is_array_slice(self) -> bool {
        match self.kind() {
            Slice(_) => true,
            RawPtr(TypeAndMut { ty, .. }) | Ref(_, ty, _) => matches!(ty.kind(), Slice(_)),
            _ => false,
        }
    }

    #[inline]
    pub fn is_array(self) -> bool {
        matches!(self.kind(), Array(..))
    }

    #[inline]
    pub fn is_simd(self) -> bool {
        match self.kind() {
            Adt(def, _) => def.repr().simd(),
            _ => false,
        }
    }

    pub fn sequence_element_type(self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match self.kind() {
            Array(ty, _) | Slice(ty) => *ty,
            Str => tcx.types.u8,
            _ => bug!("`sequence_element_type` called on non-sequence value: {}", self),
        }
    }

    pub fn simd_size_and_type(self, tcx: TyCtxt<'tcx>) -> (u64, Ty<'tcx>) {
        match self.kind() {
            Adt(def, substs) => {
                assert!(def.repr().simd(), "`simd_size_and_type` called on non-SIMD type");
                let variant = def.non_enum_variant();
                let f0_ty = variant.fields[0].ty(tcx, substs);

                match f0_ty.kind() {
                    // If the first field is an array, we assume it is the only field and its
                    // elements are the SIMD components.
                    Array(f0_elem_ty, f0_len) => {
                        // FIXME(repr_simd): https://github.com/rust-lang/rust/pull/78863#discussion_r522784112
                        // The way we evaluate the `N` in `[T; N]` here only works since we use
                        // `simd_size_and_type` post-monomorphization. It will probably start to ICE
                        // if we use it in generic code. See the `simd-array-trait` ui test.
                        (f0_len.eval_usize(tcx, ParamEnv::empty()) as u64, *f0_elem_ty)
                    }
                    // Otherwise, the fields of this Adt are the SIMD components (and we assume they
                    // all have the same type).
                    _ => (variant.fields.len() as u64, f0_ty),
                }
            }
            _ => bug!("`simd_size_and_type` called on invalid type"),
        }
    }

    #[inline]
    pub fn is_region_ptr(self) -> bool {
        matches!(self.kind(), Ref(..))
    }

    #[inline]
    pub fn is_mutable_ptr(self) -> bool {
        matches!(
            self.kind(),
            RawPtr(TypeAndMut { mutbl: hir::Mutability::Mut, .. })
                | Ref(_, _, hir::Mutability::Mut)
        )
    }

    /// Get the mutability of the reference or `None` when not a reference
    #[inline]
    pub fn ref_mutability(self) -> Option<hir::Mutability> {
        match self.kind() {
            Ref(_, _, mutability) => Some(*mutability),
            _ => None,
        }
    }

    #[inline]
    pub fn is_unsafe_ptr(self) -> bool {
        matches!(self.kind(), RawPtr(_))
    }

    /// Tests if this is any kind of primitive pointer type (reference, raw pointer, fn pointer).
    #[inline]
    pub fn is_any_ptr(self) -> bool {
        self.is_region_ptr() || self.is_unsafe_ptr() || self.is_fn_ptr()
    }

    #[inline]
    pub fn is_box(self) -> bool {
        match self.kind() {
            Adt(def, _) => def.is_box(),
            _ => false,
        }
    }

    /// Panics if called on any type other than `Box<T>`.
    pub fn boxed_ty(self) -> Ty<'tcx> {
        match self.kind() {
            Adt(def, substs) if def.is_box() => substs.type_at(0),
            _ => bug!("`boxed_ty` is called on non-box type {:?}", self),
        }
    }

    /// A scalar type is one that denotes an atomic datum, with no sub-components.
    /// (A RawPtr is scalar because it represents a non-managed pointer, so its
    /// contents are abstract to rustc.)
    #[inline]
    pub fn is_scalar(self) -> bool {
        matches!(
            self.kind(),
            Bool | Char
                | Int(_)
                | Float(_)
                | Uint(_)
                | FnDef(..)
                | FnPtr(_)
                | RawPtr(_)
                | Infer(IntVar(_) | FloatVar(_))
        )
    }

    /// Returns `true` if this type is a floating point type.
    #[inline]
    pub fn is_floating_point(self) -> bool {
        matches!(self.kind(), Float(_) | Infer(FloatVar(_)))
    }

    #[inline]
    pub fn is_trait(self) -> bool {
        matches!(self.kind(), Dynamic(..))
    }

    #[inline]
    pub fn is_enum(self) -> bool {
        matches!(self.kind(), Adt(adt_def, _) if adt_def.is_enum())
    }

    #[inline]
    pub fn is_union(self) -> bool {
        matches!(self.kind(), Adt(adt_def, _) if adt_def.is_union())
    }

    #[inline]
    pub fn is_closure(self) -> bool {
        matches!(self.kind(), Closure(..))
    }

    #[inline]
    pub fn is_generator(self) -> bool {
        matches!(self.kind(), Generator(..))
    }

    #[inline]
    pub fn is_integral(self) -> bool {
        matches!(self.kind(), Infer(IntVar(_)) | Int(_) | Uint(_))
    }

    #[inline]
    pub fn is_fresh_ty(self) -> bool {
        matches!(self.kind(), Infer(FreshTy(_)))
    }

    #[inline]
    pub fn is_fresh(self) -> bool {
        matches!(self.kind(), Infer(FreshTy(_) | FreshIntTy(_) | FreshFloatTy(_)))
    }

    #[inline]
    pub fn is_char(self) -> bool {
        matches!(self.kind(), Char)
    }

    #[inline]
    pub fn is_numeric(self) -> bool {
        self.is_integral() || self.is_floating_point()
    }

    #[inline]
    pub fn is_signed(self) -> bool {
        matches!(self.kind(), Int(_))
    }

    #[inline]
    pub fn is_ptr_sized_integral(self) -> bool {
        matches!(self.kind(), Int(ty::IntTy::Isize) | Uint(ty::UintTy::Usize))
    }

    #[inline]
    pub fn has_concrete_skeleton(self) -> bool {
        !matches!(self.kind(), Param(_) | Infer(_) | Error(_))
    }

    /// Checks whether a type recursively contains another type
    ///
    /// Example: `Option<()>` contains `()`
    pub fn contains(self, other: Ty<'tcx>) -> bool {
        struct ContainsTyVisitor<'tcx>(Ty<'tcx>);

        impl<'tcx> TypeVisitor<'tcx> for ContainsTyVisitor<'tcx> {
            type BreakTy = ();

            fn visit_ty(&mut self, t: Ty<'tcx>) -> ControlFlow<Self::BreakTy> {
                if self.0 == t { ControlFlow::BREAK } else { t.super_visit_with(self) }
            }
        }

        let cf = self.visit_with(&mut ContainsTyVisitor(other));
        cf.is_break()
    }

    /// Returns the type and mutability of `*ty`.
    ///
    /// The parameter `explicit` indicates if this is an *explicit* dereference.
    /// Some types -- notably unsafe ptrs -- can only be dereferenced explicitly.
    pub fn builtin_deref(self, explicit: bool) -> Option<TypeAndMut<'tcx>> {
        match self.kind() {
            Adt(def, _) if def.is_box() => {
                Some(TypeAndMut { ty: self.boxed_ty(), mutbl: hir::Mutability::Not })
            }
            Ref(_, ty, mutbl) => Some(TypeAndMut { ty: *ty, mutbl: *mutbl }),
            RawPtr(mt) if explicit => Some(*mt),
            _ => None,
        }
    }

    /// Returns the type of `ty[i]`.
    pub fn builtin_index(self) -> Option<Ty<'tcx>> {
        match self.kind() {
            Array(ty, _) | Slice(ty) => Some(*ty),
            _ => None,
        }
    }

    pub fn fn_sig(self, tcx: TyCtxt<'tcx>) -> PolyFnSig<'tcx> {
        match self.kind() {
            FnDef(def_id, substs) => tcx.bound_fn_sig(*def_id).subst(tcx, substs),
            FnPtr(f) => *f,
            Error(_) => {
                // ignore errors (#54954)
                ty::Binder::dummy(FnSig::fake())
            }
            Closure(..) => bug!(
                "to get the signature of a closure, use `substs.as_closure().sig()` not `fn_sig()`",
            ),
            _ => bug!("Ty::fn_sig() called on non-fn type: {:?}", self),
        }
    }

    #[inline]
    pub fn is_fn(self) -> bool {
        matches!(self.kind(), FnDef(..) | FnPtr(_))
    }

    #[inline]
    pub fn is_fn_ptr(self) -> bool {
        matches!(self.kind(), FnPtr(_))
    }

    #[inline]
    pub fn is_impl_trait(self) -> bool {
        matches!(self.kind(), Opaque(..))
    }

    #[inline]
    pub fn ty_adt_def(self) -> Option<AdtDef<'tcx>> {
        match self.kind() {
            Adt(adt, _) => Some(*adt),
            _ => None,
        }
    }

    /// Iterates over tuple fields.
    /// Panics when called on anything but a tuple.
    #[inline]
    pub fn tuple_fields(self) -> &'tcx List<Ty<'tcx>> {
        match self.kind() {
            Tuple(substs) => substs,
            _ => bug!("tuple_fields called on non-tuple"),
        }
    }

    /// If the type contains variants, returns the valid range of variant indices.
    //
    // FIXME: This requires the optimized MIR in the case of generators.
    #[inline]
    pub fn variant_range(self, tcx: TyCtxt<'tcx>) -> Option<Range<VariantIdx>> {
        match self.kind() {
            TyKind::Adt(adt, _) => Some(adt.variant_range()),
            TyKind::Generator(def_id, substs, _) => {
                Some(substs.as_generator().variant_range(*def_id, tcx))
            }
            _ => None,
        }
    }

    /// If the type contains variants, returns the variant for `variant_index`.
    /// Panics if `variant_index` is out of range.
    //
    // FIXME: This requires the optimized MIR in the case of generators.
    #[inline]
    pub fn discriminant_for_variant(
        self,
        tcx: TyCtxt<'tcx>,
        variant_index: VariantIdx,
    ) -> Option<Discr<'tcx>> {
        match self.kind() {
            TyKind::Adt(adt, _) if adt.variants().is_empty() => {
                // This can actually happen during CTFE, see
                // https://github.com/rust-lang/rust/issues/89765.
                None
            }
            TyKind::Adt(adt, _) if adt.is_enum() => {
                Some(adt.discriminant_for_variant(tcx, variant_index))
            }
            TyKind::Generator(def_id, substs, _) => {
                Some(substs.as_generator().discriminant_for_variant(*def_id, tcx, variant_index))
            }
            _ => None,
        }
    }

    /// Returns the type of the discriminant of this type.
    pub fn discriminant_ty(self, tcx: TyCtxt<'tcx>) -> Ty<'tcx> {
        match self.kind() {
            ty::Adt(adt, _) if adt.is_enum() => adt.repr().discr_type().to_ty(tcx),
            ty::Generator(_, substs, _) => substs.as_generator().discr_ty(tcx),

            ty::Param(_) | ty::Projection(_) | ty::Opaque(..) | ty::Infer(ty::TyVar(_)) => {
                let assoc_items = tcx.associated_item_def_ids(
                    tcx.require_lang_item(hir::LangItem::DiscriminantKind, None),
                );
                tcx.mk_projection(assoc_items[0], tcx.intern_substs(&[self.into()]))
            }

            ty::Bool
            | ty::Char
            | ty::Int(_)
            | ty::Uint(_)
            | ty::Float(_)
            | ty::Adt(..)
            | ty::Foreign(_)
            | ty::Str
            | ty::Array(..)
            | ty::Slice(_)
            | ty::RawPtr(_)
            | ty::Ref(..)
            | ty::FnDef(..)
            | ty::FnPtr(..)
            | ty::Dynamic(..)
            | ty::Closure(..)
            | ty::GeneratorWitness(..)
            | ty::Never
            | ty::Tuple(_)
            | ty::Error(_)
            | ty::Infer(IntVar(_) | FloatVar(_)) => tcx.types.u8,

            ty::Bound(..)
            | ty::Placeholder(_)
            | ty::Infer(FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("`discriminant_ty` applied to unexpected type: {:?}", self)
            }
        }
    }

    /// Returns the type of metadata for (potentially fat) pointers to this type,
    /// and a boolean signifying if this is conditional on this type being `Sized`.
    pub fn ptr_metadata_ty(
        self,
        tcx: TyCtxt<'tcx>,
        normalize: impl FnMut(Ty<'tcx>) -> Ty<'tcx>,
    ) -> (Ty<'tcx>, bool) {
        let tail = tcx.struct_tail_with_normalize(self, normalize, || {});
        match tail.kind() {
            // Sized types
            ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::FnDef(..)
            | ty::FnPtr(_)
            | ty::RawPtr(..)
            | ty::Char
            | ty::Ref(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
            | ty::Array(..)
            | ty::Closure(..)
            | ty::Never
            | ty::Error(_)
            // Extern types have metadata = ().
            | ty::Foreign(..)
            // If returned by `struct_tail_without_normalization` this is a unit struct
            // without any fields, or not a struct, and therefore is Sized.
            | ty::Adt(..)
            // If returned by `struct_tail_without_normalization` this is the empty tuple,
            // a.k.a. unit type, which is Sized
            | ty::Tuple(..) => (tcx.types.unit, false),

            ty::Str | ty::Slice(_) => (tcx.types.usize, false),
            ty::Dynamic(..) => {
                let dyn_metadata = tcx.lang_items().dyn_metadata().unwrap();
                (tcx.bound_type_of(dyn_metadata).subst(tcx, &[tail.into()]), false)
            },

            // type parameters only have unit metadata if they're sized, so return true
            // to make sure we double check this during confirmation
            ty::Param(_) |  ty::Projection(_) | ty::Opaque(..) => (tcx.types.unit, true),

            ty::Infer(ty::TyVar(_))
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("`ptr_metadata_ty` applied to unexpected type: {:?} (tail = {:?})", self, tail)
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
    /// inferred. Once upvar inference (in `rustc_typeck/src/check/upvar.rs`)
    /// is complete, that type variable will be unified.
    pub fn to_opt_closure_kind(self) -> Option<ty::ClosureKind> {
        match self.kind() {
            Int(int_ty) => match int_ty {
                ty::IntTy::I8 => Some(ty::ClosureKind::Fn),
                ty::IntTy::I16 => Some(ty::ClosureKind::FnMut),
                ty::IntTy::I32 => Some(ty::ClosureKind::FnOnce),
                _ => bug!("cannot convert type `{:?}` to a closure kind", self),
            },

            // "Bound" types appear in canonical queries when the
            // closure type is not yet known
            Bound(..) | Infer(_) => None,

            Error(_) => Some(ty::ClosureKind::Fn),

            _ => bug!("cannot convert type `{:?}` to a closure kind", self),
        }
    }

    /// Fast path helper for testing if a type is `Sized`.
    ///
    /// Returning true means the type is known to be sized. Returning
    /// `false` means nothing -- could be sized, might not be.
    ///
    /// Note that we could never rely on the fact that a type such as `[_]` is
    /// trivially `!Sized` because we could be in a type environment with a
    /// bound such as `[_]: Copy`. A function with such a bound obviously never
    /// can be called, but that doesn't mean it shouldn't typecheck. This is why
    /// this method doesn't return `Option<bool>`.
    pub fn is_trivially_sized(self, tcx: TyCtxt<'tcx>) -> bool {
        match self.kind() {
            ty::Infer(ty::IntVar(_) | ty::FloatVar(_))
            | ty::Uint(_)
            | ty::Int(_)
            | ty::Bool
            | ty::Float(_)
            | ty::FnDef(..)
            | ty::FnPtr(_)
            | ty::RawPtr(..)
            | ty::Char
            | ty::Ref(..)
            | ty::Generator(..)
            | ty::GeneratorWitness(..)
            | ty::Array(..)
            | ty::Closure(..)
            | ty::Never
            | ty::Error(_) => true,

            ty::Str | ty::Slice(_) | ty::Dynamic(..) | ty::Foreign(..) => false,

            ty::Tuple(tys) => tys.iter().all(|ty| ty.is_trivially_sized(tcx)),

            ty::Adt(def, _substs) => def.sized_constraint(tcx).0.is_empty(),

            ty::Projection(_) | ty::Param(_) | ty::Opaque(..) => false,

            ty::Infer(ty::TyVar(_)) => false,

            ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Infer(ty::FreshTy(_) | ty::FreshIntTy(_) | ty::FreshFloatTy(_)) => {
                bug!("`is_trivially_sized` applied to unexpected type: {:?}", self)
            }
        }
    }

    /// Fast path helper for primitives which are always `Copy` and which
    /// have a side-effect-free `Clone` impl.
    ///
    /// Returning true means the type is known to be pure and `Copy+Clone`.
    /// Returning `false` means nothing -- could be `Copy`, might not be.
    ///
    /// This is mostly useful for optimizations, as there are the types
    /// on which we can replace cloning with dereferencing.
    pub fn is_trivially_pure_clone_copy(self) -> bool {
        match self.kind() {
            ty::Bool | ty::Char | ty::Never => true,

            // These aren't even `Clone`
            ty::Str | ty::Slice(..) | ty::Foreign(..) | ty::Dynamic(..) => false,

            ty::Int(..) | ty::Uint(..) | ty::Float(..) => true,

            // The voldemort ZSTs are fine.
            ty::FnDef(..) => true,

            ty::Array(element_ty, _len) => element_ty.is_trivially_pure_clone_copy(),

            // A 100-tuple isn't "trivial", so doing this only for reasonable sizes.
            ty::Tuple(field_tys) => {
                field_tys.len() <= 3 && field_tys.iter().all(Self::is_trivially_pure_clone_copy)
            }

            // Sometimes traits aren't implemented for every ABI or arity,
            // because we can't be generic over everything yet.
            ty::FnPtr(..) => false,

            // Definitely absolutely not copy.
            ty::Ref(_, _, hir::Mutability::Mut) => false,

            // Thin pointers & thin shared references are pure-clone-copy, but for
            // anything with custom metadata it might be more complicated.
            ty::Ref(_, _, hir::Mutability::Not) | ty::RawPtr(..) => false,

            ty::Generator(..) | ty::GeneratorWitness(..) => false,

            // Might be, but not "trivial" so just giving the safe answer.
            ty::Adt(..) | ty::Closure(..) | ty::Opaque(..) => false,

            ty::Projection(..) | ty::Param(..) | ty::Infer(..) | ty::Error(..) => false,

            ty::Bound(..) | ty::Placeholder(..) => {
                bug!("`is_trivially_pure_clone_copy` applied to unexpected type: {:?}", self);
            }
        }
    }
}

/// Extra information about why we ended up with a particular variance.
/// This is only used to add more information to error messages, and
/// has no effect on soundness. While choosing the 'wrong' `VarianceDiagInfo`
/// may lead to confusing notes in error messages, it will never cause
/// a miscompilation or unsoundness.
///
/// When in doubt, use `VarianceDiagInfo::default()`
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum VarianceDiagInfo<'tcx> {
    /// No additional information - this is the default.
    /// We will not add any additional information to error messages.
    #[default]
    None,
    /// We switched our variance because a generic argument occurs inside
    /// the invariant generic argument of another type.
    Invariant {
        /// The generic type containing the generic parameter
        /// that changes the variance (e.g. `*mut T`, `MyStruct<T>`)
        ty: Ty<'tcx>,
        /// The index of the generic parameter being used
        /// (e.g. `0` for `*mut T`, `1` for `MyStruct<'CovariantParam, 'InvariantParam>`)
        param_index: u32,
    },
}

impl<'tcx> VarianceDiagInfo<'tcx> {
    /// Mirrors `Variance::xform` - used to 'combine' the existing
    /// and new `VarianceDiagInfo`s when our variance changes.
    pub fn xform(self, other: VarianceDiagInfo<'tcx>) -> VarianceDiagInfo<'tcx> {
        // For now, just use the first `VarianceDiagInfo::Invariant` that we see
        match self {
            VarianceDiagInfo::None => other,
            VarianceDiagInfo::Invariant { .. } => self,
        }
    }
}
