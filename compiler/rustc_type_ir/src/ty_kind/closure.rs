use std::ops::ControlFlow;

use derive_where::derive_where;
use rustc_type_ir_macros::{Lift_Generic, TypeFoldable_Generic, TypeVisitable_Generic};

use crate::data_structures::DelayedMap;
use crate::fold::{TypeFoldable, TypeFolder, TypeSuperFoldable, shift_region};
use crate::inherent::*;
use crate::visit::{TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor};
use crate::{self as ty, Interner};

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
/// ## Coroutines
///
/// Coroutines are handled similarly in `CoroutineArgs`. The set of
/// type parameters is similar, but `CK` and `CS` are replaced by the
/// following type parameters:
///
/// * `GS`: The coroutine's "resume type", which is the type of the
///   argument passed to `resume`, and the type of `yield` expressions
///   inside the coroutine.
/// * `GY`: The "yield type", which is the type of values passed to
///   `yield` inside the coroutine.
/// * `GR`: The "return type", which is the type of value returned upon
///   completion of the coroutine.
/// * `GW`: The "coroutine witness".
#[derive_where(Clone, Copy, PartialEq, Eq, Hash, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
pub struct ClosureArgs<I: Interner> {
    /// Lifetime and type parameters from the enclosing function,
    /// concatenated with a tuple containing the types of the upvars.
    ///
    /// These are separated out because codegen wants to pass them around
    /// when monomorphizing.
    pub args: I::GenericArgs,
}

/// Struct returned by `split()`.
pub struct ClosureArgsParts<I: Interner> {
    /// This is the args of the typeck root.
    pub parent_args: I::GenericArgsSlice,
    /// Represents the maximum calling capability of the closure.
    pub closure_kind_ty: I::Ty,
    /// Captures the closure's signature. This closure signature is "tupled", and
    /// thus has a peculiar signature of `extern "rust-call" fn((Args, ...)) -> Ty`.
    pub closure_sig_as_fn_ptr_ty: I::Ty,
    /// The upvars captured by the closure. Remains an inference variable
    /// until the upvar analysis, which happens late in HIR typeck.
    pub tupled_upvars_ty: I::Ty,
}

impl<I: Interner> ClosureArgs<I> {
    /// Construct `ClosureArgs` from `ClosureArgsParts`, containing `Args`
    /// for the closure parent, alongside additional closure-specific components.
    pub fn new(cx: I, parts: ClosureArgsParts<I>) -> ClosureArgs<I> {
        ClosureArgs {
            args: cx.mk_args_from_iter(parts.parent_args.iter().chain([
                parts.closure_kind_ty.into(),
                parts.closure_sig_as_fn_ptr_ty.into(),
                parts.tupled_upvars_ty.into(),
            ])),
        }
    }

    /// Divides the closure args into their respective components.
    /// The ordering assumed here must match that used by `ClosureArgs::new` above.
    fn split(self) -> ClosureArgsParts<I> {
        self.args.split_closure_args()
    }

    /// Returns the generic parameters of the closure's parent.
    pub fn parent_args(self) -> I::GenericArgsSlice {
        self.split().parent_args
    }

    /// Returns an iterator over the list of types of captured paths by the closure.
    /// In case there was a type error in figuring out the types of the captured path, an
    /// empty iterator is returned.
    #[inline]
    pub fn upvar_tys(self) -> I::Tys {
        match self.tupled_upvars_ty().kind() {
            ty::Error(_) => Default::default(),
            ty::Tuple(tys) => tys,
            ty::Infer(_) => panic!("upvar_tys called before capture types are inferred"),
            ty => panic!("Unexpected representation of upvar types tuple {:?}", ty),
        }
    }

    /// Returns the tuple type representing the upvars for this closure.
    #[inline]
    pub fn tupled_upvars_ty(self) -> I::Ty {
        self.split().tupled_upvars_ty
    }

    /// Returns the closure kind for this closure; may return a type
    /// variable during inference. To get the closure kind during
    /// inference, use `infcx.closure_kind(args)`.
    pub fn kind_ty(self) -> I::Ty {
        self.split().closure_kind_ty
    }

    /// Returns the `fn` pointer type representing the closure signature for this
    /// closure.
    // FIXME(eddyb) this should be unnecessary, as the shallowly resolved
    // type is known at the time of the creation of `ClosureArgs`,
    // see `rustc_hir_analysis::check::closure`.
    pub fn sig_as_fn_ptr_ty(self) -> I::Ty {
        self.split().closure_sig_as_fn_ptr_ty
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
    pub fn sig(self) -> ty::Binder<I, ty::FnSig<I>> {
        match self.sig_as_fn_ptr_ty().kind() {
            ty::FnPtr(sig_tys, hdr) => sig_tys.with(hdr),
            ty => panic!("closure_sig_as_fn_ptr_ty is not a fn-ptr: {ty:?}"),
        }
    }
}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
pub struct CoroutineClosureArgs<I: Interner> {
    pub args: I::GenericArgs,
}

/// See docs for explanation of how each argument is used.
///
/// See [`CoroutineClosureSignature`] for how these arguments are put together
/// to make a callable [`ty::FnSig`] suitable for typeck and borrowck.
pub struct CoroutineClosureArgsParts<I: Interner> {
    /// This is the args of the typeck root.
    pub parent_args: I::GenericArgsSlice,
    /// Represents the maximum calling capability of the closure.
    pub closure_kind_ty: I::Ty,
    /// Represents all of the relevant parts of the coroutine returned by this
    /// coroutine-closure. This signature parts type will have the general
    /// shape of `fn(tupled_inputs, resume_ty) -> (return_ty, yield_ty)`, where
    /// `resume_ty`, `return_ty`, and `yield_ty` are the respective types for the
    /// coroutine returned by the coroutine-closure.
    ///
    /// Use `coroutine_closure_sig` to break up this type rather than using it
    /// yourself.
    pub signature_parts_ty: I::Ty,
    /// The upvars captured by the closure. Remains an inference variable
    /// until the upvar analysis, which happens late in HIR typeck.
    pub tupled_upvars_ty: I::Ty,
    /// a function pointer that has the shape `for<'env> fn() -> (&'env T, ...)`.
    /// This allows us to represent the binder of the self-captures of the closure.
    ///
    /// For example, if the coroutine returned by the closure borrows `String`
    /// from the closure's upvars, this will be `for<'env> fn() -> (&'env String,)`,
    /// while the `tupled_upvars_ty`, representing the by-move version of the same
    /// captures, will be `(String,)`.
    pub coroutine_captures_by_ref_ty: I::Ty,
    /// Witness type returned by the generator produced by this coroutine-closure.
    pub coroutine_witness_ty: I::Ty,
}

impl<I: Interner> CoroutineClosureArgs<I> {
    pub fn new(cx: I, parts: CoroutineClosureArgsParts<I>) -> CoroutineClosureArgs<I> {
        CoroutineClosureArgs {
            args: cx.mk_args_from_iter(parts.parent_args.iter().chain([
                parts.closure_kind_ty.into(),
                parts.signature_parts_ty.into(),
                parts.tupled_upvars_ty.into(),
                parts.coroutine_captures_by_ref_ty.into(),
                parts.coroutine_witness_ty.into(),
            ])),
        }
    }

    fn split(self) -> CoroutineClosureArgsParts<I> {
        self.args.split_coroutine_closure_args()
    }

    pub fn parent_args(self) -> I::GenericArgsSlice {
        self.split().parent_args
    }

    #[inline]
    pub fn upvar_tys(self) -> I::Tys {
        match self.tupled_upvars_ty().kind() {
            ty::Error(_) => Default::default(),
            ty::Tuple(..) => self.tupled_upvars_ty().tuple_fields(),
            ty::Infer(_) => panic!("upvar_tys called before capture types are inferred"),
            ty => panic!("Unexpected representation of upvar types tuple {:?}", ty),
        }
    }

    #[inline]
    pub fn tupled_upvars_ty(self) -> I::Ty {
        self.split().tupled_upvars_ty
    }

    pub fn kind_ty(self) -> I::Ty {
        self.split().closure_kind_ty
    }

    pub fn kind(self) -> ty::ClosureKind {
        self.kind_ty().to_opt_closure_kind().unwrap()
    }

    pub fn signature_parts_ty(self) -> I::Ty {
        self.split().signature_parts_ty
    }

    pub fn coroutine_closure_sig(self) -> ty::Binder<I, CoroutineClosureSignature<I>> {
        let interior = self.coroutine_witness_ty();
        let ty::FnPtr(sig_tys, hdr) = self.signature_parts_ty().kind() else { panic!() };
        sig_tys.map_bound(|sig_tys| {
            let [resume_ty, tupled_inputs_ty] = *sig_tys.inputs().as_slice() else {
                panic!();
            };
            let [yield_ty, return_ty] = *sig_tys.output().tuple_fields().as_slice() else {
                panic!()
            };
            CoroutineClosureSignature {
                interior,
                tupled_inputs_ty,
                resume_ty,
                yield_ty,
                return_ty,
                c_variadic: hdr.c_variadic,
                safety: hdr.safety,
                abi: hdr.abi,
            }
        })
    }

    pub fn coroutine_captures_by_ref_ty(self) -> I::Ty {
        self.split().coroutine_captures_by_ref_ty
    }

    pub fn coroutine_witness_ty(self) -> I::Ty {
        self.split().coroutine_witness_ty
    }

    pub fn has_self_borrows(&self) -> bool {
        match self.coroutine_captures_by_ref_ty().kind() {
            ty::FnPtr(sig_tys, _) => sig_tys
                .skip_binder()
                .visit_with(&mut HasRegionsBoundAt { binder: ty::INNERMOST })
                .is_break(),
            ty::Error(_) => true,
            _ => panic!(),
        }
    }
}

/// Unlike `has_escaping_bound_vars` or `outermost_exclusive_binder`, this will
/// detect only regions bound *at* the debruijn index.
struct HasRegionsBoundAt {
    binder: ty::DebruijnIndex,
}
// FIXME: Could be optimized to not walk into components with no escaping bound vars.
impl<I: Interner> TypeVisitor<I> for HasRegionsBoundAt {
    type Result = ControlFlow<()>;
    fn visit_binder<T: TypeVisitable<I>>(&mut self, t: &ty::Binder<I, T>) -> Self::Result {
        self.binder.shift_in(1);
        t.super_visit_with(self)?;
        self.binder.shift_out(1);
        ControlFlow::Continue(())
    }

    fn visit_region(&mut self, r: I::Region) -> Self::Result {
        if matches!(r.kind(), ty::ReBound(binder, _) if self.binder == binder) {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }
}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
pub struct CoroutineClosureSignature<I: Interner> {
    pub interior: I::Ty,
    pub tupled_inputs_ty: I::Ty,
    pub resume_ty: I::Ty,
    pub yield_ty: I::Ty,
    pub return_ty: I::Ty,

    // Like the `fn_sig_as_fn_ptr_ty` of a regular closure, these types
    // never actually differ. But we save them rather than recreating them
    // from scratch just for good measure.
    /// Always false
    pub c_variadic: bool,
    /// Always `Normal` (safe)
    #[type_visitable(ignore)]
    #[type_foldable(identity)]
    pub safety: I::Safety,
    /// Always `RustCall`
    #[type_visitable(ignore)]
    #[type_foldable(identity)]
    pub abi: I::Abi,
}

impl<I: Interner> CoroutineClosureSignature<I> {
    /// Construct a coroutine from the closure signature. Since a coroutine signature
    /// is agnostic to the type of generator that is returned (by-ref/by-move),
    /// the caller must specify what "flavor" of generator that they'd like to
    /// create. Additionally, they must manually compute the upvars of the closure.
    ///
    /// This helper is not really meant to be used directly except for early on
    /// during typeck, when we want to put inference vars into the kind and upvars tys.
    /// When the kind and upvars are known, use the other helper functions.
    pub fn to_coroutine(
        self,
        cx: I,
        parent_args: I::GenericArgsSlice,
        coroutine_kind_ty: I::Ty,
        coroutine_def_id: I::DefId,
        tupled_upvars_ty: I::Ty,
    ) -> I::Ty {
        let coroutine_args = ty::CoroutineArgs::new(
            cx,
            ty::CoroutineArgsParts {
                parent_args,
                kind_ty: coroutine_kind_ty,
                resume_ty: self.resume_ty,
                yield_ty: self.yield_ty,
                return_ty: self.return_ty,
                witness: self.interior,
                tupled_upvars_ty,
            },
        );

        Ty::new_coroutine(cx, coroutine_def_id, coroutine_args.args)
    }

    /// Given known upvars and a [`ClosureKind`](ty::ClosureKind), compute the coroutine
    /// returned by that corresponding async fn trait.
    ///
    /// This function expects the upvars to have been computed already, and doesn't check
    /// that the `ClosureKind` is actually supported by the coroutine-closure.
    pub fn to_coroutine_given_kind_and_upvars(
        self,
        cx: I,
        parent_args: I::GenericArgsSlice,
        coroutine_def_id: I::DefId,
        goal_kind: ty::ClosureKind,
        env_region: I::Region,
        closure_tupled_upvars_ty: I::Ty,
        coroutine_captures_by_ref_ty: I::Ty,
    ) -> I::Ty {
        let tupled_upvars_ty = Self::tupled_upvars_by_closure_kind(
            cx,
            goal_kind,
            self.tupled_inputs_ty,
            closure_tupled_upvars_ty,
            coroutine_captures_by_ref_ty,
            env_region,
        );

        self.to_coroutine(
            cx,
            parent_args,
            Ty::from_coroutine_closure_kind(cx, goal_kind),
            coroutine_def_id,
            tupled_upvars_ty,
        )
    }

    /// Compute the tupled upvars that a coroutine-closure's output coroutine
    /// would return for the given `ClosureKind`.
    ///
    /// When `ClosureKind` is `FnMut`/`Fn`, then this will use the "captures by ref"
    /// to return a set of upvars which are borrowed with the given `env_region`.
    ///
    /// This ensures that the `AsyncFn::call` will return a coroutine whose upvars'
    /// lifetimes are related to the lifetime of the borrow on the closure made for
    /// the call. This allows borrowck to enforce the self-borrows correctly.
    pub fn tupled_upvars_by_closure_kind(
        cx: I,
        kind: ty::ClosureKind,
        tupled_inputs_ty: I::Ty,
        closure_tupled_upvars_ty: I::Ty,
        coroutine_captures_by_ref_ty: I::Ty,
        env_region: I::Region,
    ) -> I::Ty {
        match kind {
            ty::ClosureKind::Fn | ty::ClosureKind::FnMut => {
                let ty::FnPtr(sig_tys, _) = coroutine_captures_by_ref_ty.kind() else {
                    panic!();
                };
                let coroutine_captures_by_ref_ty =
                    sig_tys.output().skip_binder().fold_with(&mut FoldEscapingRegions {
                        interner: cx,
                        region: env_region,
                        debruijn: ty::INNERMOST,
                        cache: Default::default(),
                    });
                Ty::new_tup_from_iter(
                    cx,
                    tupled_inputs_ty
                        .tuple_fields()
                        .iter()
                        .chain(coroutine_captures_by_ref_ty.tuple_fields().iter()),
                )
            }
            ty::ClosureKind::FnOnce => Ty::new_tup_from_iter(
                cx,
                tupled_inputs_ty
                    .tuple_fields()
                    .iter()
                    .chain(closure_tupled_upvars_ty.tuple_fields().iter()),
            ),
        }
    }
}

/// Instantiates a `for<'env> ...` binder with a specific region.
// FIXME(async_closures): Get rid of this in favor of `BoundVarReplacerDelegate`
// when that is uplifted.
struct FoldEscapingRegions<I: Interner> {
    interner: I,
    debruijn: ty::DebruijnIndex,
    region: I::Region,

    // Depends on `debruijn` because we may have types with regions of different
    // debruijn depths depending on the binders we've entered.
    cache: DelayedMap<(ty::DebruijnIndex, I::Ty), I::Ty>,
}

impl<I: Interner> TypeFolder<I> for FoldEscapingRegions<I> {
    fn cx(&self) -> I {
        self.interner
    }

    fn fold_ty(&mut self, t: I::Ty) -> I::Ty {
        if !t.has_vars_bound_at_or_above(self.debruijn) {
            t
        } else if let Some(&t) = self.cache.get(&(self.debruijn, t)) {
            t
        } else {
            let res = t.super_fold_with(self);
            assert!(self.cache.insert((self.debruijn, t), res));
            res
        }
    }

    fn fold_binder<T>(&mut self, t: ty::Binder<I, T>) -> ty::Binder<I, T>
    where
        T: TypeFoldable<I>,
    {
        self.debruijn.shift_in(1);
        let result = t.super_fold_with(self);
        self.debruijn.shift_out(1);
        result
    }

    fn fold_region(&mut self, r: <I as Interner>::Region) -> <I as Interner>::Region {
        if let ty::ReBound(debruijn, _) = r.kind() {
            assert!(
                debruijn <= self.debruijn,
                "cannot instantiate binder with escaping bound vars"
            );
            if self.debruijn == debruijn {
                shift_region(self.interner, self.region, self.debruijn.as_u32())
            } else {
                r
            }
        } else {
            r
        }
    }
}

#[derive_where(Clone, Copy, PartialEq, Eq, Hash, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic)]
pub struct GenSig<I: Interner> {
    pub resume_ty: I::Ty,
    pub yield_ty: I::Ty,
    pub return_ty: I::Ty,
}

/// Similar to `ClosureArgs`; see the above documentation for more.
#[derive_where(Clone, Copy, PartialEq, Eq, Hash, Debug; I: Interner)]
#[derive(TypeVisitable_Generic, TypeFoldable_Generic, Lift_Generic)]
pub struct CoroutineArgs<I: Interner> {
    pub args: I::GenericArgs,
}

pub struct CoroutineArgsParts<I: Interner> {
    /// This is the args of the typeck root.
    pub parent_args: I::GenericArgsSlice,

    /// The coroutines returned by a coroutine-closure's `AsyncFnOnce`/`AsyncFnMut`
    /// implementations must be distinguished since the former takes the closure's
    /// upvars by move, and the latter takes the closure's upvars by ref.
    ///
    /// This field distinguishes these fields so that codegen can select the right
    /// body for the coroutine. This has the same type representation as the closure
    /// kind: `i8`/`i16`/`i32`.
    ///
    /// For regular coroutines, this field will always just be `()`.
    pub kind_ty: I::Ty,

    pub resume_ty: I::Ty,
    pub yield_ty: I::Ty,
    pub return_ty: I::Ty,

    /// The interior type of the coroutine.
    /// Represents all types that are stored in locals
    /// in the coroutine's body.
    pub witness: I::Ty,

    /// The upvars captured by the closure. Remains an inference variable
    /// until the upvar analysis, which happens late in HIR typeck.
    pub tupled_upvars_ty: I::Ty,
}

impl<I: Interner> CoroutineArgs<I> {
    /// Construct `CoroutineArgs` from `CoroutineArgsParts`, containing `Args`
    /// for the coroutine parent, alongside additional coroutine-specific components.
    pub fn new(cx: I, parts: CoroutineArgsParts<I>) -> CoroutineArgs<I> {
        CoroutineArgs {
            args: cx.mk_args_from_iter(parts.parent_args.iter().chain([
                parts.kind_ty.into(),
                parts.resume_ty.into(),
                parts.yield_ty.into(),
                parts.return_ty.into(),
                parts.witness.into(),
                parts.tupled_upvars_ty.into(),
            ])),
        }
    }

    /// Divides the coroutine args into their respective components.
    /// The ordering assumed here must match that used by `CoroutineArgs::new` above.
    fn split(self) -> CoroutineArgsParts<I> {
        self.args.split_coroutine_args()
    }

    /// Returns the generic parameters of the coroutine's parent.
    pub fn parent_args(self) -> I::GenericArgsSlice {
        self.split().parent_args
    }

    // Returns the kind of the coroutine. See docs on the `kind_ty` field.
    pub fn kind_ty(self) -> I::Ty {
        self.split().kind_ty
    }

    /// This describes the types that can be contained in a coroutine.
    /// It will be a type variable initially and unified in the last stages of typeck of a body.
    /// It contains a tuple of all the types that could end up on a coroutine frame.
    /// The state transformation MIR pass may only produce layouts which mention types
    /// in this tuple. Upvars are not counted here.
    pub fn witness(self) -> I::Ty {
        self.split().witness
    }

    /// Returns an iterator over the list of types of captured paths by the coroutine.
    /// In case there was a type error in figuring out the types of the captured path, an
    /// empty iterator is returned.
    #[inline]
    pub fn upvar_tys(self) -> I::Tys {
        match self.tupled_upvars_ty().kind() {
            ty::Error(_) => Default::default(),
            ty::Tuple(tys) => tys,
            ty::Infer(_) => panic!("upvar_tys called before capture types are inferred"),
            ty => panic!("Unexpected representation of upvar types tuple {:?}", ty),
        }
    }

    /// Returns the tuple type representing the upvars for this coroutine.
    #[inline]
    pub fn tupled_upvars_ty(self) -> I::Ty {
        self.split().tupled_upvars_ty
    }

    /// Returns the type representing the resume type of the coroutine.
    pub fn resume_ty(self) -> I::Ty {
        self.split().resume_ty
    }

    /// Returns the type representing the yield type of the coroutine.
    pub fn yield_ty(self) -> I::Ty {
        self.split().yield_ty
    }

    /// Returns the type representing the return type of the coroutine.
    pub fn return_ty(self) -> I::Ty {
        self.split().return_ty
    }

    /// Returns the "coroutine signature", which consists of its resume, yield
    /// and return types.
    pub fn sig(self) -> GenSig<I> {
        let parts = self.split();
        GenSig { resume_ty: parts.resume_ty, yield_ty: parts.yield_ty, return_ty: parts.return_ty }
    }
}
