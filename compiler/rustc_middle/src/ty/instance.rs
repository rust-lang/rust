use std::assert_matches::assert_matches;
use std::fmt;

use rustc_data_structures::fx::FxHashMap;
use rustc_errors::ErrorGuaranteed;
use rustc_hir as hir;
use rustc_hir::def::{CtorKind, DefKind, Namespace};
use rustc_hir::def_id::{CrateNum, DefId};
use rustc_hir::lang_items::LangItem;
use rustc_index::bit_set::FiniteBitSet;
use rustc_macros::{Decodable, Encodable, HashStable, Lift, TyDecodable, TyEncodable};
use rustc_span::def_id::LOCAL_CRATE;
use rustc_span::{DUMMY_SP, Span, Symbol};
use tracing::{debug, instrument};

use crate::error;
use crate::middle::codegen_fn_attrs::CodegenFnAttrFlags;
use crate::ty::normalize_erasing_regions::NormalizationError;
use crate::ty::print::{FmtPrinter, Print};
use crate::ty::{
    self, AssocContainer, EarlyBinder, GenericArgs, GenericArgsRef, Ty, TyCtxt, TypeFoldable,
    TypeSuperVisitable, TypeVisitable, TypeVisitableExt, TypeVisitor,
};

/// An `InstanceKind` along with the args that are needed to substitute the instance.
///
/// Monomorphization happens on-the-fly and no monomorphized MIR is ever created. Instead, this type
/// simply couples a potentially generic `InstanceKind` with some args, and codegen and const eval
/// will do all required instantiations as they run.
///
/// Note: the `Lift` impl is currently not used by rustc, but is used by
/// rustc_codegen_cranelift when the `jit` feature is enabled.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, TyEncodable, TyDecodable)]
#[derive(HashStable, Lift, TypeFoldable, TypeVisitable)]
pub struct Instance<'tcx> {
    pub def: InstanceKind<'tcx>,
    pub args: GenericArgsRef<'tcx>,
}

/// Describes why a `ReifyShim` was created. This is needed to distinguish a ReifyShim created to
/// adjust for things like `#[track_caller]` in a vtable from a `ReifyShim` created to produce a
/// function pointer from a vtable entry.
/// Currently, this is only used when KCFI is enabled, as only KCFI needs to treat those two
/// `ReifyShim`s differently.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[derive(TyEncodable, TyDecodable, HashStable)]
pub enum ReifyReason {
    /// The `ReifyShim` was created to produce a function pointer. This happens when:
    /// * A vtable entry is directly converted to a function call (e.g. creating a fn ptr from a
    ///   method on a `dyn` object).
    /// * A function with `#[track_caller]` is converted to a function pointer
    /// * If KCFI is enabled, creating a function pointer from a method on a dyn-compatible trait.
    /// This includes the case of converting `::call`-like methods on closure-likes to function
    /// pointers.
    FnPtr,
    /// This `ReifyShim` was created to populate a vtable. Currently, this happens when a
    /// `#[track_caller]` mismatch occurs between the implementation of a method and the method.
    /// This includes the case of `::call`-like methods in closure-likes' vtables.
    Vtable,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
#[derive(TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable, Lift)]
pub enum InstanceKind<'tcx> {
    /// A user-defined callable item.
    ///
    /// This includes:
    /// - `fn` items
    /// - closures
    /// - coroutines
    Item(DefId),

    /// An intrinsic `fn` item (with`#[rustc_intrinsic]`).
    ///
    /// Alongside `Virtual`, this is the only `InstanceKind` that does not have its own callable MIR.
    /// Instead, codegen and const eval "magically" evaluate calls to intrinsics purely in the
    /// caller.
    Intrinsic(DefId),

    /// `<T as Trait>::method` where `method` receives unsizeable `self: Self` (part of the
    /// `unsized_fn_params` feature).
    ///
    /// The generated shim will take `Self` via `*mut Self` - conceptually this is `&owned Self` -
    /// and dereference the argument to call the original function.
    VTableShim(DefId),

    /// `fn()` pointer where the function itself cannot be turned into a pointer.
    ///
    /// One example is `<dyn Trait as Trait>::fn`, where the shim contains
    /// a virtual call, which codegen supports only via a direct call to the
    /// `<dyn Trait as Trait>::fn` instance (an `InstanceKind::Virtual`).
    ///
    /// Another example is functions annotated with `#[track_caller]`, which
    /// must have their implicit caller location argument populated for a call.
    /// Because this is a required part of the function's ABI but can't be tracked
    /// as a property of the function pointer, we use a single "caller location"
    /// (the definition of the function itself).
    ///
    /// The second field encodes *why* this shim was created. This allows distinguishing between
    /// a `ReifyShim` that appears in a vtable vs one that appears as a function pointer.
    ///
    /// This field will only be populated if we are compiling in a mode that needs these shims
    /// to be separable, currently only when KCFI is enabled.
    ReifyShim(DefId, Option<ReifyReason>),

    /// `<fn() as FnTrait>::call_*` (generated `FnTrait` implementation for `fn()` pointers).
    ///
    /// `DefId` is `FnTrait::call_*`.
    FnPtrShim(DefId, Ty<'tcx>),

    /// Dynamic dispatch to `<dyn Trait as Trait>::fn`.
    ///
    /// This `InstanceKind` may have a callable MIR as the default implementation.
    /// Calls to `Virtual` instances must be codegen'd as virtual calls through the vtable.
    /// *This means we might not know exactly what is being called.*
    ///
    /// If this is reified to a `fn` pointer, a `ReifyShim` is used (see `ReifyShim` above for more
    /// details on that).
    Virtual(DefId, usize),

    /// `<[FnMut/Fn closure] as FnOnce>::call_once`.
    ///
    /// The `DefId` is the ID of the `call_once` method in `FnOnce`.
    ///
    /// This generates a body that will just borrow the (owned) self type,
    /// and dispatch to the `FnMut::call_mut` instance for the closure.
    ClosureOnceShim { call_once: DefId, track_caller: bool },

    /// `<[FnMut/Fn coroutine-closure] as FnOnce>::call_once`
    ///
    /// The body generated here differs significantly from the `ClosureOnceShim`,
    /// since we need to generate a distinct coroutine type that will move the
    /// closure's upvars *out* of the closure.
    ConstructCoroutineInClosureShim {
        coroutine_closure_def_id: DefId,
        // Whether the generated MIR body takes the coroutine by-ref. This is
        // because the signature of `<{async fn} as FnMut>::call_mut` is:
        // `fn(&mut self, args: A) -> <Self as FnOnce>::Output`, that is to say
        // that it returns the `FnOnce`-flavored coroutine but takes the closure
        // by mut ref (and similarly for `Fn::call`).
        receiver_by_ref: bool,
    },

    /// Compiler-generated accessor for thread locals which returns a reference to the thread local
    /// the `DefId` defines. This is used to export thread locals from dylibs on platforms lacking
    /// native support.
    ThreadLocalShim(DefId),

    /// Proxy shim for async drop of future (def_id, proxy_cor_ty, impl_cor_ty)
    FutureDropPollShim(DefId, Ty<'tcx>, Ty<'tcx>),

    /// `core::ptr::drop_in_place::<T>`.
    ///
    /// The `DefId` is for `core::ptr::drop_in_place`.
    /// The `Option<Ty<'tcx>>` is either `Some(T)`, or `None` for empty drop
    /// glue.
    DropGlue(DefId, Option<Ty<'tcx>>),

    /// Compiler-generated `<T as Clone>::clone` implementation.
    ///
    /// For all types that automatically implement `Copy`, a trivial `Clone` impl is provided too.
    /// Additionally, arrays, tuples, and closures get a `Clone` shim even if they aren't `Copy`.
    ///
    /// The `DefId` is for `Clone::clone`, the `Ty` is the type `T` with the builtin `Clone` impl.
    CloneShim(DefId, Ty<'tcx>),

    /// Compiler-generated `<T as FnPtr>::addr` implementation.
    ///
    /// Automatically generated for all potentially higher-ranked `fn(I) -> R` types.
    ///
    /// The `DefId` is for `FnPtr::addr`, the `Ty` is the type `T`.
    FnPtrAddrShim(DefId, Ty<'tcx>),

    /// `core::future::async_drop::async_drop_in_place::<'_, T>`.
    ///
    /// The `DefId` is for `core::future::async_drop::async_drop_in_place`, the `Ty`
    /// is the type `T`.
    AsyncDropGlueCtorShim(DefId, Ty<'tcx>),

    /// `core::future::async_drop::async_drop_in_place::<'_, T>::{closure}`.
    ///
    /// async_drop_in_place poll function implementation (for generated coroutine).
    /// `Ty` here is `async_drop_in_place<T>::{closure}` coroutine type, not just `T`
    AsyncDropGlue(DefId, Ty<'tcx>),
}

impl<'tcx> Instance<'tcx> {
    /// Returns the `Ty` corresponding to this `Instance`, with generic instantiations applied and
    /// lifetimes erased, allowing a `ParamEnv` to be specified for use during normalization.
    pub fn ty(&self, tcx: TyCtxt<'tcx>, typing_env: ty::TypingEnv<'tcx>) -> Ty<'tcx> {
        let ty = tcx.type_of(self.def.def_id());
        tcx.instantiate_and_normalize_erasing_regions(self.args, typing_env, ty)
    }

    /// Finds a crate that contains a monomorphization of this instance that
    /// can be linked to from the local crate. A return value of `None` means
    /// no upstream crate provides such an exported monomorphization.
    ///
    /// This method already takes into account the global `-Zshare-generics`
    /// setting, always returning `None` if `share-generics` is off.
    pub fn upstream_monomorphization(&self, tcx: TyCtxt<'tcx>) -> Option<CrateNum> {
        // If this is an item that is defined in the local crate, no upstream
        // crate can know about it/provide a monomorphization.
        if self.def_id().is_local() {
            return None;
        }

        // If we are not in share generics mode, we don't link to upstream
        // monomorphizations but always instantiate our own internal versions
        // instead.
        if !tcx.sess.opts.share_generics()
            // However, if the def_id is marked inline(never), then it's fine to just reuse the
            // upstream monomorphization.
            && tcx.codegen_fn_attrs(self.def_id()).inline != rustc_hir::attrs::InlineAttr::Never
        {
            return None;
        }

        // If this a non-generic instance, it cannot be a shared monomorphization.
        self.args.non_erasable_generics().next()?;

        // compiler_builtins cannot use upstream monomorphizations.
        if tcx.is_compiler_builtins(LOCAL_CRATE) {
            return None;
        }

        match self.def {
            InstanceKind::Item(def) => tcx
                .upstream_monomorphizations_for(def)
                .and_then(|monos| monos.get(&self.args).cloned()),
            InstanceKind::DropGlue(_, Some(_)) => tcx.upstream_drop_glue_for(self.args),
            InstanceKind::AsyncDropGlue(_, _) => None,
            InstanceKind::FutureDropPollShim(_, _, _) => None,
            InstanceKind::AsyncDropGlueCtorShim(_, _) => {
                tcx.upstream_async_drop_glue_for(self.args)
            }
            _ => None,
        }
    }
}

impl<'tcx> InstanceKind<'tcx> {
    #[inline]
    pub fn def_id(self) -> DefId {
        match self {
            InstanceKind::Item(def_id)
            | InstanceKind::VTableShim(def_id)
            | InstanceKind::ReifyShim(def_id, _)
            | InstanceKind::FnPtrShim(def_id, _)
            | InstanceKind::Virtual(def_id, _)
            | InstanceKind::Intrinsic(def_id)
            | InstanceKind::ThreadLocalShim(def_id)
            | InstanceKind::ClosureOnceShim { call_once: def_id, track_caller: _ }
            | ty::InstanceKind::ConstructCoroutineInClosureShim {
                coroutine_closure_def_id: def_id,
                receiver_by_ref: _,
            }
            | InstanceKind::DropGlue(def_id, _)
            | InstanceKind::CloneShim(def_id, _)
            | InstanceKind::FnPtrAddrShim(def_id, _)
            | InstanceKind::FutureDropPollShim(def_id, _, _)
            | InstanceKind::AsyncDropGlue(def_id, _)
            | InstanceKind::AsyncDropGlueCtorShim(def_id, _) => def_id,
        }
    }

    /// Returns the `DefId` of instances which might not require codegen locally.
    pub fn def_id_if_not_guaranteed_local_codegen(self) -> Option<DefId> {
        match self {
            ty::InstanceKind::Item(def) => Some(def),
            ty::InstanceKind::DropGlue(def_id, Some(_))
            | InstanceKind::AsyncDropGlueCtorShim(def_id, _)
            | InstanceKind::AsyncDropGlue(def_id, _)
            | InstanceKind::FutureDropPollShim(def_id, ..)
            | InstanceKind::ThreadLocalShim(def_id) => Some(def_id),
            InstanceKind::VTableShim(..)
            | InstanceKind::ReifyShim(..)
            | InstanceKind::FnPtrShim(..)
            | InstanceKind::Virtual(..)
            | InstanceKind::Intrinsic(..)
            | InstanceKind::ClosureOnceShim { .. }
            | ty::InstanceKind::ConstructCoroutineInClosureShim { .. }
            | InstanceKind::DropGlue(..)
            | InstanceKind::CloneShim(..)
            | InstanceKind::FnPtrAddrShim(..) => None,
        }
    }

    #[inline]
    pub fn get_attrs(
        &self,
        tcx: TyCtxt<'tcx>,
        attr: Symbol,
    ) -> impl Iterator<Item = &'tcx hir::Attribute> {
        tcx.get_attrs(self.def_id(), attr)
    }

    /// Returns `true` if the LLVM version of this instance is unconditionally
    /// marked with `inline`. This implies that a copy of this instance is
    /// generated in every codegen unit.
    /// Note that this is only a hint. See the documentation for
    /// `generates_cgu_internal_copy` for more information.
    pub fn requires_inline(&self, tcx: TyCtxt<'tcx>) -> bool {
        use rustc_hir::definitions::DefPathData;
        let def_id = match *self {
            ty::InstanceKind::Item(def) => def,
            ty::InstanceKind::DropGlue(_, Some(_)) => return false,
            ty::InstanceKind::AsyncDropGlueCtorShim(_, ty) => return ty.is_coroutine(),
            ty::InstanceKind::FutureDropPollShim(_, _, _) => return false,
            ty::InstanceKind::AsyncDropGlue(_, _) => return false,
            ty::InstanceKind::ThreadLocalShim(_) => return false,
            _ => return true,
        };
        matches!(
            tcx.def_key(def_id).disambiguated_data.data,
            DefPathData::Ctor | DefPathData::Closure
        )
    }

    pub fn requires_caller_location(&self, tcx: TyCtxt<'_>) -> bool {
        match *self {
            InstanceKind::Item(def_id) | InstanceKind::Virtual(def_id, _) => {
                tcx.body_codegen_attrs(def_id).flags.contains(CodegenFnAttrFlags::TRACK_CALLER)
            }
            InstanceKind::ClosureOnceShim { call_once: _, track_caller } => track_caller,
            _ => false,
        }
    }

    /// Returns `true` when the MIR body associated with this instance should be monomorphized
    /// by its users (e.g. codegen or miri) by instantiating the `args` from `Instance` (see
    /// `Instance::args_for_mir_body`).
    ///
    /// Otherwise, returns `false` only for some kinds of shims where the construction of the MIR
    /// body should perform necessary instantiations.
    pub fn has_polymorphic_mir_body(&self) -> bool {
        match *self {
            InstanceKind::CloneShim(..)
            | InstanceKind::ThreadLocalShim(..)
            | InstanceKind::FnPtrAddrShim(..)
            | InstanceKind::FnPtrShim(..)
            | InstanceKind::DropGlue(_, Some(_))
            | InstanceKind::FutureDropPollShim(..)
            | InstanceKind::AsyncDropGlue(_, _) => false,
            InstanceKind::AsyncDropGlueCtorShim(_, _) => false,
            InstanceKind::ClosureOnceShim { .. }
            | InstanceKind::ConstructCoroutineInClosureShim { .. }
            | InstanceKind::DropGlue(..)
            | InstanceKind::Item(_)
            | InstanceKind::Intrinsic(..)
            | InstanceKind::ReifyShim(..)
            | InstanceKind::Virtual(..)
            | InstanceKind::VTableShim(..) => true,
        }
    }
}

fn type_length<'tcx>(item: impl TypeVisitable<TyCtxt<'tcx>>) -> usize {
    struct Visitor<'tcx> {
        type_length: usize,
        cache: FxHashMap<Ty<'tcx>, usize>,
    }
    impl<'tcx> TypeVisitor<TyCtxt<'tcx>> for Visitor<'tcx> {
        fn visit_ty(&mut self, t: Ty<'tcx>) {
            if let Some(&value) = self.cache.get(&t) {
                self.type_length += value;
                return;
            }

            let prev = self.type_length;
            self.type_length += 1;
            t.super_visit_with(self);

            // We don't try to use the cache if the type is fairly small.
            if self.type_length > 16 {
                self.cache.insert(t, self.type_length - prev);
            }
        }

        fn visit_const(&mut self, ct: ty::Const<'tcx>) {
            self.type_length += 1;
            ct.super_visit_with(self);
        }
    }
    let mut visitor = Visitor { type_length: 0, cache: Default::default() };
    item.visit_with(&mut visitor);

    visitor.type_length
}

impl<'tcx> fmt::Display for Instance<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        ty::tls::with(|tcx| {
            let mut p = FmtPrinter::new(tcx, Namespace::ValueNS);
            let instance = tcx.lift(*self).expect("could not lift for printing");
            instance.print(&mut p)?;
            let s = p.into_buffer();
            f.write_str(&s)
        })
    }
}

// async_drop_in_place<T>::coroutine.poll, when T is a standard coroutine,
// should be resolved to this coroutine's future_drop_poll (through FutureDropPollShim proxy).
// async_drop_in_place<async_drop_in_place<T>::coroutine>::coroutine.poll,
// when T is a standard coroutine, should be resolved to this coroutine's future_drop_poll.
// async_drop_in_place<async_drop_in_place<T>::coroutine>::coroutine.poll,
// when T is not a coroutine, should be resolved to the innermost
// async_drop_in_place<T>::coroutine's poll function (through FutureDropPollShim proxy)
fn resolve_async_drop_poll<'tcx>(mut cor_ty: Ty<'tcx>) -> Instance<'tcx> {
    let first_cor = cor_ty;
    let ty::Coroutine(poll_def_id, proxy_args) = first_cor.kind() else {
        bug!();
    };
    let poll_def_id = *poll_def_id;
    let mut child_ty = cor_ty;
    loop {
        if let ty::Coroutine(child_def, child_args) = child_ty.kind() {
            cor_ty = child_ty;
            if *child_def == poll_def_id {
                child_ty = child_args.first().unwrap().expect_ty();
                continue;
            } else {
                return Instance {
                    def: ty::InstanceKind::FutureDropPollShim(poll_def_id, first_cor, cor_ty),
                    args: proxy_args,
                };
            }
        } else {
            let ty::Coroutine(_, child_args) = cor_ty.kind() else {
                bug!();
            };
            if first_cor != cor_ty {
                return Instance {
                    def: ty::InstanceKind::FutureDropPollShim(poll_def_id, first_cor, cor_ty),
                    args: proxy_args,
                };
            } else {
                return Instance {
                    def: ty::InstanceKind::AsyncDropGlue(poll_def_id, cor_ty),
                    args: child_args,
                };
            }
        }
    }
}

impl<'tcx> Instance<'tcx> {
    /// Creates a new [`InstanceKind::Item`] from the `def_id` and `args`.
    ///
    /// Note that this item corresponds to the body of `def_id` directly, which
    /// likely does not make sense for trait items which need to be resolved to an
    /// implementation, and which may not even have a body themselves. Usages of
    /// this function should probably use [`Instance::expect_resolve`], or if run
    /// in a polymorphic environment or within a lint (that may encounter ambiguity)
    /// [`Instance::try_resolve`] instead.
    pub fn new_raw(def_id: DefId, args: GenericArgsRef<'tcx>) -> Instance<'tcx> {
        assert!(
            !args.has_escaping_bound_vars(),
            "args of instance {def_id:?} has escaping bound vars: {args:?}"
        );
        Instance { def: InstanceKind::Item(def_id), args }
    }

    pub fn mono(tcx: TyCtxt<'tcx>, def_id: DefId) -> Instance<'tcx> {
        let args = GenericArgs::for_item(tcx, def_id, |param, _| match param.kind {
            ty::GenericParamDefKind::Lifetime => tcx.lifetimes.re_erased.into(),
            ty::GenericParamDefKind::Type { .. } => {
                bug!("Instance::mono: {:?} has type parameters", def_id)
            }
            ty::GenericParamDefKind::Const { .. } => {
                bug!("Instance::mono: {:?} has const parameters", def_id)
            }
        });

        Instance::new_raw(def_id, args)
    }

    #[inline]
    pub fn def_id(&self) -> DefId {
        self.def.def_id()
    }

    /// Resolves a `(def_id, args)` pair to an (optional) instance -- most commonly,
    /// this is used to find the precise code that will run for a trait method invocation,
    /// if known. This should only be used for functions and consts. If you want to
    /// resolve an associated type, use [`TyCtxt::try_normalize_erasing_regions`].
    ///
    /// Returns `Ok(None)` if we cannot resolve `Instance` to a specific instance.
    /// For example, in a context like this,
    ///
    /// ```ignore (illustrative)
    /// fn foo<T: Debug>(t: T) { ... }
    /// ```
    ///
    /// trying to resolve `Debug::fmt` applied to `T` will yield `Ok(None)`, because we do not
    /// know what code ought to run. This setting is also affected by the current `TypingMode`
    /// of the environment.
    ///
    /// Presuming that coherence and type-check have succeeded, if this method is invoked
    /// in a monomorphic context (i.e., like during codegen), then it is guaranteed to return
    /// `Ok(Some(instance))`, **except** for when the instance's inputs hit the type size limit,
    /// in which case it may bail out and return `Ok(None)`.
    ///
    /// Returns `Err(ErrorGuaranteed)` when the `Instance` resolution process
    /// couldn't complete due to errors elsewhere - this is distinct
    /// from `Ok(None)` to avoid misleading diagnostics when an error
    /// has already been/will be emitted, for the original cause
    #[instrument(level = "debug", skip(tcx), ret)]
    pub fn try_resolve(
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> Result<Option<Instance<'tcx>>, ErrorGuaranteed> {
        assert_matches!(
            tcx.def_kind(def_id),
            DefKind::Fn
                | DefKind::AssocFn
                | DefKind::Const
                | DefKind::AssocConst
                | DefKind::AnonConst
                | DefKind::InlineConst
                | DefKind::Static { .. }
                | DefKind::Ctor(_, CtorKind::Fn)
                | DefKind::Closure
                | DefKind::SyntheticCoroutineBody,
            "`Instance::try_resolve` should only be used to resolve instances of \
            functions, statics, and consts; to resolve associated types, use \
            `try_normalize_erasing_regions`."
        );

        // Rust code can easily create exponentially-long types using only a
        // polynomial recursion depth. Even with the default recursion
        // depth, you can easily get cases that take >2^60 steps to run,
        // which means that rustc basically hangs.
        //
        // Bail out in these cases to avoid that bad user experience.
        if tcx.sess.opts.unstable_opts.enforce_type_length_limit
            && !tcx.type_length_limit().value_within_limit(type_length(args))
        {
            return Ok(None);
        }

        // All regions in the result of this query are erased, so it's
        // fine to erase all of the input regions.
        tcx.resolve_instance_raw(
            tcx.erase_and_anonymize_regions(typing_env.as_query_input((def_id, args))),
        )
    }

    pub fn expect_resolve(
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
        span: Span,
    ) -> Instance<'tcx> {
        // We compute the span lazily, to avoid unnecessary query calls.
        // If `span` is a DUMMY_SP, and the def id is local, then use the
        // def span of the def id.
        let span_or_local_def_span =
            || if span.is_dummy() && def_id.is_local() { tcx.def_span(def_id) } else { span };

        match ty::Instance::try_resolve(tcx, typing_env, def_id, args) {
            Ok(Some(instance)) => instance,
            Ok(None) => {
                let type_length = type_length(args);
                if !tcx.type_length_limit().value_within_limit(type_length) {
                    tcx.dcx().emit_fatal(error::TypeLengthLimit {
                        // We don't use `def_span(def_id)` so that diagnostics point
                        // to the crate root during mono instead of to foreign items.
                        // This is arguably better.
                        span: span_or_local_def_span(),
                        instance: Instance::new_raw(def_id, args),
                        type_length,
                    });
                } else {
                    span_bug!(
                        span_or_local_def_span(),
                        "failed to resolve instance for {}",
                        tcx.def_path_str_with_args(def_id, args)
                    )
                }
            }
            instance => span_bug!(
                span_or_local_def_span(),
                "failed to resolve instance for {}: {instance:#?}",
                tcx.def_path_str_with_args(def_id, args)
            ),
        }
    }

    pub fn resolve_for_fn_ptr(
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
    ) -> Option<Instance<'tcx>> {
        debug!("resolve(def_id={:?}, args={:?})", def_id, args);
        // Use either `resolve_closure` or `resolve_for_vtable`
        assert!(!tcx.is_closure_like(def_id), "Called `resolve_for_fn_ptr` on closure: {def_id:?}");
        let reason = tcx.sess.is_sanitizer_kcfi_enabled().then_some(ReifyReason::FnPtr);
        Instance::try_resolve(tcx, typing_env, def_id, args).ok().flatten().map(|mut resolved| {
            match resolved.def {
                InstanceKind::Item(def) if resolved.def.requires_caller_location(tcx) => {
                    debug!(" => fn pointer created for function with #[track_caller]");
                    resolved.def = InstanceKind::ReifyShim(def, reason);
                }
                InstanceKind::Virtual(def_id, _) => {
                    debug!(" => fn pointer created for virtual call");
                    resolved.def = InstanceKind::ReifyShim(def_id, reason);
                }
                _ if tcx.sess.is_sanitizer_kcfi_enabled() => {
                    // Reify `::call`-like method implementations
                    if tcx.is_closure_like(resolved.def_id()) {
                        // Reroute through a reify via the *unresolved* instance. The resolved one can't
                        // be directly reified because it's closure-like. The reify can handle the
                        // unresolved instance.
                        resolved = Instance { def: InstanceKind::ReifyShim(def_id, reason), args }
                    // Reify `Trait::method` implementations
                    // FIXME(maurer) only reify it if it is a vtable-safe function
                    } else if let Some(assoc) = tcx.opt_associated_item(def_id)
                        && let AssocContainer::Trait | AssocContainer::TraitImpl(Ok(_)) =
                            assoc.container
                    {
                        // If this function could also go in a vtable, we need to `ReifyShim` it with
                        // KCFI because it can only attach one type per function.
                        resolved.def = InstanceKind::ReifyShim(resolved.def_id(), reason)
                    }
                }
                _ => {}
            }

            resolved
        })
    }

    pub fn expect_resolve_for_vtable(
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        def_id: DefId,
        args: GenericArgsRef<'tcx>,
        span: Span,
    ) -> Instance<'tcx> {
        debug!("resolve_for_vtable(def_id={:?}, args={:?})", def_id, args);
        let fn_sig = tcx.fn_sig(def_id).instantiate_identity();
        let is_vtable_shim = !fn_sig.inputs().skip_binder().is_empty()
            && fn_sig.input(0).skip_binder().is_param(0)
            && tcx.generics_of(def_id).has_self;

        if is_vtable_shim {
            debug!(" => associated item with unsizeable self: Self");
            return Instance { def: InstanceKind::VTableShim(def_id), args };
        }

        let mut resolved = Instance::expect_resolve(tcx, typing_env, def_id, args, span);

        let reason = tcx.sess.is_sanitizer_kcfi_enabled().then_some(ReifyReason::Vtable);
        match resolved.def {
            InstanceKind::Item(def) => {
                // We need to generate a shim when we cannot guarantee that
                // the caller of a trait object method will be aware of
                // `#[track_caller]` - this ensures that the caller
                // and callee ABI will always match.
                //
                // The shim is generated when all of these conditions are met:
                //
                // 1) The underlying method expects a caller location parameter
                // in the ABI
                let needs_track_caller_shim = resolved.def.requires_caller_location(tcx)
                    // 2) The caller location parameter comes from having `#[track_caller]`
                    // on the implementation, and *not* on the trait method.
                    && !tcx.should_inherit_track_caller(def)
                    // If the method implementation comes from the trait definition itself
                    // (e.g. `trait Foo { #[track_caller] my_fn() { /* impl */ } }`),
                    // then we don't need to generate a shim. This check is needed because
                    // `should_inherit_track_caller` returns `false` if our method
                    // implementation comes from the trait block, and not an impl block
                    && !matches!(
                        tcx.opt_associated_item(def),
                        Some(ty::AssocItem {
                            container: ty::AssocContainer::Trait,
                            ..
                        })
                    );
                if needs_track_caller_shim {
                    if tcx.is_closure_like(def) {
                        debug!(
                            " => vtable fn pointer created for closure with #[track_caller]: {:?} for method {:?} {:?}",
                            def, def_id, args
                        );

                        // Create a shim for the `FnOnce/FnMut/Fn` method we are calling
                        // - unlike functions, invoking a closure always goes through a
                        // trait.
                        resolved = Instance { def: InstanceKind::ReifyShim(def_id, reason), args };
                    } else {
                        debug!(
                            " => vtable fn pointer created for function with #[track_caller]: {:?}",
                            def
                        );
                        resolved.def = InstanceKind::ReifyShim(def, reason);
                    }
                }
            }
            InstanceKind::Virtual(def_id, _) => {
                debug!(" => vtable fn pointer created for virtual call");
                resolved.def = InstanceKind::ReifyShim(def_id, reason)
            }
            _ => {}
        }

        resolved
    }

    pub fn resolve_closure(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        args: ty::GenericArgsRef<'tcx>,
        requested_kind: ty::ClosureKind,
    ) -> Instance<'tcx> {
        let actual_kind = args.as_closure().kind();

        match needs_fn_once_adapter_shim(actual_kind, requested_kind) {
            Ok(true) => Instance::fn_once_adapter_instance(tcx, def_id, args),
            _ => Instance::new_raw(def_id, args),
        }
    }

    pub fn resolve_drop_in_place(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> ty::Instance<'tcx> {
        let def_id = tcx.require_lang_item(LangItem::DropInPlace, DUMMY_SP);
        let args = tcx.mk_args(&[ty.into()]);
        Instance::expect_resolve(
            tcx,
            ty::TypingEnv::fully_monomorphized(),
            def_id,
            args,
            ty.ty_adt_def().and_then(|adt| tcx.hir_span_if_local(adt.did())).unwrap_or(DUMMY_SP),
        )
    }

    pub fn resolve_async_drop_in_place(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> ty::Instance<'tcx> {
        let def_id = tcx.require_lang_item(LangItem::AsyncDropInPlace, DUMMY_SP);
        let args = tcx.mk_args(&[ty.into()]);
        Instance::expect_resolve(
            tcx,
            ty::TypingEnv::fully_monomorphized(),
            def_id,
            args,
            ty.ty_adt_def().and_then(|adt| tcx.hir_span_if_local(adt.did())).unwrap_or(DUMMY_SP),
        )
    }

    pub fn resolve_async_drop_in_place_poll(
        tcx: TyCtxt<'tcx>,
        def_id: DefId,
        ty: Ty<'tcx>,
    ) -> ty::Instance<'tcx> {
        let args = tcx.mk_args(&[ty.into()]);
        Instance::expect_resolve(tcx, ty::TypingEnv::fully_monomorphized(), def_id, args, DUMMY_SP)
    }

    #[instrument(level = "debug", skip(tcx), ret)]
    pub fn fn_once_adapter_instance(
        tcx: TyCtxt<'tcx>,
        closure_did: DefId,
        args: ty::GenericArgsRef<'tcx>,
    ) -> Instance<'tcx> {
        let fn_once = tcx.require_lang_item(LangItem::FnOnce, DUMMY_SP);
        let call_once = tcx
            .associated_items(fn_once)
            .in_definition_order()
            .find(|it| it.is_fn())
            .unwrap()
            .def_id;
        let track_caller =
            tcx.codegen_fn_attrs(closure_did).flags.contains(CodegenFnAttrFlags::TRACK_CALLER);
        let def = ty::InstanceKind::ClosureOnceShim { call_once, track_caller };

        let self_ty = Ty::new_closure(tcx, closure_did, args);

        let tupled_inputs_ty = args.as_closure().sig().map_bound(|sig| sig.inputs()[0]);
        let tupled_inputs_ty = tcx.instantiate_bound_regions_with_erased(tupled_inputs_ty);
        let args = tcx.mk_args_trait(self_ty, [tupled_inputs_ty.into()]);

        debug!(?self_ty, args=?tupled_inputs_ty.tuple_fields());
        Instance { def, args }
    }

    pub fn try_resolve_item_for_coroutine(
        tcx: TyCtxt<'tcx>,
        trait_item_id: DefId,
        trait_id: DefId,
        rcvr_args: ty::GenericArgsRef<'tcx>,
    ) -> Option<Instance<'tcx>> {
        let ty::Coroutine(coroutine_def_id, args) = *rcvr_args.type_at(0).kind() else {
            return None;
        };
        let coroutine_kind = tcx.coroutine_kind(coroutine_def_id).unwrap();

        let coroutine_callable_item = if tcx.is_lang_item(trait_id, LangItem::Future) {
            assert_matches!(
                coroutine_kind,
                hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Async, _)
            );
            hir::LangItem::FuturePoll
        } else if tcx.is_lang_item(trait_id, LangItem::Iterator) {
            assert_matches!(
                coroutine_kind,
                hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::Gen, _)
            );
            hir::LangItem::IteratorNext
        } else if tcx.is_lang_item(trait_id, LangItem::AsyncIterator) {
            assert_matches!(
                coroutine_kind,
                hir::CoroutineKind::Desugared(hir::CoroutineDesugaring::AsyncGen, _)
            );
            hir::LangItem::AsyncIteratorPollNext
        } else if tcx.is_lang_item(trait_id, LangItem::Coroutine) {
            assert_matches!(coroutine_kind, hir::CoroutineKind::Coroutine(_));
            hir::LangItem::CoroutineResume
        } else {
            return None;
        };

        if tcx.is_lang_item(trait_item_id, coroutine_callable_item) {
            if tcx.is_async_drop_in_place_coroutine(coroutine_def_id) {
                return Some(resolve_async_drop_poll(rcvr_args.type_at(0)));
            }
            let ty::Coroutine(_, id_args) = *tcx.type_of(coroutine_def_id).skip_binder().kind()
            else {
                bug!()
            };

            // If the closure's kind ty disagrees with the identity closure's kind ty,
            // then this must be a coroutine generated by one of the `ConstructCoroutineInClosureShim`s.
            if args.as_coroutine().kind_ty() == id_args.as_coroutine().kind_ty() {
                Some(Instance { def: ty::InstanceKind::Item(coroutine_def_id), args })
            } else {
                Some(Instance {
                    def: ty::InstanceKind::Item(
                        tcx.coroutine_by_move_body_def_id(coroutine_def_id),
                    ),
                    args,
                })
            }
        } else {
            // All other methods should be defaulted methods of the built-in trait.
            // This is important for `Iterator`'s combinators, but also useful for
            // adding future default methods to `Future`, for instance.
            debug_assert!(tcx.defaultness(trait_item_id).has_value());
            Some(Instance::new_raw(trait_item_id, rcvr_args))
        }
    }

    /// Depending on the kind of `InstanceKind`, the MIR body associated with an
    /// instance is expressed in terms of the generic parameters of `self.def_id()`, and in other
    /// cases the MIR body is expressed in terms of the types found in the generic parameter array.
    /// In the former case, we want to instantiate those generic types and replace them with the
    /// values from the args when monomorphizing the function body. But in the latter case, we
    /// don't want to do that instantiation, since it has already been done effectively.
    ///
    /// This function returns `Some(args)` in the former case and `None` otherwise -- i.e., if
    /// this function returns `None`, then the MIR body does not require instantiation during
    /// codegen.
    fn args_for_mir_body(&self) -> Option<GenericArgsRef<'tcx>> {
        self.def.has_polymorphic_mir_body().then_some(self.args)
    }

    pub fn instantiate_mir<T>(&self, tcx: TyCtxt<'tcx>, v: EarlyBinder<'tcx, &T>) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>> + Copy,
    {
        let v = v.map_bound(|v| *v);
        if let Some(args) = self.args_for_mir_body() {
            v.instantiate(tcx, args)
        } else {
            v.instantiate_identity()
        }
    }

    #[inline(always)]
    // Keep me in sync with try_instantiate_mir_and_normalize_erasing_regions
    pub fn instantiate_mir_and_normalize_erasing_regions<T>(
        &self,
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        v: EarlyBinder<'tcx, T>,
    ) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        if let Some(args) = self.args_for_mir_body() {
            tcx.instantiate_and_normalize_erasing_regions(args, typing_env, v)
        } else {
            tcx.normalize_erasing_regions(typing_env, v.instantiate_identity())
        }
    }

    #[inline(always)]
    // Keep me in sync with instantiate_mir_and_normalize_erasing_regions
    pub fn try_instantiate_mir_and_normalize_erasing_regions<T>(
        &self,
        tcx: TyCtxt<'tcx>,
        typing_env: ty::TypingEnv<'tcx>,
        v: EarlyBinder<'tcx, T>,
    ) -> Result<T, NormalizationError<'tcx>>
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        if let Some(args) = self.args_for_mir_body() {
            tcx.try_instantiate_and_normalize_erasing_regions(args, typing_env, v)
        } else {
            // We're using `instantiate_identity` as e.g.
            // `FnPtrShim` is separately generated for every
            // instantiation of the `FnDef`, so the MIR body
            // is already instantiated. Any generic parameters it
            // contains are generic parameters from the caller.
            tcx.try_normalize_erasing_regions(typing_env, v.instantiate_identity())
        }
    }
}

fn needs_fn_once_adapter_shim(
    actual_closure_kind: ty::ClosureKind,
    trait_closure_kind: ty::ClosureKind,
) -> Result<bool, ()> {
    match (actual_closure_kind, trait_closure_kind) {
        (ty::ClosureKind::Fn, ty::ClosureKind::Fn)
        | (ty::ClosureKind::FnMut, ty::ClosureKind::FnMut)
        | (ty::ClosureKind::FnOnce, ty::ClosureKind::FnOnce) => {
            // No adapter needed.
            Ok(false)
        }
        (ty::ClosureKind::Fn, ty::ClosureKind::FnMut) => {
            // The closure fn is a `fn(&self, ...)`, but we want a `fn(&mut self, ...)`.
            // At codegen time, these are basically the same, so we can just return the closure.
            Ok(false)
        }
        (ty::ClosureKind::Fn | ty::ClosureKind::FnMut, ty::ClosureKind::FnOnce) => {
            // The closure fn is a `fn(&self, ...)` or `fn(&mut self, ...)`, but
            // we want a `fn(self, ...)`. We can produce this by doing something like:
            //
            //     fn call_once(self, ...) { Fn::call(&self, ...) }
            //     fn call_once(mut self, ...) { FnMut::call_mut(&mut self, ...) }
            //
            // These are both the same at codegen time.
            Ok(true)
        }
        (ty::ClosureKind::FnMut | ty::ClosureKind::FnOnce, _) => Err(()),
    }
}

// Set bits represent unused generic parameters.
// An empty set indicates that all parameters are used.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Decodable, Encodable, HashStable)]
pub struct UnusedGenericParams(FiniteBitSet<u32>);

impl Default for UnusedGenericParams {
    fn default() -> Self {
        UnusedGenericParams::new_all_used()
    }
}

impl UnusedGenericParams {
    pub fn new_all_unused(amount: u32) -> Self {
        let mut bitset = FiniteBitSet::new_empty();
        bitset.set_range(0..amount);
        Self(bitset)
    }

    pub fn new_all_used() -> Self {
        Self(FiniteBitSet::new_empty())
    }

    pub fn mark_used(&mut self, idx: u32) {
        self.0.clear(idx);
    }

    pub fn is_unused(&self, idx: u32) -> bool {
        self.0.contains(idx).unwrap_or(false)
    }

    pub fn is_used(&self, idx: u32) -> bool {
        !self.is_unused(idx)
    }

    pub fn all_used(&self) -> bool {
        self.0.is_empty()
    }

    pub fn bits(&self) -> u32 {
        self.0.0
    }

    pub fn from_bits(bits: u32) -> UnusedGenericParams {
        UnusedGenericParams(FiniteBitSet(bits))
    }
}
