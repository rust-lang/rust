//! To improve compile times and code size for the compiler itself, query
//! values are "erased" in some contexts (e.g. inside in-memory cache types),
//! to reduce the number of generic instantiations created during codegen.
//!
//! See <https://github.com/rust-lang/rust/pull/151715> for some bootstrap-time
//! and performance benchmarks.

use std::ffi::OsStr;
use std::intrinsics::transmute_unchecked;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::Arc;

use rustc_abi::Align;
use rustc_ast as ast;
use rustc_ast::expand::allocator::AllocatorKind;
use rustc_ast::tokenstream::TokenStream;
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::steal::Steal;
use rustc_data_structures::svh::Svh;
use rustc_data_structures::sync::{DynSend, DynSync};
use rustc_hir as hir;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefIdMap;
use rustc_index::IndexVec;
use rustc_middle::traits::solve::QueryResult;
use rustc_middle::traits::solve::inspect::Probe;
use rustc_session::Limits;
use rustc_session::config::{EntryFnType, OptLevel, SymbolManglingVersion};
use rustc_session::cstore::CrateDepKind;
use rustc_span::def_id::{CrateNum, DefId, LocalDefId};
use rustc_span::{ErrorGuaranteed, ExpnId, Span, Spanned, Symbol};
use rustc_target::callconv::FnAbi;
use rustc_target::spec::PanicStrategy;

use crate::infer::canonical::{Canonical, QueryResponse};
use crate::middle::codegen_fn_attrs::SanitizerFnAttrs;
use crate::middle::resolve_bound_vars::ObjectLifetimeDefault;
use crate::middle::stability::DeprecationEntry;
use crate::mir::mono::{MonoItem, NormalizationErrorInMono};
use crate::traits::query::{
    DropckOutlivesResult, MethodAutoderefStepsResult, NoSolution, NormalizationResult,
    OutlivesBound,
};
use crate::traits::{
    CodegenObligationError, EvaluationResult, ImplSource, OverflowError, specialization_graph,
};
use crate::ty::{self, Ty, TyCtxt};
use crate::{mir, thir};

unsafe extern "C" {
    type NoAutoTraits;
}

/// Internal implementation detail of [`Erased`].
#[derive(Copy, Clone)]
pub struct ErasedData<Storage: Copy> {
    /// We use `MaybeUninit` here to make sure it's legal to store a transmuted
    /// value that isn't actually of type `Storage`.
    data: MaybeUninit<Storage>,
    /// `Storage` is an erased type, so we use an external type here to opt-out of auto traits
    /// as those would be incorrect.
    no_auto_traits: PhantomData<NoAutoTraits>,
}

// SAFETY: The bounds on `erase_val` ensure the types we erase are `DynSync` and `DynSend`
unsafe impl<Storage: Copy> DynSync for ErasedData<Storage> {}
unsafe impl<Storage: Copy> DynSend for ErasedData<Storage> {}

/// Trait for types that can be erased into [`Erased<Self>`].
///
/// Erasing and unerasing values is performed by [`erase_val`] and [`restore_val`].
///
/// Most impls are done via the `impl_erasable_for_types_with_no_type_params!`
/// macro. A small number of hand-written generic impls are used for common
/// types like `&T` and `Option<&T>`; these generic impls avoid many concrete
/// entries being needed in the macro.
///
/// FIXME: This whole trait could potentially be replaced by `T: Copy` and the
/// storage type `[u8; size_of::<T>()]` when support for that is more mature.
pub trait Erasable: Copy {
    /// Storage type to used for erased values of this type.
    /// Should be `[u8; N]`, where N is equal to `size_of::<Self>`.
    ///
    /// [`ErasedData`] wraps this storage type in `MaybeUninit` to ensure that
    /// transmutes to/from erased storage are well-defined.
    type Storage: Copy;
}

/// A value of `T` that has been "erased" into some opaque storage type.
///
/// This is helpful for reducing the number of concrete instantiations needed
/// during codegen when building the compiler.
///
/// Using an opaque type alias allows the type checker to enforce that
/// `Erased<T>` and `Erased<U>` are still distinct types, while allowing
/// monomorphization to see that they might actually use the same storage type.
pub type Erased<T: Erasable> = ErasedData<impl Copy>;

/// Erases a value of type `T` into `Erased<T>`.
///
/// `Erased<T>` and `Erased<U>` are type-checked as distinct types, but codegen
/// can see whether they actually have the same storage type.
#[inline(always)]
#[define_opaque(Erased)]
// The `DynSend` and `DynSync` bounds on `T` are used to
// justify the safety of the implementations of these traits for `ErasedData`.
pub fn erase_val<T: Erasable + DynSend + DynSync>(value: T) -> Erased<T> {
    // Ensure the sizes match
    const {
        if size_of::<T>() != size_of::<T::Storage>() {
            panic!("size of T must match erased type <T as Erasable>::Storage")
        }
    };

    ErasedData::<<T as Erasable>::Storage> {
        // `transmute_unchecked` is needed here because it does not have `transmute`'s size check
        // (and thus allows to transmute between `T` and `MaybeUninit<T::Storage>`) (we do the size
        // check ourselves in the `const` block above).
        //
        // `transmute_copy` is also commonly used for this (and it would work here since
        // `Erasable: Copy`), but `transmute_unchecked` better explains the intent.
        //
        // SAFETY: It is safe to transmute to MaybeUninit for types with the same sizes.
        data: unsafe { transmute_unchecked::<T, MaybeUninit<T::Storage>>(value) },
        no_auto_traits: PhantomData,
    }
}

/// Restores an erased value to its real type.
///
/// This relies on the fact that `Erased<T>` and `Erased<U>` are type-checked
/// as distinct types, even if they use the same storage type.
#[inline(always)]
#[define_opaque(Erased)]
pub fn restore_val<T: Erasable>(erased_value: Erased<T>) -> T {
    let ErasedData { data, .. }: ErasedData<<T as Erasable>::Storage> = erased_value;
    // See comment in `erase_val` for why we use `transmute_unchecked`.
    //
    // SAFETY: Due to the use of impl Trait in `Erased` the only way to safely create an instance
    // of `Erased` is to call `erase_val`, so we know that `erased_value.data` is a valid instance
    // of `T` of the right size.
    unsafe { transmute_unchecked::<MaybeUninit<T::Storage>, T>(data) }
}

impl<T> Erasable for &'_ T {
    type Storage = [u8; size_of::<&'_ ()>()];
}

impl<T> Erasable for &'_ [T] {
    type Storage = [u8; size_of::<&'_ [()]>()];
}

impl<T> Erasable for Option<&'_ T> {
    type Storage = [u8; size_of::<Option<&'_ ()>>()];
}

macro_rules! impl_erasable_for_types_with_no_type_params {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl Erasable for $ty {
                type Storage = [u8; size_of::<$ty>()];
            }
        )*
    }
}

// For types with no type parameters the erased storage for `Foo` is
// `[u8; size_of::<Foo>()]`. ('_ lifetimes are allowed.)
impl_erasable_for_types_with_no_type_params! {
    // tidy-alphabetical-start
    // Note: these `List` impls do not overlap with the impl for `&'_ T` above because `List` is
    // unsized and does not satisfy the implicit `T: Sized` bound.
    //
    // Furthermore, even if that implicit bound was removed (by adding `T: ?Sized`) these impls
    // still wouldn't overlap because `?Sized` is equivalent to `MetaSized` and `RawList` does not
    // satisfy `MetaSized` because it contains an extern type.
    &'_ ty::List<DefId>,
    &'_ ty::List<LocalDefId>,
    &'_ ty::List<Ty<'_>>,
    &'_ ty::ListWithCachedTypeInfo<ty::Clause<'_>>,
    (&'_ Steal<(ty::ResolverAstLowering<'_>, Arc<ast::Crate>)>, &'_ ty::ResolverGlobalCtxt),
    (&'_ Steal<mir::Body<'_>>, &'_ Steal<IndexVec<mir::Promoted, mir::Body<'_>>>),
    (&'_ ty::CrateInherentImpls, Result<(), ErrorGuaranteed>),
    (),
    (QueryResult<'_>, &'_ Probe<TyCtxt<'_>>),
    CrateDepKind,
    DefId,
    DefKind,
    ExpnId,
    Limits,
    MethodAutoderefStepsResult<'_>,
    ObjectLifetimeDefault,
    OptLevel,
    Option<&'_ OsStr>,
    Option<&'_ [hir::PreciseCapturingArgKind<Symbol, Symbol>]>,
    Option<(DefId, EntryFnType)>,
    Option<(mir::ConstValue, Ty<'_>)>,
    Option<Align>,
    Option<AllocatorKind>,
    Option<CrateNum>,
    Option<DefId>,
    Option<DeprecationEntry>,
    Option<LocalDefId>,
    Option<PanicStrategy>,
    Option<Span>,
    Option<Svh>,
    Option<hir::ConstStability>,
    Option<hir::CoroutineKind>,
    Option<hir::DefaultBodyStability>,
    Option<hir::Stability>,
    Option<ty::AsyncDestructor>,
    Option<ty::Destructor>,
    Option<ty::EarlyBinder<'_, Ty<'_>>>,
    Option<ty::IntrinsicDef>,
    Option<ty::ScalarInt>,
    Option<ty::Value<'_>>,
    Option<usize>,
    PanicStrategy,
    Result<&'_ Canonical<'_, QueryResponse<'_, ()>>, NoSolution>,
    Result<&'_ Canonical<'_, QueryResponse<'_, DropckOutlivesResult<'_>>>, NoSolution>,
    Result<&'_ Canonical<'_, QueryResponse<'_, NormalizationResult<'_>>>, NoSolution>,
    Result<&'_ Canonical<'_, QueryResponse<'_, Ty<'_>>>, NoSolution>,
    Result<&'_ Canonical<'_, QueryResponse<'_, Vec<OutlivesBound<'_>>>>, NoSolution>,
    Result<&'_ Canonical<'_, QueryResponse<'_, ty::Clause<'_>>>, NoSolution>,
    Result<&'_ Canonical<'_, QueryResponse<'_, ty::FnSig<'_>>>, NoSolution>,
    Result<&'_ Canonical<'_, QueryResponse<'_, ty::PolyFnSig<'_>>>, NoSolution>,
    Result<&'_ DefIdMap<ty::EarlyBinder<'_, Ty<'_>>>, ErrorGuaranteed>,
    Result<&'_ FnAbi<'_, Ty<'_>>, &'_ ty::layout::FnAbiError<'_>>,
    Result<&'_ FxIndexMap<LocalDefId, ty::DefinitionSiteHiddenType<'_>>, ErrorGuaranteed>,
    Result<&'_ ImplSource<'_, ()>, CodegenObligationError>,
    Result<&'_ TokenStream, ()>,
    Result<&'_ specialization_graph::Graph, ErrorGuaranteed>,
    Result<&'_ ty::List<Ty<'_>>, ty::util::AlwaysRequiresDrop>,
    Result<(&'_ Steal<thir::Thir<'_>>, thir::ExprId), ErrorGuaranteed>,
    Result<(&'_ [Spanned<MonoItem<'_>>], &'_ [Spanned<MonoItem<'_>>]), NormalizationErrorInMono>,
    Result<(), ErrorGuaranteed>,
    Result<EvaluationResult, OverflowError>,
    Result<Option<ty::EarlyBinder<'_, ty::Const<'_>>>, ErrorGuaranteed>,
    Result<Option<ty::Instance<'_>>, ErrorGuaranteed>,
    Result<bool, &ty::layout::LayoutError<'_>>,
    Result<mir::ConstAlloc<'_>, mir::interpret::ErrorHandled>,
    Result<mir::ConstValue, mir::interpret::ErrorHandled>,
    Result<ty::GenericArg<'_>, NoSolution>,
    Result<ty::adjustment::CoerceUnsizedInfo, ErrorGuaranteed>,
    Result<ty::layout::TyAndLayout<'_>, &ty::layout::LayoutError<'_>>,
    SanitizerFnAttrs,
    Span,
    Svh,
    Symbol,
    SymbolManglingVersion,
    Ty<'_>,
    bool,
    hir::Constness,
    hir::Defaultness,
    hir::HirId,
    hir::MaybeOwner<'_>,
    hir::OpaqueTyOrigin<DefId>,
    mir::ConstQualifs,
    mir::ConstValue,
    mir::interpret::AllocId,
    mir::interpret::EvalStaticInitializerRawResult<'_>,
    mir::interpret::EvalToValTreeResult<'_>,
    mir::mono::MonoItemPartitions<'_>,
    ty::AdtDef<'_>,
    ty::AnonConstKind,
    ty::AssocItem,
    ty::Asyncness,
    ty::Binder<'_, ty::CoroutineWitnessTypes<TyCtxt<'_>>>,
    ty::Binder<'_, ty::FnSig<'_>>,
    ty::ClosureTypeInfo<'_>,
    ty::Const<'_>,
    ty::ConstConditions<'_>,
    ty::EarlyBinder<'_, &'_ [(ty::Clause<'_>, Span)]>,
    ty::EarlyBinder<'_, &'_ [(ty::PolyTraitRef<'_>, Span)]>,
    ty::EarlyBinder<'_, Ty<'_>>,
    ty::EarlyBinder<'_, ty::Binder<'_, ty::CoroutineWitnessTypes<TyCtxt<'_>>>>,
    ty::EarlyBinder<'_, ty::Clauses<'_>>,
    ty::EarlyBinder<'_, ty::Const<'_>>,
    ty::EarlyBinder<'_, ty::PolyFnSig<'_>>,
    ty::GenericPredicates<'_>,
    ty::ImplTraitHeader<'_>,
    ty::ParamEnv<'_>,
    ty::SymbolName<'_>,
    ty::TypingEnv<'_>,
    ty::Visibility<DefId>,
    ty::inhabitedness::InhabitedPredicate<'_>,
    usize,
    // tidy-alphabetical-end
}
