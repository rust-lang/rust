use rustc_abi::FieldIdx;
use rustc_hir as hir;
use rustc_hir::def_id::DefId;
use rustc_hir::lang_items::LangItem;
use rustc_macros::{HashStable, TyDecodable, TyEncodable, TypeFoldable, TypeVisitable};
use rustc_span::Span;

use crate::ty::{Ty, TyCtxt};

#[derive(Clone, Copy, Debug, PartialEq, Eq, TyEncodable, TyDecodable, Hash, HashStable)]
pub enum PointerCoercion {
    /// Go from a fn-item type to a fn-pointer type.
    ReifyFnPointer,

    /// Go from a safe fn pointer to an unsafe fn pointer.
    UnsafeFnPointer,

    /// Go from a non-capturing closure to an fn pointer or an unsafe fn pointer.
    /// It cannot convert a closure that requires unsafe.
    ClosureFnPointer(hir::Safety),

    /// Go from a mut raw pointer to a const raw pointer.
    MutToConstPointer,

    /// Go from `*const [T; N]` to `*const T`
    ArrayToPointer,

    /// Unsize a pointer/reference value, e.g., `&[T; n]` to
    /// `&[T]`. Note that the source could be a thin or wide pointer.
    /// This will do things like convert thin pointers to wide
    /// pointers, or convert structs containing thin pointers to
    /// structs containing wide pointers, or convert between wide
    /// pointers. We don't store the details of how the transform is
    /// done (in fact, we don't know that, because it might depend on
    /// the precise type parameters). We just store the target
    /// type. Codegen backends and miri figure out what has to be done
    /// based on the precise source/target type at hand.
    Unsize,
}

/// Represents coercing a value to a different type of value.
///
/// We transform values by following a number of `Adjust` steps in order.
/// See the documentation on variants of `Adjust` for more details.
///
/// Here are some common scenarios:
///
/// 1. The simplest cases are where a pointer is not adjusted fat vs thin.
///    Here the pointer will be dereferenced N times (where a dereference can
///    happen to raw or borrowed pointers or any smart pointer which implements
///    `Deref`, including `Box<_>`). The types of dereferences is given by
///    `autoderefs`. It can then be auto-referenced zero or one times, indicated
///    by `autoref`, to either a raw or borrowed pointer. In these cases unsize is
///    `false`.
///
/// 2. A thin-to-fat coercion involves unsizing the underlying data. We start
///    with a thin pointer, deref a number of times, unsize the underlying data,
///    then autoref. The 'unsize' phase may change a fixed length array to a
///    dynamically sized one, a concrete object to a trait object, or statically
///    sized struct to a dynamically sized one. E.g., `&[i32; 4]` -> `&[i32]` is
///    represented by:
///
///    ```ignore (illustrative)
///    Deref(None) -> [i32; 4],
///    Borrow(AutoBorrow::Ref) -> &[i32; 4],
///    Unsize -> &[i32],
///    ```
///
///    Note that for a struct, the 'deep' unsizing of the struct is not recorded.
///    E.g., `struct Foo<T> { x: T }` we can coerce `&Foo<[i32; 4]>` to `&Foo<[i32]>`
///    The autoderef and -ref are the same as in the above example, but the type
///    stored in `unsize` is `Foo<[i32]>`, we don't store any further detail about
///    the underlying conversions from `[i32; 4]` to `[i32]`.
///
/// 3. Coercing a `Box<T>` to `Box<dyn Trait>` is an interesting special case. In
///    that case, we have the pointer we need coming in, so there are no
///    autoderefs, and no autoref. Instead we just do the `Unsize` transformation.
///    At some point, of course, `Box` should move out of the compiler, in which
///    case this is analogous to transforming a struct. E.g., `Box<[i32; 4]>` ->
///    `Box<[i32]>` is an `Adjust::Unsize` with the target `Box<[i32]>`.
#[derive(Clone, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct Adjustment<'tcx> {
    pub kind: Adjust,
    pub target: Ty<'tcx>,
}

impl<'tcx> Adjustment<'tcx> {
    pub fn is_region_borrow(&self) -> bool {
        matches!(self.kind, Adjust::Borrow(AutoBorrow::Ref(..)))
    }
}

#[derive(Clone, Debug, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub enum Adjust {
    /// Go from ! to any type.
    NeverToAny,

    /// Dereference once, producing a place.
    Deref(Option<OverloadedDeref>),

    /// Take the address and produce either a `&` or `*` pointer.
    Borrow(AutoBorrow),

    Pointer(PointerCoercion),

    /// Take a pinned reference and reborrow as a `Pin<&mut T>` or `Pin<&T>`.
    ReborrowPin(hir::Mutability),
}

/// An overloaded autoderef step, representing a `Deref(Mut)::deref(_mut)`
/// call, with the signature `&'a T -> &'a U` or `&'a mut T -> &'a mut U`.
/// The target type is `U` in both cases, with the region and mutability
/// being those shared by both the receiver and the returned reference.
#[derive(Copy, Clone, PartialEq, Debug, TyEncodable, TyDecodable, HashStable)]
#[derive(TypeFoldable, TypeVisitable)]
pub struct OverloadedDeref {
    pub mutbl: hir::Mutability,
    /// The `Span` associated with the field access or method call
    /// that triggered this overloaded deref.
    pub span: Span,
}

impl OverloadedDeref {
    /// Get the [`DefId`] of the method call for the given `Deref`/`DerefMut` trait
    /// for this overloaded deref's mutability.
    pub fn method_call<'tcx>(&self, tcx: TyCtxt<'tcx>) -> DefId {
        let trait_def_id = match self.mutbl {
            hir::Mutability::Not => tcx.require_lang_item(LangItem::Deref, self.span),
            hir::Mutability::Mut => tcx.require_lang_item(LangItem::DerefMut, self.span),
        };
        tcx.associated_items(trait_def_id)
            .in_definition_order()
            .find(|item| item.is_fn())
            .unwrap()
            .def_id
    }
}

/// At least for initial deployment, we want to limit two-phase borrows to
/// only a few specific cases. Right now, those are mostly "things that desugar"
/// into method calls:
/// - using `x.some_method()` syntax, where some_method takes `&mut self`,
/// - using `Foo::some_method(&mut x, ...)` syntax,
/// - binary assignment operators (`+=`, `-=`, `*=`, etc.).
/// Anything else should be rejected until generalized two-phase borrow support
/// is implemented. Right now, dataflow can't handle the general case where there
/// is more than one use of a mutable borrow, and we don't want to accept too much
/// new code via two-phase borrows, so we try to limit where we create two-phase
/// capable mutable borrows.
/// See #49434 for tracking.
#[derive(Copy, Clone, PartialEq, Debug, TyEncodable, TyDecodable, HashStable)]
pub enum AllowTwoPhase {
    Yes,
    No,
}

#[derive(Copy, Clone, PartialEq, Debug, TyEncodable, TyDecodable, HashStable)]
pub enum AutoBorrowMutability {
    Mut { allow_two_phase_borrow: AllowTwoPhase },
    Not,
}

impl AutoBorrowMutability {
    /// Creates an `AutoBorrowMutability` from a mutability and allowance of two phase borrows.
    ///
    /// Note that when `mutbl.is_not()`, `allow_two_phase_borrow` is ignored
    pub fn new(mutbl: hir::Mutability, allow_two_phase_borrow: AllowTwoPhase) -> Self {
        match mutbl {
            hir::Mutability::Not => Self::Not,
            hir::Mutability::Mut => Self::Mut { allow_two_phase_borrow },
        }
    }
}

impl From<AutoBorrowMutability> for hir::Mutability {
    fn from(m: AutoBorrowMutability) -> Self {
        match m {
            AutoBorrowMutability::Mut { .. } => hir::Mutability::Mut,
            AutoBorrowMutability::Not => hir::Mutability::Not,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug, TyEncodable, TyDecodable, HashStable)]
#[derive(TypeFoldable, TypeVisitable)]
pub enum AutoBorrow {
    /// Converts from T to &T.
    Ref(AutoBorrowMutability),

    /// Converts from T to *T.
    RawPtr(hir::Mutability),
}

/// Information for `CoerceUnsized` impls, storing information we
/// have computed about the coercion.
///
/// This struct can be obtained via the `coerce_impl_info` query.
/// Demanding this struct also has the side-effect of reporting errors
/// for inappropriate impls.
#[derive(Clone, Copy, TyEncodable, TyDecodable, Debug, HashStable)]
pub struct CoerceUnsizedInfo {
    /// If this is a "custom coerce" impl, then what kind of custom
    /// coercion is it? This applies to impls of `CoerceUnsized` for
    /// structs, primarily, where we store a bit of info about which
    /// fields need to be coerced.
    pub custom_kind: Option<CustomCoerceUnsized>,
}

#[derive(Clone, Copy, TyEncodable, TyDecodable, Debug, HashStable)]
pub enum CustomCoerceUnsized {
    /// Records the index of the field being coerced.
    Struct(FieldIdx),
}

/// Represents an implicit coercion applied to the scrutinee of a match before testing a pattern
/// against it. Currently, this is used only for implicit dereferences.
#[derive(Clone, Copy, TyEncodable, TyDecodable, HashStable, TypeFoldable, TypeVisitable)]
pub struct PatAdjustment<'tcx> {
    pub kind: PatAdjust,
    /// The type of the scrutinee before the adjustment is applied, or the "adjusted type" of the
    /// pattern.
    pub source: Ty<'tcx>,
}

/// Represents implicit coercions of patterns' types, rather than values' types.
#[derive(Clone, Copy, PartialEq, Debug, TyEncodable, TyDecodable, HashStable)]
#[derive(TypeFoldable, TypeVisitable)]
pub enum PatAdjust {
    /// An implicit dereference before matching, such as when matching the pattern `0` against a
    /// scrutinee of type `&u8` or `&mut u8`.
    BuiltinDeref,
    /// An implicit call to `Deref(Mut)::deref(_mut)` before matching, such as when matching the
    /// pattern `[..]` against a scrutinee of type `Vec<T>`.
    OverloadedDeref,
}
