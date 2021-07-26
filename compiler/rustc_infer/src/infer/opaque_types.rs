use rustc_data_structures::vec_map::VecMap;
use rustc_hir as hir;
use rustc_middle::ty::{OpaqueTypeKey, Ty};
use rustc_span::Span;

pub type OpaqueTypeMap<'tcx> = VecMap<OpaqueTypeKey<'tcx>, OpaqueTypeDecl<'tcx>>;

/// Information about the opaque types whose values we
/// are inferring in this function (these are the `impl Trait` that
/// appear in the return type).
#[derive(Copy, Clone, Debug)]
pub struct OpaqueTypeDecl<'tcx> {
    /// The opaque type (`ty::Opaque`) for this declaration.
    pub opaque_type: Ty<'tcx>,

    /// The span of this particular definition of the opaque type. So
    /// for example:
    ///
    /// ```ignore (incomplete snippet)
    /// type Foo = impl Baz;
    /// fn bar() -> Foo {
    /// //          ^^^ This is the span we are looking for!
    /// }
    /// ```
    ///
    /// In cases where the fn returns `(impl Trait, impl Trait)` or
    /// other such combinations, the result is currently
    /// over-approximated, but better than nothing.
    pub definition_span: Span,

    /// The type variable that represents the value of the opaque type
    /// that we require. In other words, after we compile this function,
    /// we will be created a constraint like:
    ///
    ///     Foo<'a, T> = ?C
    ///
    /// where `?C` is the value of this type variable. =) It may
    /// naturally refer to the type and lifetime parameters in scope
    /// in this function, though ultimately it should only reference
    /// those that are arguments to `Foo` in the constraint above. (In
    /// other words, `?C` should not include `'b`, even though it's a
    /// lifetime parameter on `foo`.)
    pub concrete_ty: Ty<'tcx>,

    /// Returns `true` if the `impl Trait` bounds include region bounds.
    /// For example, this would be true for:
    ///
    ///     fn foo<'a, 'b, 'c>() -> impl Trait<'c> + 'a + 'b
    ///
    /// but false for:
    ///
    ///     fn foo<'c>() -> impl Trait<'c>
    ///
    /// unless `Trait` was declared like:
    ///
    ///     trait Trait<'c>: 'c
    ///
    /// in which case it would be true.
    ///
    /// This is used during regionck to decide whether we need to
    /// impose any additional constraints to ensure that region
    /// variables in `concrete_ty` wind up being constrained to
    /// something from `substs` (or, at minimum, things that outlive
    /// the fn body). (Ultimately, writeback is responsible for this
    /// check.)
    pub has_required_region_bounds: bool,

    /// The origin of the opaque type.
    pub origin: hir::OpaqueTyOrigin,
}
