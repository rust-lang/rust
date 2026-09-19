use serde::Serialize;

use crate::abi::ReprOptions;
use crate::crate_def::{CrateDef, CrateDefType};
use crate::mir::Body;
use crate::ty::tys::*;
use crate::{AssocItems, DefId, IndexedVal, Symbol, with};

crate_def! {
    #[derive(Serialize)]
    pub ForeignModuleDef;

    #[derive(Serialize)]
    pub ClosureDef;

    #[derive(Serialize)]
    pub CoroutineDef;

    #[derive(Serialize)]
    pub CoroutineClosureDef;

    #[derive(Serialize)]
    pub ParamDef;

    #[derive(Serialize)]
    pub BrNamedDef;

    #[derive(Serialize)]
    pub AdtDef;

    #[derive(Serialize)]
    pub AliasDef;

    /// A trait's definition.
    #[derive(Serialize)]
    pub TraitDef;

    #[derive(Serialize)]
    pub GenericDef;

    #[derive(Serialize)]
    pub RegionDef;

    #[derive(Serialize)]
    pub CoroutineWitnessDef;

    /// Hold information about an Opaque definition, particularly useful in `RPITIT`.
    #[derive(Serialize)]
    pub OpaqueDef;

    #[derive(Serialize)]
    pub AssocDef;
}

crate_def_with_ty! {
    /// Hold information about a ForeignItem in a crate.
    #[derive(Serialize)]
    pub ForeignDef;

    /// Hold information about a function definition in a crate.
    #[derive(Serialize)]
    pub FnDef;

    #[derive(Serialize)]
    pub IntrinsicDef;

    #[derive(Serialize)]
    pub FieldDef {
        /// The field name.
        pub name: Symbol,
    }

    #[derive(Serialize)]
    pub ConstDef;

    /// A trait impl definition.
    #[derive(Serialize)]
    pub ImplDef;
}

impl ForeignModuleDef {
    pub fn module(&self) -> ForeignModule {
        with(|cx| cx.foreign_module(*self))
    }
}

impl ForeignDef {
    pub fn kind(&self) -> ForeignItemKind {
        with(|cx| cx.foreign_item_kind(*self))
    }
}

impl FnDef {
    // Get the function body if available.
    pub fn body(&self) -> Option<Body> {
        with(|ctx| ctx.has_body(self.0).then(|| ctx.mir_body(self.0)))
    }

    // Check if the function body is available.
    pub fn has_body(&self) -> bool {
        with(|ctx| ctx.has_body(self.0))
    }

    /// Get the information of the intrinsic if this function is a definition of one.
    pub fn as_intrinsic(&self) -> Option<IntrinsicDef> {
        with(|cx| cx.intrinsic(self.def_id()))
    }

    /// Check if the function is an intrinsic.
    #[inline]
    pub fn is_intrinsic(&self) -> bool {
        self.as_intrinsic().is_some()
    }

    /// Get the constness of this function definition.
    pub fn constness(&self) -> Constness {
        with(|cx| cx.constness(*self))
    }

    /// Get the asyncness of this function definition.
    pub fn asyncness(&self) -> Asyncness {
        with(|cx| cx.asyncness(*self))
    }

    /// Get the function signature for this function definition.
    pub fn fn_sig(&self) -> PolyFnSig {
        let kind = self.ty().kind();
        kind.fn_sig().unwrap()
    }

    /// Get the generics of this function definition.
    pub fn generics_of(&self) -> Generics {
        with(|cx| cx.generics_of(self.0))
    }

    /// Get the associated item information if this function is one.
    pub fn associated_item(&self) -> Option<AssocItem> {
        with(|cx| cx.associated_item(self.0))
    }
}

impl IntrinsicDef {
    /// Returns the plain name of the intrinsic.
    /// e.g., `transmute` for `core::intrinsics::transmute`.
    pub fn fn_name(&self) -> Symbol {
        with(|cx| cx.intrinsic_name(*self))
    }

    /// Returns whether the intrinsic has no meaningful body and all backends
    /// need to shim all calls to it.
    pub fn must_be_overridden(&self) -> bool {
        with(|cx| !cx.has_body(self.0))
    }
}

impl From<IntrinsicDef> for FnDef {
    fn from(def: IntrinsicDef) -> Self {
        FnDef(def.0)
    }
}

impl ClosureDef {
    /// Retrieves the body of the closure definition. Returns None if the body
    /// isn't available.
    pub fn body(&self) -> Option<Body> {
        with(|ctx| ctx.has_body(self.0).then(|| ctx.mir_body(self.0)))
    }
}

impl CoroutineDef {
    /// Retrieves the body of the coroutine definition. Returns None if the body
    /// isn't available.
    pub fn body(&self) -> Option<Body> {
        with(|cx| cx.has_body(self.0).then(|| cx.mir_body(self.0)))
    }

    pub fn discriminant_for_variant(&self, args: &GenericArgs, idx: VariantIdx) -> Discr {
        with(|cx| cx.coroutine_discr_for_variant(*self, args, idx))
    }
}

impl AdtDef {
    pub fn kind(&self) -> AdtKind {
        with(|cx| cx.adt_kind(*self))
    }

    /// Retrieve the type of this Adt.
    pub fn ty(&self) -> Ty {
        with(|cx| cx.def_ty(self.0))
    }

    /// Retrieve the type of this Adt by instantiating and normalizing it with the given arguments.
    ///
    /// This will assume the type can be instantiated with these arguments.
    pub fn ty_with_args(&self, args: &GenericArgs) -> Ty {
        with(|cx| cx.def_ty_with_args(self.0, args))
    }

    pub fn is_box(&self) -> bool {
        with(|cx| cx.adt_is_box(*self))
    }

    pub fn is_simd(&self) -> bool {
        with(|cx| cx.adt_is_simd(*self))
    }

    /// The number of variants in this ADT.
    pub fn num_variants(&self) -> usize {
        with(|cx| cx.adt_variants_len(*self))
    }

    /// Retrieve the variants in this ADT.
    pub fn variants(&self) -> Vec<VariantDef> {
        self.variants_iter().collect()
    }

    /// Iterate over the variants in this ADT.
    pub fn variants_iter(&self) -> impl Iterator<Item = VariantDef> {
        (0..self.num_variants())
            .map(|idx| VariantDef { idx: VariantIdx::to_val(idx), adt_def: *self })
    }

    pub fn variant(&self, idx: VariantIdx) -> Option<VariantDef> {
        (idx.to_index() < self.num_variants()).then_some(VariantDef { idx, adt_def: *self })
    }

    pub fn repr(&self) -> ReprOptions {
        with(|cx| cx.adt_repr(*self))
    }

    pub fn discriminant_for_variant(&self, idx: VariantIdx) -> Discr {
        with(|cx| cx.adt_discr_for_variant(*self, idx))
    }

    /// Get the generics of this ADT definition.
    pub fn generics_of(&self) -> Generics {
        with(|cx| cx.generics_of(self.0))
    }

    /// Retrieve the inherent implementations for this ADT.
    pub fn inherent_impls(&self) -> Vec<ImplDef> {
        with(|cx| cx.inherent_impls(*self))
    }
}

/// Definition of a variant, which can be either a struct / union field or an enum variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct VariantDef {
    /// The variant index.
    pub(crate) idx: VariantIdx,
    /// The data type where this variant comes from.
    /// For now, we use this to retrieve information about the variant itself so we don't need to
    /// cache more information.
    pub(crate) adt_def: AdtDef,
}

impl VariantDef {
    /// The name of the variant, struct or union.
    ///
    /// This will not include the name of the enum or qualified path.
    pub fn name(&self) -> Symbol {
        with(|cx| cx.variant_name(*self))
    }

    /// Retrieve all the fields in this variant.
    // We expect user to cache this and use it directly since today it is expensive to generate all
    // fields name.
    pub fn fields(&self) -> Vec<FieldDef> {
        with(|cx| cx.variant_fields(*self))
    }

    /// Returns the variant index.
    pub fn idx(&self) -> VariantIdx {
        self.idx
    }

    /// Returns the `AdtDef` which this variant comes from.
    pub fn adt_def(&self) -> AdtDef {
        self.adt_def
    }
}

impl TraitDef {
    pub fn declaration(trait_def: &TraitDef) -> TraitDecl {
        with(|cx| cx.trait_decl(trait_def))
    }

    pub fn associated_items(&self) -> AssocItems {
        with(|cx| cx.associated_items(self.def_id()))
    }
}

impl ImplDef {
    /// Retrieve information about this implementation.
    pub fn trait_impl(&self) -> ImplTrait {
        with(|cx| cx.trait_impl(self))
    }

    pub fn associated_items(&self) -> AssocItems {
        with(|cx| cx.associated_items(self.def_id()))
    }

    /// Get the generics of this implementation.
    pub fn generics_of(&self) -> Generics {
        with(|cx| cx.generics_of(self.0))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct GenericParamDef {
    pub name: Symbol,
    pub def_id: GenericDef,
    pub index: u32,
    pub pure_wrt_drop: bool,
    pub kind: GenericParamDefKind,
}
