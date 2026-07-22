//! Monomorphization-time enforcement of wasm `externref` exclusion rules:
//! `externref` values only exist as bare wasm locals, function arguments and
//! return values. They can never be placed in linear memory, so we reject any
//! local whose layout would aggregate one, any borrow of one, and any load or
//! store of one through a pointer.

use rustc_abi::{AddressSpace, BackendRepr, FieldsShape, Layout, Primitive, Scalar, Variants};
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::mir::visit::Visitor as MirVisitor;
use rustc_middle::mir::{self, Location, traversal};
use rustc_middle::ty::layout::{LayoutCx, TyAndLayout};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt, TypeFoldable};
use rustc_span::Span;

pub(crate) fn check_externref<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &'tcx mir::Body<'tcx>,
) {
    if tcx.lang_items().externref().is_none() {
        return;
    }

    let mut visitor =
        ExternRefVisitor { tcx, instance, body, contains_cache: FxHashMap::default() };

    for decl in body.local_decls.iter() {
        let ty = visitor.monomorphize(decl.ty);
        if let Some(layout) = visitor.layout_of(ty)
            && !matches!(layout.backend_repr, BackendRepr::Scalar(_))
            && visitor.contains_externref(layout)
        {
            visitor.emit_storage_error(ty, decl.source_info.span);
        }
    }

    for (bb, data) in traversal::mono_reachable(body, tcx, instance) {
        visitor.visit_basic_block_data(bb, data);
    }
}

struct ExternRefVisitor<'tcx> {
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    body: &'tcx mir::Body<'tcx>,
    contains_cache: FxHashMap<Layout<'tcx>, bool>,
}

fn scalar_is_externref(scalar: Scalar) -> bool {
    matches!(scalar.primitive(), Primitive::Pointer(AddressSpace::WASM_EXTERNREF))
}

impl<'tcx> ExternRefVisitor<'tcx> {
    fn monomorphize<T>(&self, value: T) -> T
    where
        T: TypeFoldable<TyCtxt<'tcx>>,
    {
        self.instance.instantiate_mir_and_normalize_erasing_regions(
            self.tcx,
            ty::TypingEnv::fully_monomorphized(),
            ty::EarlyBinder::bind(self.tcx, value),
        )
    }

    fn layout_of(&self, ty: Ty<'tcx>) -> Option<TyAndLayout<'tcx>> {
        self.tcx.layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(ty)).ok()
    }

    /// Whether any value of this layout includes an `externref`, at any depth,
    /// without following pointer indirection.
    fn contains_externref(&mut self, layout: TyAndLayout<'tcx>) -> bool {
        if let Some(&cached) = self.contains_cache.get(&layout.layout) {
            return cached;
        }
        let result = self.contains_externref_uncached(layout);
        self.contains_cache.insert(layout.layout, result);
        result
    }

    fn contains_externref_uncached(&mut self, layout: TyAndLayout<'tcx>) -> bool {
        match layout.backend_repr {
            BackendRepr::Scalar(s) => scalar_is_externref(s),
            BackendRepr::ScalarPair { a, b, .. } => {
                scalar_is_externref(a) || scalar_is_externref(b)
            }
            BackendRepr::SimdVector { element, .. }
            | BackendRepr::SimdScalableVector { element, .. } => scalar_is_externref(element),
            BackendRepr::Memory { .. } => {
                let cx = LayoutCx::new(self.tcx, ty::TypingEnv::fully_monomorphized());
                let fields_contain =
                    |this: &mut Self, layout: TyAndLayout<'tcx>| match layout.fields {
                        FieldsShape::Primitive => false,
                        FieldsShape::Array { count, .. } => {
                            count > 0 && this.contains_externref(layout.field(&cx, 0))
                        }
                        FieldsShape::Union(_) | FieldsShape::Arbitrary { .. } => {
                            (0..layout.fields.count())
                                .any(|i| this.contains_externref(layout.field(&cx, i)))
                        }
                    };
                if fields_contain(self, layout) {
                    return true;
                }
                if let Variants::Multiple { variants, .. } = &layout.variants {
                    let variant_indices = variants.indices();
                    for vidx in variant_indices {
                        if fields_contain(self, layout.for_variant(&cx, vidx)) {
                            return true;
                        }
                    }
                }
                false
            }
        }
    }

    fn place_contains_externref(&mut self, place: &mir::Place<'tcx>) -> Option<Ty<'tcx>> {
        let ty = self.monomorphize(place.ty(self.body, self.tcx).ty);
        let layout = self.layout_of(ty)?;
        self.contains_externref(layout).then_some(ty)
    }

    fn emit_storage_error(&self, ty: Ty<'tcx>, span: Span) {
        self.tcx
            .dcx()
            .struct_span_err(
                span,
                format!("values of type `{ty}` cannot exist: it contains a wasm `externref`, which cannot be stored in memory"),
            )
            .with_help("wasm `externref` values may only be used as bare function arguments, return values and locals")
            .emit();
    }

    fn span_of(&self, location: Location) -> Span {
        self.body.source_info(location).span
    }
}

impl<'tcx> MirVisitor<'tcx> for ExternRefVisitor<'tcx> {
    fn visit_rvalue(&mut self, rvalue: &mir::Rvalue<'tcx>, location: Location) {
        if let mir::Rvalue::Ref(_, _, place) | mir::Rvalue::RawPtr(_, place) = rvalue
            && let Some(ty) = self.place_contains_externref(place)
        {
            self.tcx
                .dcx()
                .struct_span_err(
                    self.span_of(location),
                    format!("cannot take a reference to `{ty}`: wasm `externref` values have no memory address"),
                )
                .emit();
        }
        self.super_rvalue(rvalue, location);
    }

    fn visit_place(
        &mut self,
        place: &mir::Place<'tcx>,
        context: mir::visit::PlaceContext,
        location: Location,
    ) {
        // References to externref-containing values cannot be created, but
        // unsafe code can still conjure pointers; reject the loads/stores.
        // Borrows of indirect places are reported by `visit_rvalue` instead.
        if !context.is_borrow()
            && !context.is_address_of()
            && place.is_indirect()
            && let Some(ty) = self.place_contains_externref(place)
        {
            self.tcx
                .dcx()
                .struct_span_err(
                    self.span_of(location),
                    format!("cannot load or store `{ty}` through a pointer: wasm `externref` values cannot be placed in memory"),
                )
                .emit();
        }
        self.super_place(place, context, location);
    }
}
