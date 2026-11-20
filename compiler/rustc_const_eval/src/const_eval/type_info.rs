mod adt;
mod intrinsics;

use rustc_abi::{ExternAbi, FieldIdx};
use rustc_attr_ir::Attribute::Tool;
use rustc_attr_ir::lang_items::LangItem;
use rustc_attr_ir::{AttrItem, Attribute, AttributeKind, HasAttrs};
use rustc_hir::intravisit::HirTyCtxt;
use rustc_lint_defs::{Reflectiveness, ToolOptions};
use rustc_middle::ty::layout::HasTyCtxt;
use rustc_middle::ty::{self, AdtKind, FnHeader, FnSigKind, FnSigTys, Ty, TyCtxt};
use rustc_span::def_id::DefId;
use rustc_span::{Symbol, span_bug, sym};

use crate::const_eval::CompileTimeMachine;
use crate::interpret::{
    CtfeProvenance, Immediate, InterpCx, InterpResult, MPlaceTy, MemoryKind, Scalar, Writeable,
    interp_ok,
};

impl<'tcx> InterpCx<'tcx, CompileTimeMachine<'tcx>> {
    // A general method to write an array to a static slice place.
    fn allocate_fill_and_write_slice_ptr(
        &mut self,
        slice_place: &impl Writeable<'tcx, CtfeProvenance>,
        len: u64,
        writer: impl Fn(&mut Self, /* index */ u64, MPlaceTy<'tcx>) -> InterpResult<'tcx>,
    ) -> InterpResult<'tcx> {
        // Array element type
        let field_ty = slice_place
            .layout()
            .ty
            .builtin_deref(false)
            .unwrap()
            .sequence_element_type(self.tcx.tcx);

        // Allocate an array
        let array_layout = self.layout_of(Ty::new_array(self.tcx.tcx, field_ty, len))?;
        let array_place = self.allocate(array_layout, MemoryKind::Stack)?;

        // Fill the array fields
        let mut field_places = self.project_array_fields(&array_place)?;
        while let Some((i, place)) = field_places.next(self)? {
            writer(self, i, place)?;
        }

        // Write the slice pointing to the array
        let array_place = array_place.map_provenance(CtfeProvenance::as_immutable);
        let ptr = Immediate::new_slice(array_place.ptr(), len, self);
        self.write_immediate(ptr, slice_place)
    }

    /// Writes a `core::mem::type_info::TypeKind` for a given type, `ty` to the given place.
    pub(crate) fn write_type_kind(
        &mut self,
        ty: Ty<'tcx>,
        dest: &impl Writeable<'tcx, CtfeProvenance>,
    ) -> InterpResult<'tcx> {
        let ty_enum = self.tcx.require_lang_item(LangItem::TypeKind, self.tcx.span);
        let ty_enum = self.tcx.type_of(ty_enum).no_bound_vars().unwrap();
        assert_eq!(ty_enum, dest.layout().ty);

        let (variant_index, _) = self.project_downcast_named(
            dest,
            match ty.kind() {
                ty::Adt(def, _) => match def.adt_kind() {
                    AdtKind::Struct => sym::Struct,
                    AdtKind::Union => sym::Union,
                    AdtKind::Enum => sym::Enum,
                },
                ty::Tuple(_) => sym::Tuple,
                ty::Array(_, _) => sym::Array,
                ty::Slice(_) => sym::Slice,
                ty::Bool => sym::Bool,
                ty::Char => sym::Char,
                ty::Int(_) => sym::Int,
                ty::Uint(_) => sym::Int,
                ty::Float(_) => sym::Float,
                ty::Str => sym::Str,
                ty::Ref(_, _, _) => sym::Reference,
                ty::RawPtr(_, _) => sym::Pointer,
                ty::Dynamic(_, _) => sym::DynTrait,
                ty::FnPtr(_, _) => sym::FnPtr,
                ty::Never => sym::Never,
                ty::Foreign(_)
                | ty::Pat(_, _)
                | ty::FnDef(..)
                | ty::UnsafeBinder(..)
                | ty::Closure(..)
                | ty::CoroutineClosure(..)
                | ty::Coroutine(..)
                | ty::CoroutineWitness(..)
                | ty::Alias(..)
                | ty::Param(_)
                | ty::Bound(..)
                | ty::Placeholder(_)
                | ty::Infer(..)
                | ty::Error(_) => sym::Other,
                ty::View(..) => unimplemented!(), // FIXME(scrabsha)
            },
        )?;
        self.write_discriminant(variant_index, dest)?;

        interp_ok(())
    }

    pub(crate) fn write_type_id_generics(
        &mut self,
        place: &impl Writeable<'tcx, CtfeProvenance>,
        ty: Ty<'tcx>,
    ) -> InterpResult<'tcx> {
        let generics: ty::Binder<'_, ty::GenericArgs<'_>> = match *ty.kind() {
            ty::Bool
            | ty::Char
            | ty::Int(..)
            | ty::Uint(..)
            | ty::Float(..)
            | ty::Foreign(..)
            | ty::Str
            | ty::Array(..)
            | ty::Pat(..)
            | ty::RawPtr(..)
            | ty::Ref(..)
            | ty::FnPtr(..)
            | ty::Dynamic(..)
            | ty::CoroutineWitness(..)
            | ty::Never
            | ty::Tuple(..)
            | ty::Alias(..)
            | ty::Param(..)
            | ty::Bound(..)
            | ty::Placeholder(..)
            | ty::Infer(..)
            | ty::Error(..)
            | ty::Slice(..) => ty::Binder::dummy(ty::GenericArgs::default()),
            ty::Adt(_, args) => ty::Binder::dummy(args),
            ty::FnDef(_, binder) => binder,
            ty::UnsafeBinder(binder) => binder.rebind(ty::GenericArgs::default()),
            ty::Closure(_, args) => ty::Binder::dummy(args),
            ty::CoroutineClosure(_, args) => ty::Binder::dummy(args),
            ty::Coroutine(_, args) => ty::Binder::dummy(args),

            // FIXME(scrabsha)
            ty::View(..) => unimplemented!(),
        };

        // FIXME(type_info): also provide the late bound vars to reflection
        let generics = generics.skip_binder();

        self.write_generics(place, generics)
    }

    pub(crate) fn write_fn_ptr_type_info(
        &mut self,
        place: impl Writeable<'tcx, CtfeProvenance>,
        sig: &FnSigTys<TyCtxt<'tcx>>,
        fn_header: &FnHeader<TyCtxt<'tcx>>,
    ) -> InterpResult<'tcx> {
        let FnHeader { fn_sig_kind } = fn_header;

        for (field_idx, field) in
            place.layout().ty.ty_adt_def().unwrap().non_enum_variant().fields.iter_enumerated()
        {
            let field_place = self.project_field(&place, field_idx)?;

            match field.name {
                sym::is_unsafe => {
                    self.write_scalar(Scalar::from_bool(!fn_sig_kind.is_safe()), &field_place)?;
                }
                sym::abi => match fn_sig_kind.abi() {
                    ExternAbi::C { .. } => {
                        let (rust_variant, _rust_place) =
                            self.project_downcast_named(&field_place, sym::ExternC)?;
                        self.write_discriminant(rust_variant, &field_place)?;
                    }
                    ExternAbi::Rust => {
                        let (rust_variant, _rust_place) =
                            self.project_downcast_named(&field_place, sym::ExternRust)?;
                        self.write_discriminant(rust_variant, &field_place)?;
                    }
                    other_abi => {
                        let (variant, variant_place) =
                            self.project_downcast_named(&field_place, sym::Named)?;
                        let str_place = self.allocate_str_dedup(other_abi.as_str())?;
                        let str_ref = self.mplace_to_imm_ptr(&str_place, None)?;
                        let payload = self.project_field(&variant_place, FieldIdx::ZERO)?;
                        self.write_immediate(*str_ref, &payload)?;
                        self.write_discriminant(variant, &field_place)?;
                    }
                },
                sym::inputs => {
                    let inputs = sig.inputs();
                    self.allocate_fill_and_write_slice_ptr(
                        &field_place,
                        inputs.len() as _,
                        |this, i, place| this.write_type_id(inputs[i as usize], &place),
                    )?;
                }
                sym::output => {
                    let output = sig.output();
                    self.write_type_id(output, &field_place)?;
                }
                sym::variadic => {
                    self.write_scalar(Scalar::from_bool(fn_sig_kind.c_variadic()), &field_place)?;
                }
                sym::is_splatted => {
                    self.write_scalar(
                        Scalar::from_bool(fn_sig_kind.splatted().is_some()),
                        &field_place,
                    )?;
                }
                sym::splatted_index => {
                    self.write_scalar(
                        Scalar::from_u8(
                            // Currently the same encoding as FnSigKind.splatted
                            // FIXME(splat): make these two fields into a single Option<u8/u16>, or choose a stable encoding
                            fn_sig_kind.splatted().unwrap_or(FnSigKind::NO_SPLATTED_ARG_INDEX),
                        ),
                        &field_place,
                    )?;
                }
                other => span_bug!(self.tcx.def_span(field.did), "unimplemented field {other}"),
            }
        }

        interp_ok(())
    }

    pub(crate) fn write_reflect_attributes(
        &mut self,
        place: &impl Writeable<'tcx, CtfeProvenance>,
        attrs: &'tcx [rustc_attr_ir::Attribute],
    ) -> InterpResult<'tcx> {
        let attrs: Vec<_> = attrs
            .iter()
            .filter_map(|attr| {
                if let Tool {
                    item: a_itm @ AttrItem { .. },
                    opts: ToolOptions { reflectiveness: Reflectiveness::Reflective, .. },
                } = attr
                {
                    Some(rustc_hir_pretty::attritem_to_string(
                        &(&self.tcx() as &dyn HirTyCtxt<'_>),
                        &a_itm,
                    ))
                } else {
                    None
                }
            })
            .collect();

        self.allocate_fill_and_write_slice_ptr(place, attrs.len() as _, |this, i, place| {
            let attr = &attrs[i as usize];
            let ptr = this.allocate_bytes_dedup(attr.trim_end().as_bytes())?;

            this.write_immediate(
                Immediate::ScalarPair(
                    Scalar::from_pointer(ptr, this),
                    Scalar::from_target_usize(attr.trim_end().len() as u64, this),
                ),
                &place,
            )
        })
    }

    pub(crate) fn write_doc_attributes(
        &mut self,
        place: &impl Writeable<'tcx, CtfeProvenance>,
        def_id: Option<DefId>,
    ) -> InterpResult<'tcx> {
        // only the doc comments of exported items are encoded in cross-crate metadata
        // see `rustc_metadata::rmeta::encoder::EncodeContext::encode_attrs`
        // this mirrors the check so that we return the same comments regardless of crate
        let attrs = if let Some(d) = def_id
            && d.as_local().map_or(true, |d| {
                !self.tcx.needs_metadata() || self.tcx.effective_visibilities(()).is_exported(d)
            }) {
            d.get_attrs(&self.tcx)
        } else {
            &[]
        };

        let docs = attrs
            .iter()
            .filter_map(|x| match x {
                Attribute::Parsed(AttributeKind::DocComment { comment, .. }) => Some(*comment),
                _ => None,
            })
            .collect::<Vec<Symbol>>();

        self.allocate_fill_and_write_slice_ptr(place, docs.len() as _, |this, i, place| {
            let doc = docs[i as usize];

            let ptr = this.allocate_bytes_dedup(doc.as_str().as_bytes())?;

            this.write_immediate(
                Immediate::ScalarPair(
                    Scalar::from_pointer(ptr, this),
                    Scalar::from_target_usize(doc.as_str().len() as u64, this),
                ),
                &place,
            )
        })
    }
}
