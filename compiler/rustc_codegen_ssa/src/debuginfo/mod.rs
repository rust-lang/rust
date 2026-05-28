use rustc_abi::{Integer, Primitive, Size, TagEncoding, Variants};
use rustc_middle::bug;
use rustc_middle::ty::layout::{IntegerExt, PrimitiveExt, TyAndLayout};
use rustc_middle::ty::{self, Ty, TyCtxt};

// FIXME(eddyb) find a place for this (or a way to replace it).
pub mod type_names;

/// Returns true if we want to generate a DW_TAG_enumeration_type description for
/// this instead of a DW_TAG_struct_type with DW_TAG_variant_part.
///
/// NOTE: This is somewhat inconsistent right now: For empty enums and enums with a single
///       fieldless variant, we generate DW_TAG_struct_type, although a
///       DW_TAG_enumeration_type would be a better fit.
pub fn wants_c_like_enum_debuginfo<'tcx>(
    tcx: TyCtxt<'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
) -> bool {
    match enum_type_and_layout.ty.kind() {
        ty::Adt(adt_def, _) => {
            if !adt_def.is_enum() {
                return false;
            }

            if type_names::cpp_like_debuginfo(tcx)
                && tag_base_type_opt(tcx, enum_type_and_layout)
                    .map(|ty| ty.primitive_size(tcx).bits())
                    == Some(128)
            {
                // C++-like debuginfo never uses the C-like representation for 128-bit enums.
                return false;
            }

            match adt_def.variants().len() {
                0 => false,
                1 => {
                    // Univariant enums unless they are zero-sized
                    enum_type_and_layout.size != Size::ZERO && adt_def.all_fields().count() == 0
                }
                _ => {
                    // Enums with more than one variant if they have no fields
                    adt_def.all_fields().count() == 0
                }
            }
        }
        _ => false,
    }
}

/// Extract the type with which we want to describe the tag of the given enum or coroutine.
pub fn tag_base_type<'tcx>(tcx: TyCtxt<'tcx>, enum_type_and_layout: TyAndLayout<'tcx>) -> Ty<'tcx> {
    tag_base_type_opt(tcx, enum_type_and_layout).unwrap_or_else(|| {
        bug!("tag_base_type() called for enum without tag: {:?}", enum_type_and_layout)
    })
}

fn tag_base_type_opt<'tcx>(
    tcx: TyCtxt<'tcx>,
    enum_type_and_layout: TyAndLayout<'tcx>,
) -> Option<Ty<'tcx>> {
    assert!(match enum_type_and_layout.ty.kind() {
        ty::Coroutine(..) => true,
        ty::Adt(adt_def, _) => adt_def.is_enum(),
        _ => false,
    });

    match enum_type_and_layout.layout.variants() {
        // A single-variant or no-variant enum has no discriminant.
        Variants::Single { .. } | Variants::Empty => None,

        Variants::Multiple { tag_encoding: TagEncoding::Niche { .. }, tag, .. } => {
            // Niche tags are always normalized to unsized integers of the correct size.
            Some(
                match tag.primitive() {
                    Primitive::Int(t, _) => t,
                    Primitive::Float(f) => Integer::from_size(f.size()).unwrap(),
                    // FIXME(erikdesjardins): handle non-default addrspace ptr sizes
                    Primitive::Pointer(_) => {
                        // If the niche is the NULL value of a reference, then `discr_enum_ty` will
                        // be a RawPtr. CodeView doesn't know what to do with enums whose base type
                        // is a pointer so we fix this up to just be `usize`.
                        // DWARF might be able to deal with this but with an integer type we are on
                        // the safe side there too.
                        tcx.data_layout.ptr_sized_integer()
                    }
                }
                .to_ty(tcx, false),
            )
        }

        Variants::Multiple { tag_encoding: TagEncoding::Direct, tag, .. } => {
            // Direct tags preserve the sign.
            Some(tag.primitive().to_ty(tcx))
        }
    }
}
