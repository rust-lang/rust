use rustc_middle::ty::{self, layout::TyAndLayout};
use rustc_target::abi::Size;

// FIXME(eddyb) find a place for this (or a way to replace it).
pub mod type_names;

/// Returns true if we want to generate a DW_TAG_enumeration_type description for
/// this instead of a DW_TAG_struct_type with DW_TAG_variant_part.
///
/// NOTE: This is somewhat inconsistent right now: For empty enums and enums with a single
///       fieldless variant, we generate DW_TAG_struct_type, although a
///       DW_TAG_enumeration_type would be a better fit.
pub fn wants_c_like_enum_debuginfo(enum_type_and_layout: TyAndLayout<'_>) -> bool {
    match enum_type_and_layout.ty.kind() {
        ty::Adt(adt_def, _) => {
            if !adt_def.is_enum() {
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
