//! Pattern analysis sometimes wants to print patterns as part of a user-visible
//! diagnostic.
//!
//! Historically it did so by creating a synthetic [`thir::Pat`](rustc_middle::thir::Pat)
//! and printing that, but doing so was making it hard to modify the THIR pattern
//! representation for other purposes.
//!
//! So this module contains a forked copy of `thir::Pat` that is used _only_
//! for diagnostics, and has been partly simplified to remove things that aren't
//! needed for printing.

use std::fmt;

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_middle::bug;
use rustc_middle::ty::{self, AdtDef, Ty, TyCtxt};
use rustc_span::sym;

#[derive(Clone, Debug)]
pub(crate) struct FieldPat {
    pub(crate) field: FieldIdx,
    pub(crate) pattern: String,
    pub(crate) is_wildcard: bool,
}

/// Returns a closure that will return `""` when called the first time,
/// and then return `", "` when called any subsequent times.
/// Useful for printing comma-separated lists.
fn start_or_comma() -> impl FnMut() -> &'static str {
    let mut first = true;
    move || {
        if first {
            first = false;
            ""
        } else {
            ", "
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum EnumInfo<'tcx> {
    Enum { adt_def: AdtDef<'tcx>, variant_index: VariantIdx },
    NotEnum,
}

pub(crate) fn write_struct_like<'tcx>(
    f: &mut impl fmt::Write,
    tcx: TyCtxt<'_>,
    ty: Ty<'tcx>,
    enum_info: &EnumInfo<'tcx>,
    subpatterns: &[FieldPat],
) -> fmt::Result {
    let variant_and_name = match *enum_info {
        EnumInfo::Enum { adt_def, variant_index } => {
            let variant = adt_def.variant(variant_index);
            let adt_did = adt_def.did();
            let name = if tcx.is_diagnostic_item(sym::Option, adt_did)
                || tcx.is_diagnostic_item(sym::Result, adt_did)
            {
                variant.name.to_string()
            } else {
                format!("{}::{}", tcx.def_path_str(adt_def.did()), variant.name)
            };
            Some((variant, name))
        }
        EnumInfo::NotEnum => ty.ty_adt_def().and_then(|adt_def| {
            Some((adt_def.non_enum_variant(), tcx.def_path_str(adt_def.did())))
        }),
    };

    let mut start_or_comma = start_or_comma();

    if let Some((variant, name)) = &variant_and_name {
        write!(f, "{name}")?;

        // Only for Adt we can have `S {...}`,
        // which we handle separately here.
        if variant.ctor.is_none() {
            write!(f, " {{ ")?;

            let mut printed = 0;
            for &FieldPat { field, ref pattern, is_wildcard } in subpatterns {
                if is_wildcard {
                    continue;
                }
                let field_name = variant.fields[field].name;
                write!(f, "{}{field_name}: {pattern}", start_or_comma())?;
                printed += 1;
            }

            let is_union = ty.ty_adt_def().is_some_and(|adt| adt.is_union());
            if printed < variant.fields.len() && (!is_union || printed == 0) {
                write!(f, "{}..", start_or_comma())?;
            }

            return write!(f, " }}");
        }
    }

    let num_fields = variant_and_name.as_ref().map_or(subpatterns.len(), |(v, _)| v.fields.len());
    if num_fields != 0 || variant_and_name.is_none() {
        write!(f, "(")?;
        for FieldPat { pattern, .. } in subpatterns {
            write!(f, "{}{pattern}", start_or_comma())?;
        }
        if matches!(ty.kind(), ty::Tuple(..)) && num_fields == 1 {
            write!(f, ",")?;
        }
        write!(f, ")")?;
    }

    Ok(())
}

pub(crate) fn write_ref_like<'tcx>(
    f: &mut impl fmt::Write,
    ty: Ty<'tcx>,
    subpattern: &str,
) -> fmt::Result {
    match ty.kind() {
        ty::Ref(_, _, mutbl) => {
            write!(f, "&{}", mutbl.prefix_str())?;
        }
        _ => bug!("{ty} is a bad ref pattern type"),
    }
    write!(f, "{subpattern}")
}

pub(crate) fn write_slice_like(
    f: &mut impl fmt::Write,
    prefix: &[String],
    has_dot_dot: bool,
    suffix: &[String],
) -> fmt::Result {
    let mut start_or_comma = start_or_comma();
    write!(f, "[")?;
    for p in prefix.iter() {
        write!(f, "{}{}", start_or_comma(), p)?;
    }
    if has_dot_dot {
        write!(f, "{}..", start_or_comma())?;
    }
    for p in suffix.iter() {
        write!(f, "{}{}", start_or_comma(), p)?;
    }
    write!(f, "]")
}
