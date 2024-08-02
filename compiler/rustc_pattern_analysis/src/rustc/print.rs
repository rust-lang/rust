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

use rustc_middle::thir::PatRange;
use rustc_middle::ty::{self, AdtDef, Ty};
use rustc_middle::{bug, mir};
use rustc_span::sym;
use rustc_target::abi::{FieldIdx, VariantIdx};

#[derive(Clone, Debug)]
pub(crate) struct FieldPat<'tcx> {
    pub(crate) field: FieldIdx,
    pub(crate) pattern: Box<Pat<'tcx>>,
}

#[derive(Clone, Debug)]
pub(crate) struct Pat<'tcx> {
    pub(crate) ty: Ty<'tcx>,
    pub(crate) kind: PatKind<'tcx>,
}

#[derive(Clone, Debug)]
pub(crate) enum PatKind<'tcx> {
    Wild,

    StructLike {
        enum_info: EnumInfo<'tcx>,
        subpatterns: Vec<FieldPat<'tcx>>,
    },

    Deref {
        subpattern: Box<Pat<'tcx>>,
    },

    Constant {
        value: mir::Const<'tcx>,
    },

    Range(Box<PatRange<'tcx>>),

    Slice {
        prefix: Box<[Box<Pat<'tcx>>]>,
        slice: Option<Box<Pat<'tcx>>>,
        suffix: Box<[Box<Pat<'tcx>>]>,
    },

    Never,
}

impl<'tcx> fmt::Display for Pat<'tcx> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            PatKind::Wild => write!(f, "_"),
            PatKind::Never => write!(f, "!"),
            PatKind::StructLike { ref enum_info, ref subpatterns } => {
                write_struct_like(f, self.ty, enum_info, subpatterns)
            }
            PatKind::Deref { ref subpattern } => write_ref_like(f, self.ty, subpattern),
            PatKind::Constant { value } => write!(f, "{value}"),
            PatKind::Range(ref range) => write!(f, "{range}"),
            PatKind::Slice { ref prefix, ref slice, ref suffix } => {
                write_slice_like(f, prefix, slice, suffix)
            }
        }
    }
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

fn write_struct_like<'tcx>(
    f: &mut impl fmt::Write,
    ty: Ty<'tcx>,
    enum_info: &EnumInfo<'tcx>,
    subpatterns: &[FieldPat<'tcx>],
) -> fmt::Result {
    let variant_and_name = match *enum_info {
        EnumInfo::Enum { adt_def, variant_index } => ty::tls::with(|tcx| {
            let variant = adt_def.variant(variant_index);
            let adt_did = adt_def.did();
            let name = if tcx.get_diagnostic_item(sym::Option) == Some(adt_did)
                || tcx.get_diagnostic_item(sym::Result) == Some(adt_did)
            {
                variant.name.to_string()
            } else {
                format!("{}::{}", tcx.def_path_str(adt_def.did()), variant.name)
            };
            Some((variant, name))
        }),
        EnumInfo::NotEnum => ty.ty_adt_def().and_then(|adt_def| {
            if !adt_def.is_enum() {
                ty::tls::with(|tcx| {
                    Some((adt_def.non_enum_variant(), tcx.def_path_str(adt_def.did())))
                })
            } else {
                None
            }
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
            for p in subpatterns {
                if let PatKind::Wild = p.pattern.kind {
                    continue;
                }
                let name = variant.fields[p.field].name;
                write!(f, "{}{}: {}", start_or_comma(), name, p.pattern)?;
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
        for i in 0..num_fields {
            write!(f, "{}", start_or_comma())?;

            // Common case: the field is where we expect it.
            if let Some(p) = subpatterns.get(i) {
                if p.field.index() == i {
                    write!(f, "{}", p.pattern)?;
                    continue;
                }
            }

            // Otherwise, we have to go looking for it.
            if let Some(p) = subpatterns.iter().find(|p| p.field.index() == i) {
                write!(f, "{}", p.pattern)?;
            } else {
                write!(f, "_")?;
            }
        }
        write!(f, ")")?;
    }

    Ok(())
}

fn write_ref_like<'tcx>(
    f: &mut impl fmt::Write,
    ty: Ty<'tcx>,
    subpattern: &Pat<'tcx>,
) -> fmt::Result {
    match ty.kind() {
        ty::Adt(def, _) if def.is_box() => write!(f, "box ")?,
        ty::Ref(_, _, mutbl) => {
            write!(f, "&{}", mutbl.prefix_str())?;
        }
        _ => bug!("{ty} is a bad ref pattern type"),
    }
    write!(f, "{subpattern}")
}

fn write_slice_like<'tcx>(
    f: &mut impl fmt::Write,
    prefix: &[Box<Pat<'tcx>>],
    slice: &Option<Box<Pat<'tcx>>>,
    suffix: &[Box<Pat<'tcx>>],
) -> fmt::Result {
    let mut start_or_comma = start_or_comma();
    write!(f, "[")?;
    for p in prefix.iter() {
        write!(f, "{}{}", start_or_comma(), p)?;
    }
    if let Some(ref slice) = *slice {
        write!(f, "{}", start_or_comma())?;
        match slice.kind {
            PatKind::Wild => {}
            _ => write!(f, "{slice}")?,
        }
        write!(f, "..")?;
    }
    for p in suffix.iter() {
        write!(f, "{}{}", start_or_comma(), p)?;
    }
    write!(f, "]")
}
