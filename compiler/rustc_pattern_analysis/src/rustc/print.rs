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

    Variant {
        adt_def: AdtDef<'tcx>,
        variant_index: VariantIdx,
        subpatterns: Vec<FieldPat<'tcx>>,
    },

    Leaf {
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
        // Printing lists is a chore.
        let mut first = true;
        let mut start_or_continue = |s| {
            if first {
                first = false;
                ""
            } else {
                s
            }
        };
        let mut start_or_comma = || start_or_continue(", ");

        match self.kind {
            PatKind::Wild => write!(f, "_"),
            PatKind::Never => write!(f, "!"),
            PatKind::Variant { ref subpatterns, .. } | PatKind::Leaf { ref subpatterns } => {
                let variant_and_name = match self.kind {
                    PatKind::Variant { adt_def, variant_index, .. } => ty::tls::with(|tcx| {
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
                    _ => self.ty.ty_adt_def().and_then(|adt_def| {
                        if !adt_def.is_enum() {
                            ty::tls::with(|tcx| {
                                Some((adt_def.non_enum_variant(), tcx.def_path_str(adt_def.did())))
                            })
                        } else {
                            None
                        }
                    }),
                };

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

                        let is_union = self.ty.ty_adt_def().is_some_and(|adt| adt.is_union());
                        if printed < variant.fields.len() && (!is_union || printed == 0) {
                            write!(f, "{}..", start_or_comma())?;
                        }

                        return write!(f, " }}");
                    }
                }

                let num_fields =
                    variant_and_name.as_ref().map_or(subpatterns.len(), |(v, _)| v.fields.len());
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
            PatKind::Deref { ref subpattern } => {
                match self.ty.kind() {
                    ty::Adt(def, _) if def.is_box() => write!(f, "box ")?,
                    ty::Ref(_, _, mutbl) => {
                        write!(f, "&{}", mutbl.prefix_str())?;
                    }
                    _ => bug!("{} is a bad Deref pattern type", self.ty),
                }
                write!(f, "{subpattern}")
            }
            PatKind::Constant { value } => write!(f, "{value}"),
            PatKind::Range(ref range) => write!(f, "{range}"),
            PatKind::Slice { ref prefix, ref slice, ref suffix } => {
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
        }
    }
}
