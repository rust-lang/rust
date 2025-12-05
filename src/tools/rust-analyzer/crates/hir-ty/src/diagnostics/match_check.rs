//! Validation of matches.
//!
//! This module provides lowering from [hir_def::hir::Pat] to [self::Pat] and match
//! checking algorithm.
//!
//! It is modeled on the rustc module `rustc_mir_build::thir::pattern`.

mod pat_util;

pub(crate) mod pat_analysis;

use hir_def::{
    AdtId, EnumVariantId, LocalFieldId, Lookup, VariantId,
    expr_store::{Body, path::Path},
    hir::PatId,
    item_tree::FieldsShape,
};
use hir_expand::name::Name;
use rustc_type_ir::inherent::{IntoKind, SliceLike};
use span::Edition;
use stdx::{always, never, variance::PhantomCovariantLifetime};

use crate::{
    InferenceResult,
    db::HirDatabase,
    display::{HirDisplay, HirDisplayError, HirFormatter},
    infer::BindingMode,
    next_solver::{GenericArgs, Mutability, Ty, TyKind},
};

use self::pat_util::EnumerateAndAdjustIterator;

#[derive(Clone, Debug)]
pub(crate) enum PatternError {
    Unimplemented,
    UnexpectedType,
    UnresolvedVariant,
    MissingField,
    ExtraFields,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FieldPat<'db> {
    pub(crate) field: LocalFieldId,
    pub(crate) pattern: Pat<'db>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Pat<'db> {
    pub(crate) ty: Ty<'db>,
    pub(crate) kind: Box<PatKind<'db>>,
}

/// Close relative to `rustc_mir_build::thir::pattern::PatKind`
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum PatKind<'db> {
    Wild,
    Never,

    /// `x`, `ref x`, `x @ P`, etc.
    Binding {
        name: Name,
        subpattern: Option<Pat<'db>>,
    },

    /// `Foo(...)` or `Foo{...}` or `Foo`, where `Foo` is a variant name from an ADT with
    /// multiple variants.
    Variant {
        substs: GenericArgs<'db>,
        enum_variant: EnumVariantId,
        subpatterns: Vec<FieldPat<'db>>,
    },

    /// `(...)`, `Foo(...)`, `Foo{...}`, or `Foo`, where `Foo` is a variant name from an ADT with
    /// a single variant.
    Leaf {
        subpatterns: Vec<FieldPat<'db>>,
    },

    /// `&P`, `&mut P`, etc.
    Deref {
        subpattern: Pat<'db>,
    },

    // FIXME: for now, only bool literals are implemented
    LiteralBool {
        value: bool,
    },

    /// An or-pattern, e.g. `p | q`.
    /// Invariant: `pats.len() >= 2`.
    Or {
        pats: Vec<Pat<'db>>,
    },
}

pub(crate) struct PatCtxt<'a, 'db> {
    db: &'db dyn HirDatabase,
    infer: &'a InferenceResult<'db>,
    body: &'a Body,
    pub(crate) errors: Vec<PatternError>,
}

impl<'a, 'db> PatCtxt<'a, 'db> {
    pub(crate) fn new(
        db: &'db dyn HirDatabase,
        infer: &'a InferenceResult<'db>,
        body: &'a Body,
    ) -> Self {
        Self { db, infer, body, errors: Vec::new() }
    }

    pub(crate) fn lower_pattern(&mut self, pat: PatId) -> Pat<'db> {
        // XXX(iDawer): Collecting pattern adjustments feels imprecise to me.
        // When lowering of & and box patterns are implemented this should be tested
        // in a manner of `match_ergonomics_issue_9095` test.
        // Pattern adjustment is part of RFC 2005-match-ergonomics.
        // More info https://github.com/rust-lang/rust/issues/42640#issuecomment-313535089
        let unadjusted_pat = self.lower_pattern_unadjusted(pat);
        self.infer.pat_adjustments.get(&pat).map(|it| &**it).unwrap_or_default().iter().rev().fold(
            unadjusted_pat,
            |subpattern, ref_ty| Pat { ty: *ref_ty, kind: Box::new(PatKind::Deref { subpattern }) },
        )
    }

    fn lower_pattern_unadjusted(&mut self, pat: PatId) -> Pat<'db> {
        let mut ty = self.infer[pat];
        let variant = self.infer.variant_resolution_for_pat(pat);

        let kind = match self.body[pat] {
            hir_def::hir::Pat::Wild => PatKind::Wild,

            hir_def::hir::Pat::Lit(expr) => self.lower_lit(expr),

            hir_def::hir::Pat::Path(ref path) => {
                return self.lower_path(pat, path);
            }

            hir_def::hir::Pat::Tuple { ref args, ellipsis } => {
                let arity = match ty.kind() {
                    TyKind::Tuple(tys) => tys.len(),
                    _ => {
                        never!("unexpected type for tuple pattern: {:?}", ty);
                        self.errors.push(PatternError::UnexpectedType);
                        return Pat { ty, kind: PatKind::Wild.into() };
                    }
                };
                let subpatterns = self.lower_tuple_subpats(args, arity, ellipsis);
                PatKind::Leaf { subpatterns }
            }

            hir_def::hir::Pat::Bind { id, subpat, .. } => {
                let bm = self.infer.binding_modes[pat];
                ty = self.infer[id];
                let name = &self.body[id].name;
                match (bm, ty.kind()) {
                    (BindingMode::Ref(_), TyKind::Ref(_, rty, _)) => ty = rty,
                    (BindingMode::Ref(_), _) => {
                        never!(
                            "`ref {}` has wrong type {:?}",
                            name.display(self.db, Edition::LATEST),
                            ty
                        );
                        self.errors.push(PatternError::UnexpectedType);
                        return Pat { ty, kind: PatKind::Wild.into() };
                    }
                    _ => (),
                }
                PatKind::Binding { name: name.clone(), subpattern: self.lower_opt_pattern(subpat) }
            }

            hir_def::hir::Pat::TupleStruct { ref args, ellipsis, .. } if variant.is_some() => {
                let expected_len = variant.unwrap().fields(self.db).fields().len();
                let subpatterns = self.lower_tuple_subpats(args, expected_len, ellipsis);
                self.lower_variant_or_leaf(pat, ty, subpatterns)
            }

            hir_def::hir::Pat::Record { ref args, .. } if variant.is_some() => {
                let variant_data = variant.unwrap().fields(self.db);
                let subpatterns = args
                    .iter()
                    .map(|field| {
                        // XXX(iDawer): field lookup is inefficient
                        variant_data.field(&field.name).map(|lfield_id| FieldPat {
                            field: lfield_id,
                            pattern: self.lower_pattern(field.pat),
                        })
                    })
                    .collect();
                match subpatterns {
                    Some(subpatterns) => self.lower_variant_or_leaf(pat, ty, subpatterns),
                    None => {
                        self.errors.push(PatternError::MissingField);
                        PatKind::Wild
                    }
                }
            }
            hir_def::hir::Pat::TupleStruct { .. } | hir_def::hir::Pat::Record { .. } => {
                self.errors.push(PatternError::UnresolvedVariant);
                PatKind::Wild
            }

            hir_def::hir::Pat::Or(ref pats) => PatKind::Or { pats: self.lower_patterns(pats) },

            _ => {
                self.errors.push(PatternError::Unimplemented);
                PatKind::Wild
            }
        };

        Pat { ty, kind: Box::new(kind) }
    }

    fn lower_tuple_subpats(
        &mut self,
        pats: &[PatId],
        expected_len: usize,
        ellipsis: Option<u32>,
    ) -> Vec<FieldPat<'db>> {
        if pats.len() > expected_len {
            self.errors.push(PatternError::ExtraFields);
            return Vec::new();
        }

        pats.iter()
            .enumerate_and_adjust(expected_len, ellipsis.map(|it| it as usize))
            .map(|(i, &subpattern)| FieldPat {
                field: LocalFieldId::from_raw((i as u32).into()),
                pattern: self.lower_pattern(subpattern),
            })
            .collect()
    }

    fn lower_patterns(&mut self, pats: &[PatId]) -> Vec<Pat<'db>> {
        pats.iter().map(|&p| self.lower_pattern(p)).collect()
    }

    fn lower_opt_pattern(&mut self, pat: Option<PatId>) -> Option<Pat<'db>> {
        pat.map(|p| self.lower_pattern(p))
    }

    fn lower_variant_or_leaf(
        &mut self,
        pat: PatId,
        ty: Ty<'db>,
        subpatterns: Vec<FieldPat<'db>>,
    ) -> PatKind<'db> {
        match self.infer.variant_resolution_for_pat(pat) {
            Some(variant_id) => {
                if let VariantId::EnumVariantId(enum_variant) = variant_id {
                    let substs = match ty.kind() {
                        TyKind::Adt(_, substs) => substs,
                        kind => {
                            always!(
                                matches!(kind, TyKind::FnDef(..) | TyKind::Error(_)),
                                "inappropriate type for def: {:?}",
                                ty
                            );
                            self.errors.push(PatternError::UnexpectedType);
                            return PatKind::Wild;
                        }
                    };
                    PatKind::Variant { substs, enum_variant, subpatterns }
                } else {
                    PatKind::Leaf { subpatterns }
                }
            }
            None => {
                self.errors.push(PatternError::UnresolvedVariant);
                PatKind::Wild
            }
        }
    }

    fn lower_path(&mut self, pat: PatId, _path: &Path) -> Pat<'db> {
        let ty = self.infer[pat];

        let pat_from_kind = |kind| Pat { ty, kind: Box::new(kind) };

        match self.infer.variant_resolution_for_pat(pat) {
            Some(_) => pat_from_kind(self.lower_variant_or_leaf(pat, ty, Vec::new())),
            None => {
                self.errors.push(PatternError::UnresolvedVariant);
                pat_from_kind(PatKind::Wild)
            }
        }
    }

    fn lower_lit(&mut self, expr: hir_def::hir::ExprId) -> PatKind<'db> {
        use hir_def::hir::{Expr, Literal::Bool};

        match self.body[expr] {
            Expr::Literal(Bool(value)) => PatKind::LiteralBool { value },
            _ => {
                self.errors.push(PatternError::Unimplemented);
                PatKind::Wild
            }
        }
    }
}

impl<'db> HirDisplay<'db> for Pat<'db> {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>) -> Result<(), HirDisplayError> {
        match &*self.kind {
            PatKind::Wild => write!(f, "_"),
            PatKind::Never => write!(f, "!"),
            PatKind::Binding { name, subpattern } => {
                write!(f, "{}", name.display(f.db, f.edition()))?;
                if let Some(subpattern) = subpattern {
                    write!(f, " @ ")?;
                    subpattern.hir_fmt(f)?;
                }
                Ok(())
            }
            PatKind::Variant { subpatterns, .. } | PatKind::Leaf { subpatterns } => {
                let variant = match *self.kind {
                    PatKind::Variant { enum_variant, .. } => Some(VariantId::from(enum_variant)),
                    _ => self.ty.as_adt().and_then(|(adt, _)| match adt {
                        AdtId::StructId(s) => Some(s.into()),
                        AdtId::UnionId(u) => Some(u.into()),
                        AdtId::EnumId(_) => None,
                    }),
                };

                if let Some(variant) = variant {
                    match variant {
                        VariantId::EnumVariantId(v) => {
                            let loc = v.lookup(f.db);
                            write!(
                                f,
                                "{}",
                                loc.parent.enum_variants(f.db).variants[loc.index as usize]
                                    .1
                                    .display(f.db, f.edition())
                            )?;
                        }
                        VariantId::StructId(s) => write!(
                            f,
                            "{}",
                            f.db.struct_signature(s).name.display(f.db, f.edition())
                        )?,
                        VariantId::UnionId(u) => write!(
                            f,
                            "{}",
                            f.db.union_signature(u).name.display(f.db, f.edition())
                        )?,
                    };

                    let variant_data = variant.fields(f.db);
                    if variant_data.shape == FieldsShape::Record {
                        write!(f, " {{ ")?;

                        let mut printed = 0;
                        let subpats = subpatterns
                            .iter()
                            .filter(|p| !matches!(*p.pattern.kind, PatKind::Wild))
                            .map(|p| {
                                printed += 1;
                                WriteWith::new(|f| {
                                    write!(
                                        f,
                                        "{}: ",
                                        variant_data.fields()[p.field]
                                            .name
                                            .display(f.db, f.edition())
                                    )?;
                                    p.pattern.hir_fmt(f)
                                })
                            });
                        f.write_joined(subpats, ", ")?;

                        if printed < variant_data.fields().len() {
                            write!(f, "{}..", if printed > 0 { ", " } else { "" })?;
                        }

                        return write!(f, " }}");
                    }
                }

                let num_fields =
                    variant.map_or(subpatterns.len(), |v| v.fields(f.db).fields().len());
                if num_fields != 0 || variant.is_none() {
                    write!(f, "(")?;
                    let subpats = (0..num_fields).map(|i| {
                        WriteWith::new(move |f| {
                            let fid = LocalFieldId::from_raw((i as u32).into());
                            if let Some(p) = subpatterns.get(i)
                                && p.field == fid
                            {
                                return p.pattern.hir_fmt(f);
                            }
                            if let Some(p) = subpatterns.iter().find(|p| p.field == fid) {
                                p.pattern.hir_fmt(f)
                            } else {
                                write!(f, "_")
                            }
                        })
                    });
                    f.write_joined(subpats, ", ")?;
                    if let (TyKind::Tuple(..), 1) = (self.ty.kind(), num_fields) {
                        write!(f, ",")?;
                    }
                    write!(f, ")")?;
                }

                Ok(())
            }
            PatKind::Deref { subpattern } => {
                match self.ty.kind() {
                    TyKind::Ref(.., mutbl) => {
                        write!(f, "&{}", if mutbl == Mutability::Mut { "mut " } else { "" })?
                    }
                    _ => never!("{:?} is a bad Deref pattern type", self.ty),
                }
                subpattern.hir_fmt(f)
            }
            PatKind::LiteralBool { value } => write!(f, "{value}"),
            PatKind::Or { pats } => f.write_joined(pats.iter(), " | "),
        }
    }
}

struct WriteWith<'db, F>(F, PhantomCovariantLifetime<'db>)
where
    F: Fn(&mut HirFormatter<'_, 'db>) -> Result<(), HirDisplayError>;

impl<'db, F> WriteWith<'db, F>
where
    F: Fn(&mut HirFormatter<'_, 'db>) -> Result<(), HirDisplayError>,
{
    fn new(f: F) -> Self {
        Self(f, PhantomCovariantLifetime::new())
    }
}

impl<'db, F> HirDisplay<'db> for WriteWith<'db, F>
where
    F: Fn(&mut HirFormatter<'_, 'db>) -> Result<(), HirDisplayError>,
{
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>) -> Result<(), HirDisplayError> {
        (self.0)(f)
    }
}
