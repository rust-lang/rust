use super::{Byte, Def, Ref};
use std::ops::ControlFlow;

#[cfg(test)]
mod tests;

/// A tree-based representation of a type layout.
///
/// Invariants:
/// 1. All paths through the layout have the same length (in bytes).
///
/// Nice-to-haves:
/// 1. An `Alt` is never directly nested beneath another `Alt`.
/// 2. A `Seq` is never directly nested beneath another `Seq`.
/// 3. `Seq`s and `Alt`s with a single member do not exist.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub(crate) enum Tree<D, R>
where
    D: Def,
    R: Ref,
{
    /// A sequence of successive layouts.
    Seq(Vec<Self>),
    /// A choice between alternative layouts.
    Alt(Vec<Self>),
    /// A definition node.
    Def(D),
    /// A reference node.
    Ref(R),
    /// A byte node.
    Byte(Byte),
}

impl<D, R> Tree<D, R>
where
    D: Def,
    R: Ref,
{
    /// A `Tree` consisting only of a definition node.
    pub(crate) fn def(def: D) -> Self {
        Self::Def(def)
    }

    /// A `Tree` representing an uninhabited type.
    pub(crate) fn uninhabited() -> Self {
        Self::Alt(vec![])
    }

    /// A `Tree` representing a zero-sized type.
    pub(crate) fn unit() -> Self {
        Self::Seq(Vec::new())
    }

    /// A `Tree` containing a single, uninitialized byte.
    pub(crate) fn uninit() -> Self {
        Self::Byte(Byte::Uninit)
    }

    /// A `Tree` representing the layout of `bool`.
    pub(crate) fn bool() -> Self {
        Self::from_bits(0x00).or(Self::from_bits(0x01))
    }

    /// A `Tree` whose layout matches that of a `u8`.
    pub(crate) fn u8() -> Self {
        Self::Alt((0u8..=255).map(Self::from_bits).collect())
    }

    /// A `Tree` whose layout accepts exactly the given bit pattern.
    pub(crate) fn from_bits(bits: u8) -> Self {
        Self::Byte(Byte::Init(bits))
    }

    /// A `Tree` whose layout is a number of the given width.
    pub(crate) fn number(width_in_bytes: usize) -> Self {
        Self::Seq(vec![Self::u8(); width_in_bytes])
    }

    /// A `Tree` whose layout is entirely padding of the given width.
    pub(crate) fn padding(width_in_bytes: usize) -> Self {
        Self::Seq(vec![Self::uninit(); width_in_bytes])
    }

    /// Remove all `Def` nodes, and all branches of the layout for which `f` produces false.
    pub(crate) fn prune<F>(self, f: &F) -> Tree<!, R>
    where
        F: Fn(D) -> bool,
    {
        match self {
            Self::Seq(elts) => match elts.into_iter().map(|elt| elt.prune(f)).try_fold(
                Tree::unit(),
                |elts, elt| {
                    if elt == Tree::uninhabited() {
                        ControlFlow::Break(Tree::uninhabited())
                    } else {
                        ControlFlow::Continue(elts.then(elt))
                    }
                },
            ) {
                ControlFlow::Break(node) | ControlFlow::Continue(node) => node,
            },
            Self::Alt(alts) => alts
                .into_iter()
                .map(|alt| alt.prune(f))
                .fold(Tree::uninhabited(), |alts, alt| alts.or(alt)),
            Self::Byte(b) => Tree::Byte(b),
            Self::Ref(r) => Tree::Ref(r),
            Self::Def(d) => {
                if !f(d) {
                    Tree::uninhabited()
                } else {
                    Tree::unit()
                }
            }
        }
    }

    /// Produces `true` if `Tree` is an inhabited type; otherwise false.
    pub(crate) fn is_inhabited(&self) -> bool {
        match self {
            Self::Seq(elts) => elts.into_iter().all(|elt| elt.is_inhabited()),
            Self::Alt(alts) => alts.into_iter().any(|alt| alt.is_inhabited()),
            Self::Byte(..) | Self::Ref(..) | Self::Def(..) => true,
        }
    }
}

impl<D, R> Tree<D, R>
where
    D: Def,
    R: Ref,
{
    /// Produces a new `Tree` where `other` is sequenced after `self`.
    pub(crate) fn then(self, other: Self) -> Self {
        match (self, other) {
            (Self::Seq(elts), other) | (other, Self::Seq(elts)) if elts.len() == 0 => other,
            (Self::Seq(mut lhs), Self::Seq(mut rhs)) => {
                lhs.append(&mut rhs);
                Self::Seq(lhs)
            }
            (Self::Seq(mut lhs), rhs) => {
                lhs.push(rhs);
                Self::Seq(lhs)
            }
            (lhs, Self::Seq(mut rhs)) => {
                rhs.insert(0, lhs);
                Self::Seq(rhs)
            }
            (lhs, rhs) => Self::Seq(vec![lhs, rhs]),
        }
    }

    /// Produces a new `Tree` accepting either `self` or `other` as alternative layouts.
    pub(crate) fn or(self, other: Self) -> Self {
        match (self, other) {
            (Self::Alt(alts), other) | (other, Self::Alt(alts)) if alts.len() == 0 => other,
            (Self::Alt(mut lhs), Self::Alt(rhs)) => {
                lhs.extend(rhs);
                Self::Alt(lhs)
            }
            (Self::Alt(mut alts), alt) | (alt, Self::Alt(mut alts)) => {
                alts.push(alt);
                Self::Alt(alts)
            }
            (lhs, rhs) => Self::Alt(vec![lhs, rhs]),
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub(crate) enum Err {
    /// The layout of the type is unspecified.
    Unspecified,
    /// This error will be surfaced elsewhere by rustc, so don't surface it.
    Unknown,
}

#[cfg(feature = "rustc")]
pub(crate) mod rustc {
    use super::{Err, Tree};
    use crate::layout::rustc::{Def, Ref};

    use rustc_middle::ty;
    use rustc_middle::ty::layout::LayoutError;
    use rustc_middle::ty::util::Discr;
    use rustc_middle::ty::AdtDef;
    use rustc_middle::ty::ParamEnv;
    use rustc_middle::ty::SubstsRef;
    use rustc_middle::ty::Ty;
    use rustc_middle::ty::TyCtxt;
    use rustc_middle::ty::VariantDef;
    use rustc_target::abi::Align;
    use std::alloc;

    impl<'tcx> From<LayoutError<'tcx>> for Err {
        fn from(err: LayoutError<'tcx>) -> Self {
            match err {
                LayoutError::Unknown(..) => Self::Unknown,
                err @ _ => unimplemented!("{:?}", err),
            }
        }
    }

    trait LayoutExt {
        fn clamp_align(&self, min_align: Align, max_align: Align) -> Self;
    }

    impl LayoutExt for alloc::Layout {
        fn clamp_align(&self, min_align: Align, max_align: Align) -> Self {
            let min_align = min_align.bytes().try_into().unwrap();
            let max_align = max_align.bytes().try_into().unwrap();
            Self::from_size_align(self.size(), self.align().clamp(min_align, max_align)).unwrap()
        }
    }

    struct LayoutSummary {
        total_align: Align,
        total_size: usize,
        discriminant_size: usize,
        discriminant_align: Align,
    }

    impl LayoutSummary {
        fn from_ty<'tcx>(ty: Ty<'tcx>, ctx: TyCtxt<'tcx>) -> Result<Self, LayoutError<'tcx>> {
            use rustc_middle::ty::ParamEnvAnd;
            use rustc_target::abi::{TyAndLayout, Variants};

            let param_env = ParamEnv::reveal_all();
            let param_env_and_type = ParamEnvAnd { param_env, value: ty };
            let TyAndLayout { layout, .. } = ctx.layout_of(param_env_and_type)?;

            let total_size: usize = layout.size().bytes_usize();
            let total_align: Align = layout.align().abi;
            let discriminant_align: Align;
            let discriminant_size: usize;

            if let Variants::Multiple { tag, .. } = layout.variants() {
                discriminant_align = tag.align(&ctx).abi;
                discriminant_size = tag.size(&ctx).bytes_usize();
            } else {
                discriminant_align = Align::ONE;
                discriminant_size = 0;
            };

            Ok(Self { total_align, total_size, discriminant_align, discriminant_size })
        }

        fn into(&self) -> alloc::Layout {
            alloc::Layout::from_size_align(
                self.total_size,
                self.total_align.bytes().try_into().unwrap(),
            )
            .unwrap()
        }
    }

    impl<'tcx> Tree<Def<'tcx>, Ref<'tcx>> {
        pub fn from_ty(ty: Ty<'tcx>, tcx: TyCtxt<'tcx>) -> Result<Self, Err> {
            use rustc_middle::ty::FloatTy::*;
            use rustc_middle::ty::IntTy::*;
            use rustc_middle::ty::UintTy::*;
            use rustc_target::abi::HasDataLayout;

            let target = tcx.data_layout();

            match ty.kind() {
                ty::Bool => Ok(Self::bool()),

                ty::Int(I8) | ty::Uint(U8) => Ok(Self::u8()),
                ty::Int(I16) | ty::Uint(U16) => Ok(Self::number(2)),
                ty::Int(I32) | ty::Uint(U32) | ty::Float(F32) => Ok(Self::number(4)),
                ty::Int(I64) | ty::Uint(U64) | ty::Float(F64) => Ok(Self::number(8)),
                ty::Int(I128) | ty::Uint(U128) => Ok(Self::number(16)),
                ty::Int(Isize) | ty::Uint(Usize) => {
                    Ok(Self::number(target.pointer_size.bytes_usize()))
                }

                ty::Tuple(members) => {
                    if members.len() == 0 {
                        Ok(Tree::unit())
                    } else {
                        Err(Err::Unspecified)
                    }
                }

                ty::Array(ty, len) => {
                    let len = len
                        .try_eval_target_usize(tcx, ParamEnv::reveal_all())
                        .ok_or(Err::Unspecified)?;
                    let elt = Tree::from_ty(*ty, tcx)?;
                    Ok(std::iter::repeat(elt)
                        .take(len as usize)
                        .fold(Tree::unit(), |tree, elt| tree.then(elt)))
                }

                ty::Adt(adt_def, substs_ref) => {
                    use rustc_middle::ty::AdtKind;

                    // If the layout is ill-specified, halt.
                    if !(adt_def.repr().c() || adt_def.repr().int.is_some()) {
                        return Err(Err::Unspecified);
                    }

                    // Compute a summary of the type's layout.
                    let layout_summary = LayoutSummary::from_ty(ty, tcx)?;

                    // The layout begins with this adt's visibility.
                    let vis = Self::def(Def::Adt(*adt_def));

                    // And is followed the layout(s) of its variants
                    Ok(vis.then(match adt_def.adt_kind() {
                        AdtKind::Struct => Self::from_repr_c_variant(
                            ty,
                            *adt_def,
                            substs_ref,
                            &layout_summary,
                            None,
                            adt_def.non_enum_variant(),
                            tcx,
                        )?,
                        AdtKind::Enum => {
                            trace!(?adt_def, "treeifying enum");
                            let mut tree = Tree::uninhabited();

                            for (idx, discr) in adt_def.discriminants(tcx) {
                                tree = tree.or(Self::from_repr_c_variant(
                                    ty,
                                    *adt_def,
                                    substs_ref,
                                    &layout_summary,
                                    Some(discr),
                                    adt_def.variant(idx),
                                    tcx,
                                )?);
                            }

                            tree
                        }
                        AdtKind::Union => {
                            // is the layout well-defined?
                            if !adt_def.repr().c() {
                                return Err(Err::Unspecified);
                            }

                            let ty_layout = layout_of(tcx, ty)?;

                            let mut tree = Tree::uninhabited();

                            for field in adt_def.all_fields() {
                                let variant_ty = field.ty(tcx, substs_ref);
                                let variant_layout = layout_of(tcx, variant_ty)?;
                                let padding_needed = ty_layout.size() - variant_layout.size();
                                let variant = Self::def(Def::Field(field))
                                    .then(Self::from_ty(variant_ty, tcx)?)
                                    .then(Self::padding(padding_needed));

                                tree = tree.or(variant);
                            }

                            tree
                        }
                    }))
                }
                _ => Err(Err::Unspecified),
            }
        }

        fn from_repr_c_variant(
            ty: Ty<'tcx>,
            adt_def: AdtDef<'tcx>,
            substs_ref: SubstsRef<'tcx>,
            layout_summary: &LayoutSummary,
            discr: Option<Discr<'tcx>>,
            variant_def: &'tcx VariantDef,
            tcx: TyCtxt<'tcx>,
        ) -> Result<Self, Err> {
            let mut tree = Tree::unit();

            let repr = adt_def.repr();
            let min_align = repr.align.unwrap_or(Align::ONE);
            let max_align = repr.pack.unwrap_or(Align::MAX);

            let clamp =
                |align: Align| align.clamp(min_align, max_align).bytes().try_into().unwrap();

            let variant_span = trace_span!(
                "treeifying variant",
                min_align = ?min_align,
                max_align = ?max_align,
            )
            .entered();

            let mut variant_layout = alloc::Layout::from_size_align(
                0,
                layout_summary.total_align.bytes().try_into().unwrap(),
            )
            .unwrap();

            // The layout of the variant is prefixed by the discriminant, if any.
            if let Some(discr) = discr {
                trace!(?discr, "treeifying discriminant");
                let discr_layout = alloc::Layout::from_size_align(
                    layout_summary.discriminant_size,
                    clamp(layout_summary.discriminant_align),
                )
                .unwrap();
                trace!(?discr_layout, "computed discriminant layout");
                variant_layout = variant_layout.extend(discr_layout).unwrap().0;
                tree = tree.then(Self::from_discr(discr, tcx, layout_summary.discriminant_size));
            }

            // Next come fields.
            let fields_span = trace_span!("treeifying fields").entered();
            for field_def in variant_def.fields.iter() {
                let field_ty = field_def.ty(tcx, substs_ref);
                let _span = trace_span!("treeifying field", field = ?field_ty).entered();

                // begin with the field's visibility
                tree = tree.then(Self::def(Def::Field(field_def)));

                // compute the field's layout characteristics
                let field_layout = layout_of(tcx, field_ty)?.clamp_align(min_align, max_align);

                // next comes the field's padding
                let padding_needed = variant_layout.padding_needed_for(field_layout.align());
                if padding_needed > 0 {
                    tree = tree.then(Self::padding(padding_needed));
                }

                // finally, the field's layout
                tree = tree.then(Self::from_ty(field_ty, tcx)?);

                // extend the variant layout with the field layout
                variant_layout = variant_layout.extend(field_layout).unwrap().0;
            }
            drop(fields_span);

            // finally: padding
            let padding_span = trace_span!("adding trailing padding").entered();
            if layout_summary.total_size > variant_layout.size() {
                let padding_needed = layout_summary.total_size - variant_layout.size();
                tree = tree.then(Self::padding(padding_needed));
            };
            drop(padding_span);
            drop(variant_span);
            Ok(tree)
        }

        pub fn from_discr(discr: Discr<'tcx>, tcx: TyCtxt<'tcx>, size: usize) -> Self {
            use rustc_target::abi::Endian;

            let bytes: [u8; 16];
            let bytes = match tcx.data_layout.endian {
                Endian::Little => {
                    bytes = discr.val.to_le_bytes();
                    &bytes[..size]
                }
                Endian::Big => {
                    bytes = discr.val.to_be_bytes();
                    &bytes[bytes.len() - size..]
                }
            };
            Self::Seq(bytes.iter().map(|&b| Self::from_bits(b)).collect())
        }
    }

    fn layout_of<'tcx>(
        ctx: TyCtxt<'tcx>,
        ty: Ty<'tcx>,
    ) -> Result<alloc::Layout, LayoutError<'tcx>> {
        use rustc_middle::ty::ParamEnvAnd;
        use rustc_target::abi::TyAndLayout;

        let param_env = ParamEnv::reveal_all();
        let param_env_and_type = ParamEnvAnd { param_env, value: ty };
        let TyAndLayout { layout, .. } = ctx.layout_of(param_env_and_type)?;
        let layout = alloc::Layout::from_size_align(
            layout.size().bytes_usize(),
            layout.align().abi.bytes().try_into().unwrap(),
        )
        .unwrap();
        trace!(?ty, ?layout, "computed layout for type");
        Ok(layout)
    }
}
