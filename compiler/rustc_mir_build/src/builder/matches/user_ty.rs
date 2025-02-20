use std::iter;

use rustc_abi::{FieldIdx, VariantIdx};
use rustc_middle::mir::{ProjectionElem, UserTypeProjection, UserTypeProjections};
use rustc_middle::ty::{AdtDef, UserTypeAnnotationIndex};
use rustc_span::Symbol;

/// One of a list of "operations" that can be used to lazily build projections
/// of user-specified types.
#[derive(Clone, Debug)]
pub(crate) enum ProjectedUserTypesOp {
    PushUserType { base: UserTypeAnnotationIndex },

    Index,
    Subslice { from: u64, to: u64 },
    Deref,
    Leaf { field: FieldIdx },
    Variant { name: Symbol, variant: VariantIdx, field: FieldIdx },
}

#[derive(Debug)]
pub(crate) enum ProjectedUserTypesNode<'a> {
    None,
    Chain { parent: &'a Self, op: ProjectedUserTypesOp },
}

impl<'a> ProjectedUserTypesNode<'a> {
    pub(crate) fn push_user_type(&'a self, base: UserTypeAnnotationIndex) -> Self {
        // Only `PushUserType` ops can cause an empty chain to become non-empty.
        Self::Chain { parent: self, op: ProjectedUserTypesOp::PushUserType { base } }
    }

    /// Push another projection op onto the chain, but only if it is already non-empty.
    fn maybe_push(&'a self, op_fn: impl FnOnce() -> ProjectedUserTypesOp) -> Self {
        match self {
            Self::None => Self::None,
            Self::Chain { .. } => Self::Chain { parent: self, op: op_fn() },
        }
    }

    pub(crate) fn index(&'a self) -> Self {
        self.maybe_push(|| ProjectedUserTypesOp::Index)
    }

    pub(crate) fn subslice(&'a self, from: u64, to: u64) -> Self {
        self.maybe_push(|| ProjectedUserTypesOp::Subslice { from, to })
    }

    pub(crate) fn deref(&'a self) -> Self {
        self.maybe_push(|| ProjectedUserTypesOp::Deref)
    }

    pub(crate) fn leaf(&'a self, field: FieldIdx) -> Self {
        self.maybe_push(|| ProjectedUserTypesOp::Leaf { field })
    }

    pub(crate) fn variant(
        &'a self,
        adt_def: AdtDef<'_>,
        variant: VariantIdx,
        field: FieldIdx,
    ) -> Self {
        self.maybe_push(|| {
            let name = adt_def.variant(variant).name;
            ProjectedUserTypesOp::Variant { name, variant, field }
        })
    }

    /// Traverses the chain of nodes to yield each op in the chain.
    /// Because this walks from child node to parent node, the ops are
    /// naturally yielded in "reverse" order.
    fn iter_ops_reversed(&'a self) -> impl Iterator<Item = &'a ProjectedUserTypesOp> {
        let mut next = self;
        iter::from_fn(move || match next {
            Self::None => None,
            Self::Chain { parent, op } => {
                next = parent;
                Some(op)
            }
        })
    }

    pub(crate) fn build_user_type_projections(&self) -> UserTypeProjections {
        let ops_reversed = self.iter_ops_reversed().cloned().collect::<Vec<_>>();

        let mut projections = vec![];
        for op in ops_reversed.into_iter().rev() {
            match op {
                ProjectedUserTypesOp::PushUserType { base } => {
                    projections.push(UserTypeProjection { base, projs: vec![] })
                }

                ProjectedUserTypesOp::Index => {
                    for p in &mut projections {
                        p.projs.push(ProjectionElem::Index(()))
                    }
                }
                ProjectedUserTypesOp::Subslice { from, to } => {
                    for p in &mut projections {
                        p.projs.push(ProjectionElem::Subslice { from, to, from_end: true })
                    }
                }
                ProjectedUserTypesOp::Deref => {
                    for p in &mut projections {
                        p.projs.push(ProjectionElem::Deref)
                    }
                }
                ProjectedUserTypesOp::Leaf { field } => {
                    for p in &mut projections {
                        p.projs.push(ProjectionElem::Field(field, ()))
                    }
                }
                ProjectedUserTypesOp::Variant { name, variant, field } => {
                    for p in &mut projections {
                        p.projs.push(ProjectionElem::Downcast(Some(name), variant));
                        p.projs.push(ProjectionElem::Field(field, ()));
                    }
                }
            }
        }

        UserTypeProjections { contents: projections }
    }
}
