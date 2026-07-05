use rustc_hir::attrs::{AttributeKind, InlineAttr};
use rustc_hir::{self as hir};
use rustc_span::Span;
use rustc_span::def_id::DefId;

use crate::LoweringContext;
use crate::delegation::DelegationResolution;

struct AdditionInfo {
    pub equals: fn(&hir::Attribute) -> bool,
    pub kind: AdditionKind,
}

enum AdditionKind {
    Default { factory: fn(Span) -> hir::Attribute },
    Inherit { factory: fn(Span, &hir::Attribute) -> hir::Attribute },
}

static ADDITIONS: &[AdditionInfo] = &[
    AdditionInfo {
        equals: |a| matches!(a, hir::Attribute::Parsed(AttributeKind::MustUse { .. })),
        kind: AdditionKind::Inherit {
            factory: |span, original_attr| {
                let reason = match original_attr {
                    hir::Attribute::Parsed(AttributeKind::MustUse { reason, .. }) => *reason,
                    _ => None,
                };

                hir::Attribute::Parsed(AttributeKind::MustUse { span, reason })
            },
        },
    },
    AdditionInfo {
        equals: |a| matches!(a, hir::Attribute::Parsed(AttributeKind::Inline(..))),
        kind: AdditionKind::Default {
            factory: |span| hir::Attribute::Parsed(AttributeKind::Inline(InlineAttr::Hint, span)),
        },
    },
];

impl<'hir> LoweringContext<'_, 'hir> {
    pub(super) fn add_attrs_if_needed(&mut self, resolution: &DelegationResolution) {
        let &DelegationResolution { span, sig_id, .. } = resolution;

        const PARENT_ID: hir::ItemLocalId = hir::ItemLocalId::ZERO;
        let new_attrs = self.create_new_attrs(span, sig_id, self.attrs.get(&PARENT_ID));

        if !new_attrs.is_empty() {
            let new_attrs = match self.attrs.get(&PARENT_ID) {
                Some(existing_attrs) => self.arena.alloc_from_iter(
                    existing_attrs.iter().map(|a| a.clone()).chain(new_attrs.into_iter()),
                ),
                None => self.arena.alloc_from_iter(new_attrs.into_iter()),
            };

            self.attrs.insert(PARENT_ID, new_attrs);
        }
    }

    fn create_new_attrs(
        &self,
        span: Span,
        sig_id: DefId,
        existing: Option<&&[hir::Attribute]>,
    ) -> Vec<hir::Attribute> {
        ADDITIONS
            .iter()
            .filter_map(|addition| {
                existing
                    .is_none_or(|attrs| !attrs.iter().any(|a| (addition.equals)(a)))
                    .then(|| match addition.kind {
                        AdditionKind::Default { factory } => Some(factory(span)),
                        AdditionKind::Inherit { factory, .. } =>
                        {
                            #[allow(deprecated)]
                            self.tcx
                                .get_all_attrs(sig_id)
                                .iter()
                                .find_map(|a| (addition.equals)(a).then(|| factory(span, a)))
                        }
                    })
                    .flatten()
            })
            .collect::<Vec<_>>()
    }
}
