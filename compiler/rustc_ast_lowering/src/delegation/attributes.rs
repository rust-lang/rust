use rustc_hir::attrs::{AttributeKind, InlineAttr};
use rustc_hir::{self as hir};
use rustc_span::Span;
use rustc_span::def_id::DefId;

use crate::LoweringContext;

struct AttrAdditionInfo {
    pub equals: fn(&hir::Attribute) -> bool,
    pub kind: AttrAdditionKind,
}

enum AttrAdditionKind {
    Default { factory: fn(Span) -> hir::Attribute },
    Inherit { factory: fn(Span, &hir::Attribute) -> hir::Attribute },
}

const PARENT_ID: hir::ItemLocalId = hir::ItemLocalId::ZERO;

static ATTRS_ADDITIONS: &[AttrAdditionInfo] = &[
    AttrAdditionInfo {
        equals: |a| matches!(a, hir::Attribute::Parsed(AttributeKind::MustUse { .. })),
        kind: AttrAdditionKind::Inherit {
            factory: |span, original_attr| {
                let reason = match original_attr {
                    hir::Attribute::Parsed(AttributeKind::MustUse { reason, .. }) => *reason,
                    _ => None,
                };

                hir::Attribute::Parsed(AttributeKind::MustUse { span, reason })
            },
        },
    },
    AttrAdditionInfo {
        equals: |a| matches!(a, hir::Attribute::Parsed(AttributeKind::Inline(..))),
        kind: AttrAdditionKind::Default {
            factory: |span| hir::Attribute::Parsed(AttributeKind::Inline(InlineAttr::Hint, span)),
        },
    },
];

impl<'hir> LoweringContext<'_, 'hir> {
    pub(super) fn add_attrs_if_needed(&mut self, span: Span, sig_id: DefId) {
        let new_attrs =
            self.create_new_attrs(ATTRS_ADDITIONS, span, sig_id, self.attrs.get(&PARENT_ID));

        if new_attrs.is_empty() {
            return;
        }

        let new_arena_allocated_attrs = match self.attrs.get(&PARENT_ID) {
            Some(existing_attrs) => self.arena.alloc_from_iter(
                existing_attrs.iter().map(|a| a.clone()).chain(new_attrs.into_iter()),
            ),
            None => self.arena.alloc_from_iter(new_attrs.into_iter()),
        };

        self.attrs.insert(PARENT_ID, new_arena_allocated_attrs);
    }

    fn create_new_attrs(
        &self,
        candidate_additions: &[AttrAdditionInfo],
        span: Span,
        sig_id: DefId,
        existing_attrs: Option<&&[hir::Attribute]>,
    ) -> Vec<hir::Attribute> {
        candidate_additions
            .iter()
            .filter_map(|addition_info| {
                if let Some(existing_attrs) = existing_attrs
                    && existing_attrs
                        .iter()
                        .any(|existing_attr| (addition_info.equals)(existing_attr))
                {
                    return None;
                }

                match addition_info.kind {
                    AttrAdditionKind::Default { factory } => Some(factory(span)),
                    AttrAdditionKind::Inherit { factory, .. } =>
                    {
                        #[allow(deprecated)]
                        self.tcx
                            .get_all_attrs(sig_id)
                            .iter()
                            .find_map(|a| (addition_info.equals)(a).then(|| factory(span, a)))
                    }
                }
            })
            .collect::<Vec<_>>()
    }
}
