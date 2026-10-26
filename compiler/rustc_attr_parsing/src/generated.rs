use std::num::NonZero;

use rustc_ast::GeneratedAttr;
use rustc_attr_ir::{
    Attribute, AttributeKind, CoverageAttrKind, DocAttribute, InlineAttr, PartialConstStability,
    StabilityLevel, UnstableReason,
};
use rustc_span::{Span, sym};
use thin_vec::ThinVec;

/// Contains the state necessary to convert generated attributes to parsed attributes.

#[derive(Default)]
pub(crate) struct GeneratedAttrState {
    state: ThinVec<AttributeKind>,
}

impl GeneratedAttrState {
    pub(crate) fn accept_generated_attr(&mut self, span: Span, generated: &GeneratedAttr) {
        let attr = match *generated {
            GeneratedAttr::AutomaticallyDerived => AttributeKind::AutomaticallyDerived,
            GeneratedAttr::RustcConstUnstableDerived => AttributeKind::RustcConstStability {
                stability: PartialConstStability {
                    feature: sym::derive_const,
                    promotable: false,
                    level: StabilityLevel::Unstable {
                        reason: UnstableReason::None,
                        implied_by: None,
                        issue: NonZero::new(143874),
                        old_name: None,
                    },
                },
                span,
            },
            GeneratedAttr::Inline => AttributeKind::Inline(InlineAttr::Hint, span),
            GeneratedAttr::DocHidden => AttributeKind::Doc(Box::new({
                let mut d = DocAttribute::default();
                d.hidden = Some(span);
                d
            })),
            GeneratedAttr::CoverageOff => AttributeKind::Coverage(CoverageAttrKind::Off),
            GeneratedAttr::RustcTestMarker(test_marker) => {
                AttributeKind::RustcTestMarker(test_marker)
            }
        };
        self.state.push(attr);
    }

    pub(crate) fn finalize_generated_attrs(self, attributes: &mut Vec<Attribute>) {
        for attr in self.state {
            attributes.push(Attribute::Parsed(attr))
        }
    }
}
