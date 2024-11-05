use rustc_hir::AttributeKind;
use rustc_span::{Symbol, sym};
use thin_vec::ThinVec;

use super::{AttributeFilter, AttributeGroup, AttributeMapping};
use crate::attribute_filter;
use crate::context::AttributeGroupContext;
use crate::parser::ArgParser;

// TODO: turn into CombineGroup?
#[derive(Default)]
pub(crate) struct ConfusablesGroup {
    confusables: ThinVec<Symbol>,
}

impl AttributeGroup for ConfusablesGroup {
    const ATTRIBUTES: AttributeMapping<Self> = &[(&[sym::rustc_confusables], |this, _cx, args| {
        let Some(list) = args.list() else {
            // TODO: error when not a list? Bring validation code here.
            //       NOTE: currently subsequent attributes are silently ignored using
            //       tcx.get_attr().
            return;
        };

        for param in list.mixed() {
            let Some(lit) = param.lit() else {
                // TODO: error when not a lit? Bring validation code here.
                //       curently silently ignored.
                return;
            };

            this.confusables.push(lit.symbol);
        }
    })];

    fn finalize(self, _cx: &AttributeGroupContext<'_>) -> Option<(AttributeKind, AttributeFilter)> {
        Some((AttributeKind::Confusables(self.confusables), attribute_filter!(allow all)))
    }
}
