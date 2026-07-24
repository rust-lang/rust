use rustc_ast::SyntheticAttr;
use rustc_ast::attr::data_structures::CfgEntry;
use rustc_hir::Attribute;
use rustc_hir::attrs::AttributeKind;
use rustc_span::Span;
use thin_vec::ThinVec;

/// This struct contains the state necessary to convert synthetic attributes to hir attributes
/// The only conversion that really happens here is that multiple synthetic attributes are
/// merged into a single hir attribute, representing their combined state.
/// FIXME: We should make this a nice and extendable system if this is going to be used more often
#[derive(Default)]
pub(crate) struct SyntheticAttrState {
    /// Attribute state for `SyntheticAttr::CfgTrace` attributes.
    cfg_trace: ThinVec<(CfgEntry, Span)>,

    /// Attribute state for `SyntheticAttr::CfgAttrTrace` attributes.
    cfg_attr_trace: ThinVec<(CfgEntry, Span)>,
}

impl SyntheticAttrState {
    pub(crate) fn accept_synthetic_attr(
        &mut self,
        attr_span: Span,
        lower_span: impl Copy + Fn(Span) -> Span,
        synthetic: &SyntheticAttr,
    ) {
        match synthetic {
            SyntheticAttr::CfgTrace(cfg) => {
                let mut cfg = cfg.clone();
                cfg.lower_spans(lower_span);
                self.cfg_trace.push((cfg, attr_span));
            }
            SyntheticAttr::CfgAttrTrace(cfg) => {
                let mut cfg = cfg.clone();
                cfg.lower_spans(lower_span);
                self.cfg_attr_trace.push((cfg, attr_span));
            }
        }
    }

    pub(crate) fn finalize_synthetic_attrs(self, attributes: &mut Vec<Attribute>) {
        if !self.cfg_trace.is_empty() {
            attributes.push(Attribute::Parsed(AttributeKind::CfgTrace(self.cfg_trace)));
        }
        if !self.cfg_attr_trace.is_empty() {
            attributes.push(Attribute::Parsed(AttributeKind::CfgAttrTrace(self.cfg_attr_trace)));
        }
    }
}
