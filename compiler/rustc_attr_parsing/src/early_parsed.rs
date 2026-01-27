use rustc_ast::EarlyParsedAttribute;
use rustc_ast::attr::data_structures::CfgEntry;
use rustc_hir::Attribute;
use rustc_hir::attrs::AttributeKind;
use rustc_span::{Span, Symbol, sym};
use thin_vec::ThinVec;

pub(crate) const EARLY_PARSED_ATTRIBUTES: &[&[Symbol]] =
    &[&[sym::cfg_trace], &[sym::cfg_attr_trace]];

/// This struct contains the state necessary to convert early parsed attributes to hir attributes
/// The only conversion that really happens here is that multiple early parsed attributes are
/// merged into a single hir attribute, representing their combined state.
/// FIXME: We should make this a nice and extendable system if this is going to be used more often
#[derive(Default)]
pub(crate) struct EarlyParsedState {
    /// Attribute state for `#[cfg]` trace attributes
    cfg_trace: ThinVec<(CfgEntry, Span)>,

    /// Attribute state for `#[cfg_attr]` trace attributes
    /// The arguments of these attributes is no longer relevant for any later passes, only their presence.
    /// So we discard the arguments here.
    cfg_attr_trace: bool,
}

impl EarlyParsedState {
    pub(crate) fn accept_early_parsed_attribute(
        &mut self,
        attr_span: Span,
        lower_span: impl Copy + Fn(Span) -> Span,
        parsed: &EarlyParsedAttribute,
    ) {
        match parsed {
            EarlyParsedAttribute::CfgTrace(cfg) => {
                let mut cfg = cfg.clone();
                cfg.lower_spans(lower_span);
                self.cfg_trace.push((cfg, attr_span));
            }
            EarlyParsedAttribute::CfgAttrTrace => {
                self.cfg_attr_trace = true;
            }
        }
    }

    pub(crate) fn finalize_early_parsed_attributes(self, attributes: &mut Vec<Attribute>) {
        if !self.cfg_trace.is_empty() {
            attributes.push(Attribute::Parsed(AttributeKind::CfgTrace(self.cfg_trace)));
        }
        if self.cfg_attr_trace {
            attributes.push(Attribute::Parsed(AttributeKind::CfgAttrTrace));
        }
    }
}
