use rustc_feature::{AttributeGate, GateKind, GatedAttribute};
use rustc_session::parse::feature_err;
use rustc_span::Span;

use crate::context::Stage;
use crate::{AttributeParser, ShouldEmit};

impl<'sess, S: Stage> AttributeParser<'sess, S> {
    pub fn check_attribute_gate(
        &mut self,
        gate: AttributeGate,
        span: Span,
    ) -> Option<rustc_feature::GateKind> {
        if matches!(self.stage.should_emit(), ShouldEmit::Nothing) {
            return None;
        }
        let AttributeGate::Gated {
            gated_attr: GatedAttribute { feature, message, check, notes },
            kind,
        } = gate
        else {
            return None;
        };
        if !matches!(kind, GateKind::Ignore)
            && !check(self.features())
            && !span.allows_unstable(feature)
        {
            #[allow(unused_mut)]
            let mut diag = feature_err(self.sess, feature, span, message);
            for note in notes {
                diag.note(*note);
            }
            diag.emit();
        }
        Some(kind)
    }
}
