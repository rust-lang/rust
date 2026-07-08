use rustc_feature::AttributeStability;
use rustc_hir::AttrPath;
use rustc_session::errors::feature_err;
use rustc_span::{Span, sym};

use crate::{AttributeParser, ShouldEmit};

#[macro_export]
macro_rules! unstable {
    ($feat: ident $(, $notes:expr)*) => {
        AttributeStability::Unstable {
            gate_name: rustc_span::sym::$feat,
            gate_check: rustc_feature::Features::$feat,
            notes: &[$($notes),*],
        }
    };
}

impl<'sess> AttributeParser<'sess> {
    pub fn check_attribute_stability(
        &mut self,
        attr_path: &AttrPath,
        attr_span: Span,
        expected_stability: AttributeStability,
    ) {
        if matches!(self.should_emit, ShouldEmit::Nothing) {
            return;
        }

        let AttributeStability::Unstable { gate_check, gate_name, notes } = expected_stability
        else {
            return;
        };

        if gate_check(self.features()) || attr_span.allows_unstable(gate_name) {
            return;
        }

        let (explain, default_notes): (String, &[String]) = match gate_name {
            sym::rustc_attrs => ("use of an internal attribute".to_string(), &[
                format!("the `#[{attr_path}]` attribute is an internal implementation detail that will never be stable"
            )]),
            sym::staged_api => ("stability attributes may not be used outside of the standard library".to_string(), &[]),
            sym::custom_mir => ("the `#[custom_mir]` attribute is just used for the Rust test suite".to_string(), &[]),
            sym::allow_internal_unsafe => ("allow_internal_unsafe side-steps the unsafe_code lint".to_string(), &[]),
            sym::allow_internal_unstable => ("allow_internal_unstable side-steps feature gating and stability checks".to_string(), &[]),
            sym::compiler_builtins => ("the `#[compiler_builtins]` attribute is used to identify the `compiler_builtins` crate which contains compiler-rt intrinsics and will never be stable".to_string(), &[]),
            sym::custom_test_frameworks => ("custom test frameworks are an unstable feature".to_string(), &[]),
            sym::linkage => ("the `linkage` attribute is experimental and not portable across platforms".to_string(), &[]),
            sym::dropck_eyepatch => ("`may_dangle` has unstable semantics and may be removed in the future".to_string(), &[]),
            sym::intrinsics => ("the `#[rustc_intrinsic]` attribute is used to declare intrinsics as function items".to_string(), &[]),
            sym::lang_items => ("lang items are subject to change".to_string(), &[]),
            sym::prelude_import => ("`#[prelude_import]` is for use by rustc only".to_string(), &[]),
            sym::profiler_runtime => ("the `#[profiler_runtime]` attribute is used to identify the `profiler_builtins` crate which contains the profiler runtime and will never be stable".to_string(), &[]),
            sym::thread_local => ("`#[thread_local]` is an experimental feature, and does not currently handle destructors".to_string(), &[]),
            _ => (format!("the `#[{attr_path}]` attribute is an experimental feature"), &[]),
        };

        let mut diag = feature_err(self.sess, gate_name, attr_span, explain);

        // Remove the suggestion for `#![feature(staged_api)]` as these attributes are currently
        // not usable outside std. If we do ever expose `#[stable]` etc under a different feature
        // name then it would be unfortunate to have nightlies out there suggesting `staged_api`.
        if gate_name == sym::staged_api {
            diag.children.clear();
        }

        for note in default_notes {
            diag.note(note.clone());
        }
        for note in notes {
            diag.note(*note);
        }

        diag.emit();
    }
}
