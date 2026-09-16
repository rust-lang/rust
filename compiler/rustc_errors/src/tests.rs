use std::sync::Arc;

use rustc_data_structures::sync::Lock;
use rustc_span::{BytePos, Span, create_default_session_globals_then};

use crate::diagnostic::DiagInner;
use crate::emitter::Emitter;
use crate::{DiagCtxt, Level, StashKey};

/// An emitter that records the message of each emitted diagnostic, in emission
/// order, so that tests can assert on the *order* in which diagnostics come out.
struct RecordingEmitter {
    messages: Arc<Lock<Vec<String>>>,
}

impl Emitter for RecordingEmitter {
    fn source_map(&self) -> Option<&rustc_span::source_map::SourceMap> {
        None
    }

    fn emit_diagnostic(&mut self, diag: DiagInner) {
        let message = diag
            .messages
            .into_iter()
            .map(|(msg, _)| msg.as_str().unwrap_or("").to_string())
            .collect::<Vec<_>>()
            .join(" + ");
        self.messages.lock().push(message);
    }
}

/// Stealing a stashed diagnostic must not reorder the diagnostics that remain
/// stashed.
///
/// `stashed_diagnostics` is an order-preserving map, and the remaining entries
/// are emitted in insertion order by `emit_stashed_diagnostics`. The
/// stash-stealing helpers used `swap_remove`, which moves the last entry into
/// the slot of the removed entry and therefore scrambles that order whenever
/// something other than the last entry is stolen. See the `FIXME(#120456)`
/// comments on the stealing helpers.
#[test]
fn stashed_diagnostics_keep_order_after_steal() {
    create_default_session_globals_then(|| {
        let messages = Arc::new(Lock::new(Vec::new()));
        let dcx = DiagCtxt::new(Box::new(RecordingEmitter { messages: messages.clone() }));
        let handle = dcx.handle();

        let key = StashKey::ItemNoType;
        let span_a = Span::with_root_ctxt(BytePos(0), BytePos(1));
        let span_b = Span::with_root_ctxt(BytePos(2), BytePos(3));
        let span_c = Span::with_root_ctxt(BytePos(4), BytePos(5));

        handle.stash_diagnostic(span_a, key, DiagInner::new(Level::Warning, "A"));
        handle.stash_diagnostic(span_b, key, DiagInner::new(Level::Warning, "B"));
        handle.stash_diagnostic(span_c, key, DiagInner::new(Level::Warning, "C"));

        // Steal the *first* stashed diagnostic. With `swap_remove` this moves
        // `C` into `A`'s slot, so the remaining diagnostics come out as
        // `C, B` instead of `B, C`.
        handle.steal_non_err(span_a, key).expect("A was stashed").emit();

        handle.emit_stashed_diagnostics();
        drop(dcx);

        assert_eq!(messages.lock().clone(), vec!["A".to_string(), "B".to_string(), "C".to_string()]);
    });
}
