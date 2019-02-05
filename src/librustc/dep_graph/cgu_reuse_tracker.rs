//! Some facilities for tracking how codegen-units are reused during incremental
//! compilation. This is used for incremental compilation tests and debug
//! output.

use crate::session::Session;
use rustc_data_structures::fx::FxHashMap;
use std::sync::{Arc, Mutex};
use syntax_pos::Span;

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub enum CguReuse {
    No,
    PreLto,
    PostLto,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ComparisonKind {
    Exact,
    AtLeast,
}

struct TrackerData {
    actual_reuse: FxHashMap<String, CguReuse>,
    expected_reuse: FxHashMap<String, (String, SendSpan, CguReuse, ComparisonKind)>,
}

// Span does not implement `Send`, so we can't just store it in the shared
// `TrackerData` object. Instead of splitting up `TrackerData` into shared and
// non-shared parts (which would be complicated), we just mark the `Span` here
// explicitly as `Send`. That's safe because the span data here is only ever
// accessed from the main thread.
struct SendSpan(Span);
unsafe impl Send for SendSpan {}

#[derive(Clone)]
pub struct CguReuseTracker {
    data: Option<Arc<Mutex<TrackerData>>>,
}

impl CguReuseTracker {
    pub fn new() -> CguReuseTracker {
        let data = TrackerData {
            actual_reuse: Default::default(),
            expected_reuse: Default::default(),
        };

        CguReuseTracker {
            data: Some(Arc::new(Mutex::new(data))),
        }
    }

    pub fn new_disabled() -> CguReuseTracker {
        CguReuseTracker {
            data: None,
        }
    }

    pub fn set_actual_reuse(&self, cgu_name: &str, kind: CguReuse) {
        if let Some(ref data) = self.data {
            debug!("set_actual_reuse({:?}, {:?})", cgu_name, kind);

            let prev_reuse = data.lock()
                                 .unwrap()
                                 .actual_reuse
                                 .insert(cgu_name.to_string(), kind);

            if let Some(prev_reuse) = prev_reuse {
                // The only time it is legal to overwrite reuse state is when
                // we discover during ThinLTO that we can actually reuse the
                // post-LTO version of a CGU.
                assert_eq!(prev_reuse, CguReuse::PreLto);
            }
        }
    }

    pub fn set_expectation(&self,
                           cgu_name: &str,
                           cgu_user_name: &str,
                           error_span: Span,
                           expected_reuse: CguReuse,
                           comparison_kind: ComparisonKind) {
        if let Some(ref data) = self.data {
            debug!("set_expectation({:?}, {:?}, {:?})", cgu_name,
                                                        expected_reuse,
                                                        comparison_kind);
            let mut data = data.lock().unwrap();

            data.expected_reuse.insert(cgu_name.to_string(),
                                       (cgu_user_name.to_string(),
                                        SendSpan(error_span),
                                        expected_reuse,
                                        comparison_kind));
        }
    }

    pub fn check_expected_reuse(&self, sess: &Session) {
        if let Some(ref data) = self.data {
            let data = data.lock().unwrap();

            for (cgu_name, &(ref cgu_user_name,
                             ref error_span,
                             expected_reuse,
                             comparison_kind)) in &data.expected_reuse {
                if let Some(&actual_reuse) = data.actual_reuse.get(cgu_name) {
                    let (error, at_least) = match comparison_kind {
                        ComparisonKind::Exact => {
                            (expected_reuse != actual_reuse, false)
                        }
                        ComparisonKind::AtLeast => {
                            (actual_reuse < expected_reuse, true)
                        }
                    };

                    if error {
                        let at_least = if at_least { "at least " } else { "" };
                        let msg = format!("CGU-reuse for `{}` is `{:?}` but \
                                           should be {}`{:?}`",
                                          cgu_user_name,
                                          actual_reuse,
                                          at_least,
                                          expected_reuse);
                        sess.span_err(error_span.0, &msg);
                    }
                } else {
                    let msg = format!("CGU-reuse for `{}` (mangled: `{}`) was \
                                       not recorded",
                                       cgu_user_name,
                                       cgu_name);
                    sess.span_fatal(error_span.0, &msg);
                }
            }
        }
    }
}
