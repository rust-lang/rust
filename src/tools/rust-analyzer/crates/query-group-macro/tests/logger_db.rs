use std::sync::{Arc, Mutex};

#[salsa_macros::db]
#[derive(Clone)]
pub(crate) struct LoggerDb {
    storage: salsa::Storage<Self>,
    logger: Logger,
}

impl Default for LoggerDb {
    fn default() -> Self {
        let logger = Logger::default();
        Self {
            storage: salsa::Storage::new(Some(Box::new({
                let logger = logger.clone();
                move |event| match event.kind {
                    salsa::EventKind::WillExecute { .. }
                    | salsa::EventKind::WillCheckCancellation
                    | salsa::EventKind::DidValidateMemoizedValue { .. }
                    | salsa::EventKind::WillDiscardStaleOutput { .. }
                    | salsa::EventKind::DidDiscard { .. } => {
                        logger.logs.lock().unwrap().push(format!("salsa_event({:?})", event.kind));
                    }
                    _ => {}
                }
            }))),
            logger,
        }
    }
}

#[derive(Default, Clone)]
struct Logger {
    logs: Arc<Mutex<Vec<String>>>,
}

#[salsa_macros::db]
impl salsa::Database for LoggerDb {}

impl LoggerDb {
    /// Log an event from inside a tracked function.
    pub(crate) fn push_log(&self, string: String) {
        self.logger.logs.lock().unwrap().push(string);
    }

    /// Asserts what the (formatted) logs should look like,
    /// clearing the logged events. This takes `&mut self` because
    /// it is meant to be run from outside any tracked functions.
    pub(crate) fn assert_logs(&self, expected: expect_test::Expect) {
        let logs = std::mem::take(&mut *self.logger.logs.lock().unwrap());
        expected.assert_eq(&format!("{logs:#?}"));
    }
}

/// Test the logger database.
///
/// This test isn't very interesting, but it *does* remove a dead code warning.
#[test]
fn test_logger_db() {
    let db = LoggerDb::default();
    db.push_log("test".to_string());
    db.assert_logs(expect_test::expect![
        r#"
        [
            "test",
        ]"#
    ]);
}
