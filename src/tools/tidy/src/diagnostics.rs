use std::collections::HashSet;
use std::fmt::Display;
use std::sync::{Arc, Mutex};

use crate::tidy_error;

/// Collects diagnostics from all tidy steps, and contains shared information
/// that determines how should message and logs be presented.
///
/// Since checks are executed in parallel, the context is internally synchronized, to avoid
/// all checks to lock it explicitly.
#[derive(Clone)]
pub struct DiagCtx(Arc<Mutex<DiagCtxInner>>);

impl DiagCtx {
    pub fn new(verbose: bool) -> Self {
        Self(Arc::new(Mutex::new(DiagCtxInner {
            running_checks: Default::default(),
            finished_checks: Default::default(),
            verbose,
        })))
    }

    pub fn start_check<T: Display>(&self, name: T) -> RunningCheck {
        let name = name.to_string();

        let mut ctx = self.0.lock().unwrap();
        ctx.start_check(&name);
        RunningCheck { name, bad: false, ctx: self.0.clone() }
    }

    pub fn into_conclusion(self) -> bool {
        let ctx = self.0.lock().unwrap();
        assert!(ctx.running_checks.is_empty(), "Some checks are still running");
        ctx.finished_checks.iter().any(|c| c.bad)
    }
}

struct DiagCtxInner {
    running_checks: HashSet<String>,
    finished_checks: HashSet<FinishedCheck>,
    verbose: bool,
}

impl DiagCtxInner {
    fn start_check(&mut self, name: &str) {
        if self.has_check(name) {
            panic!("Starting a check named {name} for the second time");
        }
        self.running_checks.insert(name.to_string());
    }

    fn finish_check(&mut self, check: FinishedCheck) {
        assert!(
            self.running_checks.remove(&check.name),
            "Finishing check {} that was not started",
            check.name
        );
        self.finished_checks.insert(check);
    }

    fn has_check(&self, name: &str) -> bool {
        self.running_checks
            .iter()
            .chain(self.finished_checks.iter().map(|c| &c.name))
            .any(|c| c == name)
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
struct FinishedCheck {
    name: String,
    bad: bool,
}

/// Represents a single tidy check, identified by its `name`, running.
pub struct RunningCheck {
    name: String,
    bad: bool,
    ctx: Arc<Mutex<DiagCtxInner>>,
}

impl RunningCheck {
    /// Immediately output an error and mark the check as failed.
    pub fn error<T: Display>(&mut self, t: T) {
        self.mark_as_bad();
        tidy_error(&t.to_string()).expect("failed to output error");
    }

    fn mark_as_bad(&mut self) {
        self.bad = true;
    }
}

impl Drop for RunningCheck {
    fn drop(&mut self) {
        self.ctx
            .lock()
            .unwrap()
            .finish_check(FinishedCheck { name: self.name.clone(), bad: self.bad })
    }
}
