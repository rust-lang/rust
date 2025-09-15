use std::collections::HashSet;
use std::fmt::Display;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use termcolor::WriteColor;

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

    pub fn start_check<Id: Into<CheckId>>(&self, id: Id) -> RunningCheck {
        let id = id.into();

        let mut ctx = self.0.lock().unwrap();
        ctx.start_check(id.clone());
        RunningCheck { id, bad: false, ctx: self.0.clone() }
    }

    pub fn into_conclusion(self) -> bool {
        let ctx = self.0.lock().unwrap();
        assert!(ctx.running_checks.is_empty(), "Some checks are still running");
        ctx.finished_checks.iter().any(|c| c.bad)
    }
}

struct DiagCtxInner {
    running_checks: HashSet<CheckId>,
    finished_checks: HashSet<FinishedCheck>,
    verbose: bool,
}

impl DiagCtxInner {
    fn start_check(&mut self, id: CheckId) {
        if self.has_check_id(&id) {
            panic!("Starting a check named `{id:?}` for the second time");
        }
        self.running_checks.insert(id);
    }

    fn finish_check(&mut self, check: FinishedCheck) {
        assert!(
            self.running_checks.remove(&check.id),
            "Finishing check `{:?}` that was not started",
            check.id
        );
        self.finished_checks.insert(check);
    }

    fn has_check_id(&self, id: &CheckId) -> bool {
        self.running_checks
            .iter()
            .chain(self.finished_checks.iter().map(|c| &c.id))
            .any(|c| c == id)
    }
}

/// Identifies a single step
#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct CheckId {
    name: String,
    path: Option<PathBuf>,
}

impl CheckId {
    pub fn new(name: &'static str) -> Self {
        Self { name: name.to_string(), path: None }
    }

    pub fn path(self, path: &Path) -> Self {
        Self { path: Some(path.to_path_buf()), ..self }
    }
}

impl From<&'static str> for CheckId {
    fn from(name: &'static str) -> Self {
        Self::new(name)
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
struct FinishedCheck {
    id: CheckId,
    bad: bool,
}

/// Represents a single tidy check, identified by its `name`, running.
pub struct RunningCheck {
    id: CheckId,
    bad: bool,
    ctx: Arc<Mutex<DiagCtxInner>>,
}

impl RunningCheck {
    /// Immediately output an error and mark the check as failed.
    pub fn error<T: Display>(&mut self, t: T) {
        self.mark_as_bad();
        tidy_error(&t.to_string()).expect("failed to output error");
    }

    /// Immediately output a warning.
    pub fn warning<T: Display>(&mut self, t: T) {
        eprintln!("WARNING: {t}");
    }

    /// Output an informational message
    pub fn message<T: Display>(&mut self, t: T) {
        eprintln!("{t}");
    }

    /// Output a message only if verbose output is enabled.
    pub fn verbose_msg<T: Display>(&mut self, t: T) {
        if self.is_verbose_enabled() {
            self.message(t);
        }
    }

    /// Has an error already occured for this check?
    pub fn is_bad(&self) -> bool {
        self.bad
    }

    /// Is verbose output enabled?
    pub fn is_verbose_enabled(&self) -> bool {
        self.ctx.lock().unwrap().verbose
    }

    fn mark_as_bad(&mut self) {
        self.bad = true;
    }
}

impl Drop for RunningCheck {
    fn drop(&mut self) {
        self.ctx.lock().unwrap().finish_check(FinishedCheck { id: self.id.clone(), bad: self.bad })
    }
}

fn tidy_error(args: &str) -> std::io::Result<()> {
    use std::io::Write;

    use termcolor::{Color, ColorChoice, ColorSpec, StandardStream};

    let mut stderr = StandardStream::stdout(ColorChoice::Auto);
    stderr.set_color(ColorSpec::new().set_fg(Some(Color::Red)))?;

    write!(&mut stderr, "tidy error")?;
    stderr.set_color(&ColorSpec::new())?;

    writeln!(&mut stderr, ": {args}")?;
    Ok(())
}
