use std::collections::HashSet;
use std::fmt::{Display, Formatter};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use termcolor::Color;

#[derive(Clone, Default)]
///CLI flags used by tidy.
pub struct TidyFlags {
    ///Applies style and formatting changes during a tidy run.
    bless: bool,
}

impl TidyFlags {
    pub fn new(cfg_args: &[String]) -> Self {
        let mut flags = Self::default();

        for arg in cfg_args {
            match arg.as_str() {
                "--bless" => flags.bless = true,
                _ => continue,
            }
        }
        flags
    }
}

/// Collects diagnostics from all tidy steps, and contains shared information
/// that determines how should message and logs be presented.
///
/// Since checks are executed in parallel, the context is internally synchronized, to avoid
/// all checks to lock it explicitly.
#[derive(Clone)]
pub struct TidyCtx {
    tidy_flags: TidyFlags,
    diag_ctx: Arc<Mutex<DiagCtxInner>>,
}

impl TidyCtx {
    pub fn new(root_path: &Path, verbose: bool, tidy_flags: TidyFlags) -> Self {
        Self {
            diag_ctx: Arc::new(Mutex::new(DiagCtxInner {
                running_checks: Default::default(),
                finished_checks: Default::default(),
                root_path: root_path.to_path_buf(),
                verbose,
            })),
            tidy_flags,
        }
    }

    pub fn is_bless_enabled(&self) -> bool {
        self.tidy_flags.bless
    }

    pub fn start_check<Id: Into<CheckId>>(&self, id: Id) -> RunningCheck {
        let mut id = id.into();

        let mut ctx = self.diag_ctx.lock().unwrap();

        // Shorten path for shorter diagnostics
        id.path = match id.path {
            Some(path) => Some(path.strip_prefix(&ctx.root_path).unwrap_or(&path).to_path_buf()),
            None => None,
        };

        ctx.start_check(id.clone());
        RunningCheck {
            id,
            bad: false,
            ctx: self.diag_ctx.clone(),
            #[cfg(test)]
            errors: vec![],
        }
    }

    pub fn into_failed_checks(self) -> Vec<FinishedCheck> {
        let ctx = Arc::into_inner(self.diag_ctx).unwrap().into_inner().unwrap();
        assert!(ctx.running_checks.is_empty(), "Some checks are still running");
        ctx.finished_checks.into_iter().filter(|c| c.bad).collect()
    }
}

struct DiagCtxInner {
    running_checks: HashSet<CheckId>,
    finished_checks: HashSet<FinishedCheck>,
    verbose: bool,
    root_path: PathBuf,
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

        if check.bad {
            output_message("FAIL", Some(&check.id), Some(COLOR_ERROR));
        } else if self.verbose {
            output_message("OK", Some(&check.id), Some(COLOR_SUCCESS));
        }

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
    pub name: String,
    pub path: Option<PathBuf>,
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

impl Display for CheckId {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)?;
        if let Some(path) = &self.path {
            write!(f, " ({})", path.display())?;
        }
        Ok(())
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct FinishedCheck {
    id: CheckId,
    bad: bool,
}

impl FinishedCheck {
    pub fn id(&self) -> &CheckId {
        &self.id
    }
}

/// Represents a single tidy check, identified by its `name`, running.
pub struct RunningCheck {
    id: CheckId,
    bad: bool,
    ctx: Arc<Mutex<DiagCtxInner>>,
    #[cfg(test)]
    errors: Vec<String>,
}

impl RunningCheck {
    /// Creates a new instance of a running check without going through the diag
    /// context.
    /// Useful if you want to run some functions from tidy without configuring
    /// diagnostics.
    pub fn new_noop() -> Self {
        let ctx = TidyCtx::new(Path::new(""), false, TidyFlags::default());
        ctx.start_check("noop")
    }

    /// Immediately output an error and mark the check as failed.
    pub fn error<T: Display>(&mut self, msg: T) {
        self.mark_as_bad();
        let msg = msg.to_string();
        output_message(&msg, Some(&self.id), Some(COLOR_ERROR));
        #[cfg(test)]
        self.errors.push(msg);
    }

    /// Immediately output a warning.
    pub fn warning<T: Display>(&mut self, msg: T) {
        output_message(&msg.to_string(), Some(&self.id), Some(COLOR_WARNING));
    }

    /// Output an informational message
    pub fn message<T: Display>(&mut self, msg: T) {
        output_message(&msg.to_string(), Some(&self.id), None);
    }

    /// Output a message only if verbose output is enabled.
    pub fn verbose_msg<T: Display>(&mut self, msg: T) {
        if self.is_verbose_enabled() {
            self.message(msg);
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

    #[cfg(test)]
    pub fn get_errors(&self) -> Vec<String> {
        self.errors.clone()
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

pub const COLOR_SUCCESS: Color = Color::Green;
pub const COLOR_ERROR: Color = Color::Red;
pub const COLOR_WARNING: Color = Color::Yellow;

/// Output a message to stderr.
/// The message can be optionally scoped to a certain check, and it can also have a certain color.
pub fn output_message(msg: &str, id: Option<&CheckId>, color: Option<Color>) {
    use termcolor::{ColorChoice, ColorSpec};

    let stderr: &mut dyn termcolor::WriteColor = if cfg!(test) {
        &mut StderrForUnitTests
    } else {
        &mut termcolor::StandardStream::stderr(ColorChoice::Auto)
    };

    if let Some(color) = &color {
        stderr.set_color(ColorSpec::new().set_fg(Some(*color))).unwrap();
    }

    match id {
        Some(id) => {
            write!(stderr, "tidy [{}", id.name).unwrap();
            if let Some(path) = &id.path {
                write!(stderr, " ({})", path.display()).unwrap();
            }
            write!(stderr, "]").unwrap();
        }
        None => {
            write!(stderr, "tidy").unwrap();
        }
    }
    if color.is_some() {
        stderr.set_color(&ColorSpec::new()).unwrap();
    }

    writeln!(stderr, ": {msg}").unwrap();
}

/// An implementation of `io::Write` and `termcolor::WriteColor` that writes
/// to stderr via `eprint!`, so that the output can be properly captured when
/// running tidy's unit tests.
struct StderrForUnitTests;

impl io::Write for StderrForUnitTests {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        eprint!("{}", String::from_utf8_lossy(buf));
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl termcolor::WriteColor for StderrForUnitTests {
    fn supports_color(&self) -> bool {
        false
    }

    fn set_color(&mut self, _spec: &termcolor::ColorSpec) -> io::Result<()> {
        Ok(())
    }

    fn reset(&mut self) -> io::Result<()> {
        Ok(())
    }
}
