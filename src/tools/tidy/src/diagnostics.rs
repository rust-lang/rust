use std::collections::{HashMap, HashSet};
use std::fmt::{Display, Formatter};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use build_helper::git::{get_git_untracked_files, output_result};
use termcolor::{Color, WriteColor};

#[derive(Clone, Default, Debug)]
pub struct TidyFlags {
    pub bless: bool,
    pub pre_push: bool,
    pub include_untracked: bool,
    pub pre_push_content: HashMap<PathBuf, String>,
    pub untracked_files: HashSet<PathBuf>,
}

impl TidyFlags {
    pub fn new(root_path: &Path, cfg_args: &[String]) -> Self {
        let mut flags = Self::default();

        for arg in cfg_args {
            match arg.as_str() {
                "--bless" => flags.bless = true,
                "--include-untracked" => flags.include_untracked = true,
                "--pre-push" => flags.pre_push = true,
                _ => continue,
            }
        }

        //Get a map of file names and file content from last commit for `--pre-push`.
        let pre_push_content = match flags.pre_push {
            true => get_git_last_commit_content(root_path),
            false => HashMap::new(),
        };

        //Get all of the untracked files, used by default to exclude untracked from tidy.
        let untracked_files = match get_git_untracked_files(Some(root_path)) {
            Ok(Some(untracked_paths)) => {
                untracked_paths.into_iter().map(|s| PathBuf::from(root_path).join(s)).collect()
            }
            _ => HashSet::new(),
        };

        flags.pre_push_content = pre_push_content;
        flags.untracked_files = untracked_files;

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
    diag_ctx: Arc<Mutex<DiagCtx>>,
    pub tidy_flags: TidyFlags,
}

impl TidyCtx {
    pub fn new(root_path: &Path, tidy_flags: TidyFlags, verbose: bool) -> Self {
        Self {
            diag_ctx: Arc::new(Mutex::new(DiagCtx {
                running_checks: Default::default(),
                finished_checks: Default::default(),
                root_path: root_path.to_path_buf(),
                verbose,
            })),
            tidy_flags,
        }
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

struct DiagCtx {
    running_checks: HashSet<CheckId>,
    finished_checks: HashSet<FinishedCheck>,
    verbose: bool,
    root_path: PathBuf,
}

impl DiagCtx {
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
    ctx: Arc<Mutex<DiagCtx>>,
    #[cfg(test)]
    errors: Vec<String>,
}

impl RunningCheck {
    /// Creates a new instance of a running check without going through the diag
    /// context.
    /// Useful if you want to run some functions from tidy without configuring
    /// diagnostics.
    pub fn new_noop() -> Self {
        let ctx = TidyCtx::new(Path::new(""), TidyFlags::default(), false);
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
    use std::io::Write;

    use termcolor::{ColorChoice, ColorSpec, StandardStream};

    let mut stderr = StandardStream::stderr(ColorChoice::Auto);
    if let Some(color) = &color {
        stderr.set_color(ColorSpec::new().set_fg(Some(*color))).unwrap();
    }

    match id {
        Some(id) => {
            write!(&mut stderr, "tidy [{}", id.name).unwrap();
            if let Some(path) = &id.path {
                write!(&mut stderr, " ({})", path.display()).unwrap();
            }
            write!(&mut stderr, "]").unwrap();
        }
        None => {
            write!(&mut stderr, "tidy").unwrap();
        }
    }
    if color.is_some() {
        stderr.set_color(&ColorSpec::new()).unwrap();
    }

    writeln!(&mut stderr, ": {msg}").unwrap();
}

fn get_git_last_commit_content(git_root: &Path) -> HashMap<PathBuf, String> {
    let mut content_map = HashMap::new();
    // Get all of the file names that have been modified in the working dir.
    let file_names =
        t!(output_result(std::process::Command::new("git").args(["diff", "--name-only", "HEAD"])))
            .lines()
            .map(|s| s.trim().to_owned())
            .collect::<Vec<String>>();
    for file in file_names {
        let content = t!(output_result(
            // Get the content of the files from the last commit. Used for '--pre-push' tidy flag.
            std::process::Command::new("git").arg("show").arg(format!("HEAD:{}", &file))
        ));
        content_map.insert(PathBuf::from(&git_root).join(file), content);
    }
    content_map
}
