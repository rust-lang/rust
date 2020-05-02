//! A bad shell -- small cross platform module for writing glue code

use std::{
    cell::RefCell,
    env,
    ffi::OsString,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::{bail, Context, Result};

pub mod fs2 {
    use std::{fs, path::Path};

    use anyhow::{Context, Result};

    pub fn read_dir<P: AsRef<Path>>(path: P) -> Result<fs::ReadDir> {
        let path = path.as_ref();
        fs::read_dir(path).with_context(|| format!("Failed to read {}", path.display()))
    }

    pub fn read_to_string<P: AsRef<Path>>(path: P) -> Result<String> {
        let path = path.as_ref();
        fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))
    }

    pub fn write<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> Result<()> {
        let path = path.as_ref();
        fs::write(path, contents).with_context(|| format!("Failed to write {}", path.display()))
    }

    pub fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> Result<u64> {
        let from = from.as_ref();
        let to = to.as_ref();
        fs::copy(from, to)
            .with_context(|| format!("Failed to copy {} to {}", from.display(), to.display()))
    }

    pub fn remove_file<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        fs::remove_file(path).with_context(|| format!("Failed to remove file {}", path.display()))
    }

    pub fn remove_dir_all<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        fs::remove_dir_all(path).with_context(|| format!("Failed to remove dir {}", path.display()))
    }

    pub fn create_dir_all<P: AsRef<Path>>(path: P) -> Result<()> {
        let path = path.as_ref();
        fs::create_dir_all(path).with_context(|| format!("Failed to create dir {}", path.display()))
    }
}

macro_rules! _run {
    ($($expr:expr),*) => {
        run!($($expr),*; echo = true)
    };
    ($($expr:expr),* ; echo = $echo:expr) => {
        $crate::not_bash::run_process(format!($($expr),*), $echo, None)
    };
    ($($expr:expr),* ;  <$stdin:expr) => {
        $crate::not_bash::run_process(format!($($expr),*), false, Some($stdin))
    };
}
pub(crate) use _run as run;

pub struct Pushd {
    _p: (),
}

pub fn pushd(path: impl Into<PathBuf>) -> Pushd {
    Env::with(|env| env.pushd(path.into()));
    Pushd { _p: () }
}

impl Drop for Pushd {
    fn drop(&mut self) {
        Env::with(|env| env.popd())
    }
}

pub struct Pushenv {
    _p: (),
}

pub fn pushenv(var: &str, value: &str) -> Pushenv {
    Env::with(|env| env.pushenv(var.into(), value.into()));
    Pushenv { _p: () }
}

impl Drop for Pushenv {
    fn drop(&mut self) {
        Env::with(|env| env.popenv())
    }
}

pub fn rm_rf(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(());
    }
    if path.is_file() {
        fs2::remove_file(path)
    } else {
        fs2::remove_dir_all(path)
    }
}

#[doc(hidden)]
pub fn run_process(cmd: String, echo: bool, stdin: Option<&[u8]>) -> Result<String> {
    run_process_inner(&cmd, echo, stdin).with_context(|| format!("process `{}` failed", cmd))
}

pub fn date_iso() -> Result<String> {
    run!("date --iso --utc")
}

fn run_process_inner(cmd: &str, echo: bool, stdin: Option<&[u8]>) -> Result<String> {
    let mut args = shelx(cmd);
    let binary = args.remove(0);
    let current_dir = Env::with(|it| it.cwd().to_path_buf());

    if echo {
        println!("> {}", cmd)
    }

    let mut command = Command::new(binary);
    command.args(args).current_dir(current_dir).stderr(Stdio::inherit());
    let output = match stdin {
        None => command.stdin(Stdio::null()).output(),
        Some(stdin) => {
            command.stdin(Stdio::piped()).stdout(Stdio::piped());
            let mut process = command.spawn()?;
            process.stdin.take().unwrap().write_all(stdin)?;
            process.wait_with_output()
        }
    }?;
    let stdout = String::from_utf8(output.stdout)?;

    if echo {
        print!("{}", stdout)
    }

    if !output.status.success() {
        bail!("{}", output.status)
    }

    Ok(stdout.trim().to_string())
}

// FIXME: some real shell lexing here
fn shelx(cmd: &str) -> Vec<String> {
    cmd.split_whitespace().map(|it| it.to_string()).collect()
}

struct Env {
    pushd_stack: Vec<PathBuf>,
    pushenv_stack: Vec<(OsString, Option<OsString>)>,
}

impl Env {
    fn with<F: FnOnce(&mut Env) -> T, T>(f: F) -> T {
        thread_local! {
            static ENV: RefCell<Env> = RefCell::new(Env {
                pushd_stack: vec![env::current_dir().unwrap()],
                pushenv_stack: vec![],
            });
        }
        ENV.with(|it| f(&mut *it.borrow_mut()))
    }

    fn pushd(&mut self, dir: PathBuf) {
        let dir = self.cwd().join(dir);
        self.pushd_stack.push(dir);
        env::set_current_dir(self.cwd()).unwrap();
    }
    fn popd(&mut self) {
        self.pushd_stack.pop().unwrap();
        env::set_current_dir(self.cwd()).unwrap();
    }
    fn pushenv(&mut self, var: OsString, value: OsString) {
        self.pushenv_stack.push((var.clone(), env::var_os(&var)));
        env::set_var(var, value)
    }
    fn popenv(&mut self) {
        let (var, value) = self.pushenv_stack.pop().unwrap();
        match value {
            None => env::remove_var(var),
            Some(value) => env::set_var(var, value),
        }
    }
    fn cwd(&self) -> &Path {
        self.pushd_stack.last().unwrap()
    }
}
