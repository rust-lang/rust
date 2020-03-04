//! A bad shell -- small cross platform module for writing glue code

use std::{
    cell::RefCell,
    env,
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
        $crate::not_bash::run_process(format!($($expr),*), $echo)
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

pub fn pwd() -> PathBuf {
    Env::with(|env| env.cwd())
}

impl Drop for Pushd {
    fn drop(&mut self) {
        Env::with(|env| env.popd())
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
pub fn run_process(cmd: String, echo: bool) -> Result<String> {
    run_process_inner(&cmd, echo).with_context(|| format!("process `{}` failed", cmd))
}

fn run_process_inner(cmd: &str, echo: bool) -> Result<String> {
    let mut args = shelx(cmd);
    let binary = args.remove(0);

    if echo {
        println!("> {}", cmd)
    }

    let output = Command::new(binary)
        .args(args)
        .current_dir(pwd())
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()?;
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

#[derive(Default)]
struct Env {
    pushd_stack: Vec<PathBuf>,
}

impl Env {
    fn with<F: FnOnce(&mut Env) -> T, T>(f: F) -> T {
        thread_local! {
            static ENV: RefCell<Env> = Default::default();
        }
        ENV.with(|it| f(&mut *it.borrow_mut()))
    }

    fn pushd(&mut self, dir: PathBuf) {
        let dir = self.cwd().join(dir);
        self.pushd_stack.push(dir)
    }
    fn popd(&mut self) {
        self.pushd_stack.pop().unwrap();
    }
    fn cwd(&self) -> PathBuf {
        self.pushd_stack.last().cloned().unwrap_or_else(|| env::current_dir().unwrap())
    }
}
