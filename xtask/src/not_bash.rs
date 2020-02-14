//! A bad shell -- small cross platform module for writing glue code
use std::{
    cell::RefCell,
    env,
    ffi::OsStr,
    fs,
    path::PathBuf,
    process::{Command, Stdio},
};

use anyhow::{bail, Context, Result};

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

impl Drop for Pushd {
    fn drop(&mut self) {
        Env::with(|env| env.popd())
    }
}

pub fn rm(glob: &str) -> Result<()> {
    let cwd = Env::with(|env| env.cwd());
    ls(glob)?.into_iter().try_for_each(|it| fs::remove_file(cwd.join(it)))?;
    Ok(())
}

pub fn ls(glob: &str) -> Result<Vec<PathBuf>> {
    let cwd = Env::with(|env| env.cwd());
    let mut res = Vec::new();
    for entry in fs::read_dir(&cwd)? {
        let entry = entry?;
        if matches(&entry.file_name(), glob) {
            let path = entry.path();
            let path = path.strip_prefix(&cwd).unwrap();
            res.push(path.to_path_buf())
        }
    }
    return Ok(res);

    fn matches(file_name: &OsStr, glob: &str) -> bool {
        assert!(glob.starts_with('*'));
        file_name.to_string_lossy().ends_with(&glob[1..])
    }
}

#[doc(hidden)]
pub fn run_process(cmd: String, echo: bool) -> Result<String> {
    run_process_inner(&cmd, echo).with_context(|| format!("process `{}` failed", cmd))
}

fn run_process_inner(cmd: &str, echo: bool) -> Result<String> {
    let cwd = Env::with(|env| env.cwd());
    let mut args = shelx(cmd);
    let binary = args.remove(0);

    if echo {
        println!("> {}", cmd)
    }

    let output = Command::new(binary)
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stderr(Stdio::inherit())
        .output()?;
    let stdout = String::from_utf8(output.stdout)?;

    if echo {
        print!("{}", stdout)
    }

    if !output.status.success() {
        bail!("returned non-zero status: {}", output.status)
    }

    Ok(stdout)
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
        self.pushd_stack.push(dir)
    }
    fn popd(&mut self) {
        self.pushd_stack.pop().unwrap();
    }
    fn cwd(&self) -> PathBuf {
        self.pushd_stack.last().cloned().unwrap_or_else(|| env::current_dir().unwrap())
    }
}
