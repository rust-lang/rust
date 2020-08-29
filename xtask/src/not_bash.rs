//! A bad shell -- small cross platform module for writing glue code

use std::{
    cell::RefCell,
    env,
    ffi::OsString,
    io::{self, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::{bail, Context, Result};

pub use fs_err as fs2;

#[macro_export]
macro_rules! run {
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
pub use crate::run;

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

pub fn rm_rf(path: impl AsRef<Path>) -> io::Result<()> {
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
    let mut res = Vec::new();
    for (string_piece, in_quotes) in cmd.split('\'').zip([false, true].iter().copied().cycle()) {
        if in_quotes {
            res.push(string_piece.to_string())
        } else {
            if !string_piece.is_empty() {
                res.extend(string_piece.split_ascii_whitespace().map(|it| it.to_string()))
            }
        }
    }
    res
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
        env::set_current_dir(self.cwd())
            .unwrap_or_else(|err| panic!("Failed to set cwd to {}: {}", self.cwd().display(), err));
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
