use std::process::{Command, Output, Stdio};

use anyhow::{Context, Result};

use crate::project_root;

pub struct Cmd<'a> {
    pub unix: &'a str,
    pub windows: &'a str,
    pub work_dir: &'a str,
}

impl Cmd<'_> {
    pub fn run(self) -> Result<()> {
        if cfg!(windows) {
            run(self.windows, self.work_dir)
        } else {
            run(self.unix, self.work_dir)
        }
    }
    pub fn run_with_output(self) -> Result<Output> {
        if cfg!(windows) {
            run_with_output(self.windows, self.work_dir)
        } else {
            run_with_output(self.unix, self.work_dir)
        }
    }
}

pub fn run(cmdline: &str, dir: &str) -> Result<()> {
    do_run(cmdline, dir, &mut |c| {
        c.stdout(Stdio::inherit());
    })
    .map(|_| ())
}

pub fn run_with_output(cmdline: &str, dir: &str) -> Result<Output> {
    do_run(cmdline, dir, &mut |_| {})
}

fn do_run(cmdline: &str, dir: &str, f: &mut dyn FnMut(&mut Command)) -> Result<Output> {
    eprintln!("\nwill run: {}", cmdline);
    let proj_dir = project_root().join(dir);
    let mut args = cmdline.split_whitespace();
    let exec = args.next().unwrap();
    let mut cmd = Command::new(exec);
    f(cmd.args(args).current_dir(proj_dir).stderr(Stdio::inherit()));
    let output = cmd.output().with_context(|| format!("running `{}`", cmdline))?;
    if !output.status.success() {
        anyhow::bail!("`{}` exited with {}", cmdline, output.status);
    }
    Ok(output)
}
