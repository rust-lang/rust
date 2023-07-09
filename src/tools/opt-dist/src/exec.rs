use crate::environment::Environment;
use crate::metrics::{load_metrics, record_metrics};
use crate::timer::TimerSection;
use crate::training::{LlvmBoltProfile, LlvmPGOProfile, RustcPGOProfile};
use camino::{Utf8Path, Utf8PathBuf};
use std::collections::BTreeMap;
use std::fs::File;
use std::process::{Command, Stdio};

#[derive(Default)]
pub struct CmdBuilder {
    args: Vec<String>,
    env: BTreeMap<String, String>,
    workdir: Option<Utf8PathBuf>,
    output: Option<Utf8PathBuf>,
}

impl CmdBuilder {
    pub fn arg(mut self, arg: &str) -> Self {
        self.args.push(arg.to_string());
        self
    }

    pub fn env(mut self, name: &str, value: &str) -> Self {
        self.env.insert(name.to_string(), value.to_string());
        self
    }

    pub fn workdir(mut self, path: &Utf8Path) -> Self {
        self.workdir = Some(path.to_path_buf());
        self
    }

    pub fn redirect_output(mut self, path: Utf8PathBuf) -> Self {
        self.output = Some(path);
        self
    }

    pub fn run(self) -> anyhow::Result<()> {
        let mut cmd_str = String::new();
        cmd_str.push_str(
            &self
                .env
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect::<Vec<_>>()
                .join(" "),
        );
        if !self.env.is_empty() {
            cmd_str.push(' ');
        }
        cmd_str.push_str(&self.args.join(" "));
        if let Some(ref path) = self.output {
            cmd_str.push_str(&format!(" > {path:?}"));
        }
        cmd_str.push_str(&format!(
            " [at {}]",
            self.workdir
                .clone()
                .unwrap_or_else(|| std::env::current_dir().unwrap().try_into().unwrap())
        ));
        log::info!("Executing `{cmd_str}`");

        let mut cmd = Command::new(&self.args[0]);
        cmd.stdin(Stdio::null());
        cmd.args(self.args.iter().skip(1));
        for (key, value) in &self.env {
            cmd.env(key, value);
        }
        if let Some(ref output) = self.output {
            cmd.stdout(File::create(output.clone().into_std_path_buf())?);
        }
        if let Some(ref workdir) = self.workdir {
            cmd.current_dir(workdir.clone().into_std_path_buf());
        }
        let exit_status = cmd.spawn()?.wait()?;
        if !exit_status.success() {
            Err(anyhow::anyhow!(
                "Command {cmd_str} has failed with exit code {:?}",
                exit_status.code(),
            ))
        } else {
            Ok(())
        }
    }
}

pub fn cmd(args: &[&str]) -> CmdBuilder {
    assert!(!args.is_empty());
    CmdBuilder { args: args.iter().map(|s| s.to_string()).collect(), ..Default::default() }
}

pub struct Bootstrap {
    cmd: CmdBuilder,
    metrics_path: Utf8PathBuf,
}

impl Bootstrap {
    pub fn build(env: &dyn Environment) -> Self {
        let metrics_path = env.build_root().join("build").join("metrics.json");
        let cmd = cmd(&[
            env.python_binary(),
            env.checkout_path().join("x.py").as_str(),
            "build",
            "--target",
            &env.host_triple(),
            "--host",
            &env.host_triple(),
            "--stage",
            "2",
            "library/std",
        ])
        .env("RUST_BACKTRACE", "full");
        Self { cmd, metrics_path }
    }

    pub fn dist(env: &dyn Environment, dist_args: &[String]) -> Self {
        let metrics_path = env.build_root().join("build").join("metrics.json");
        let cmd = cmd(&dist_args.iter().map(|arg| arg.as_str()).collect::<Vec<_>>())
            .env("RUST_BACKTRACE", "full");
        Self { cmd, metrics_path }
    }

    pub fn llvm_pgo_instrument(mut self, profile_dir: &Utf8Path) -> Self {
        self.cmd = self
            .cmd
            .arg("--llvm-profile-generate")
            .env("LLVM_PROFILE_DIR", profile_dir.join("prof-%p").as_str());
        self
    }

    pub fn llvm_pgo_optimize(mut self, profile: &LlvmPGOProfile) -> Self {
        self.cmd = self.cmd.arg("--llvm-profile-use").arg(profile.0.as_str());
        self
    }

    pub fn rustc_pgo_instrument(mut self, profile_dir: &Utf8Path) -> Self {
        self.cmd = self.cmd.arg("--rust-profile-generate").arg(profile_dir.as_str());
        self
    }

    pub fn rustc_pgo_optimize(mut self, profile: &RustcPGOProfile) -> Self {
        self.cmd = self.cmd.arg("--rust-profile-use").arg(profile.0.as_str());
        self
    }

    pub fn llvm_bolt_instrument(mut self) -> Self {
        self.cmd = self.cmd.arg("--llvm-bolt-profile-generate");
        self
    }

    pub fn llvm_bolt_optimize(mut self, profile: &LlvmBoltProfile) -> Self {
        self.cmd = self.cmd.arg("--llvm-bolt-profile-use").arg(profile.0.as_str());
        self
    }

    /// Do not rebuild rustc, and use a previously built rustc sysroot instead.
    pub fn avoid_rustc_rebuild(mut self) -> Self {
        self.cmd = self.cmd.arg("--keep-stage").arg("0").arg("--keep-stage").arg("1");
        self
    }

    pub fn run(self, timer: &mut TimerSection) -> anyhow::Result<()> {
        self.cmd.run()?;
        let metrics = load_metrics(&self.metrics_path)?;
        record_metrics(&metrics, timer);
        Ok(())
    }
}
