use std::ffi::{OsStr, OsString};

use rustc_data_structures::fx::FxHashMap;

use self::shims::unix::UnixEnvVars;
use self::shims::windows::WindowsEnvVars;
use crate::*;

#[derive(Default)]
pub enum EnvVars<'tcx> {
    #[default]
    Uninit,
    Unix(UnixEnvVars<'tcx>),
    Windows(WindowsEnvVars),
}

impl VisitProvenance for EnvVars<'_> {
    fn visit_provenance(&self, visit: &mut VisitWith<'_>) {
        match self {
            EnvVars::Uninit => {}
            EnvVars::Unix(env) => env.visit_provenance(visit),
            EnvVars::Windows(env) => env.visit_provenance(visit),
        }
    }
}

impl<'tcx> EnvVars<'tcx> {
    pub(crate) fn init(
        ecx: &mut InterpCx<'tcx, MiriMachine<'tcx>>,
        config: &MiriConfig,
    ) -> InterpResult<'tcx> {
        // Initialize the `env_vars` map.
        // Skip the loop entirely if we don't want to forward anything.
        let mut env_vars = FxHashMap::default();
        if ecx.machine.communicate() || !config.forwarded_env_vars.is_empty() {
            for (name, value) in &config.env {
                let forward = ecx.machine.communicate()
                    || config.forwarded_env_vars.iter().any(|v| **v == *name);
                if forward {
                    env_vars.insert(OsString::from(name), OsString::from(value));
                }
            }
        }

        for (name, value) in &config.set_env_vars {
            env_vars.insert(OsString::from(name), OsString::from(value));
        }

        let env_vars = if ecx.target_os_is_unix() {
            EnvVars::Unix(UnixEnvVars::new(ecx, env_vars)?)
        } else if ecx.tcx.sess.target.os == "windows" {
            EnvVars::Windows(WindowsEnvVars::new(ecx, env_vars)?)
        } else {
            // Used e.g. for wasi
            EnvVars::Uninit
        };
        ecx.machine.env_vars = env_vars;

        interp_ok(())
    }

    pub(crate) fn unix(&self) -> &UnixEnvVars<'tcx> {
        match self {
            EnvVars::Unix(env) => env,
            _ => unreachable!(),
        }
    }

    pub(crate) fn unix_mut(&mut self) -> &mut UnixEnvVars<'tcx> {
        match self {
            EnvVars::Unix(env) => env,
            _ => unreachable!(),
        }
    }

    pub(crate) fn windows(&self) -> &WindowsEnvVars {
        match self {
            EnvVars::Windows(env) => env,
            _ => unreachable!(),
        }
    }

    pub(crate) fn windows_mut(&mut self) -> &mut WindowsEnvVars {
        match self {
            EnvVars::Windows(env) => env,
            _ => unreachable!(),
        }
    }
}

impl<'tcx> EvalContextExt<'tcx> for crate::MiriInterpCx<'tcx> {}
pub trait EvalContextExt<'tcx>: crate::MiriInterpCxExt<'tcx> {
    /// Try to get an environment variable from the interpreted program's environment. This is
    /// useful for implementing shims which are documented to read from the environment.
    fn get_env_var(&mut self, name: &OsStr) -> InterpResult<'tcx, Option<OsString>> {
        let this = self.eval_context_ref();
        match &this.machine.env_vars {
            EnvVars::Uninit => interp_ok(None),
            EnvVars::Unix(vars) => vars.get(this, name),
            EnvVars::Windows(vars) => vars.get(name),
        }
    }

    /// Get the process identifier.
    fn get_pid(&self) -> u32 {
        let this = self.eval_context_ref();
        if this.machine.communicate() { std::process::id() } else { 1000 }
    }

    /// Get an "OS" thread ID for the current thread.
    fn get_current_tid(&self) -> u32 {
        let this = self.eval_context_ref();
        self.get_tid(this.machine.threads.active_thread())
    }

    /// Get an "OS" thread ID for any thread.
    fn get_tid(&self, thread: ThreadId) -> u32 {
        let this = self.eval_context_ref();
        let index = thread.to_u32();
        let target_os = &this.tcx.sess.target.os;
        if target_os == "linux" || target_os == "netbsd" {
            // On Linux, the main thread has PID == TID so we uphold this. NetBSD also appears
            // to exhibit the same behavior, though I can't find a citation.
            this.get_pid().strict_add(index)
        } else {
            // Other platforms do not display any relationship between PID and TID.
            index
        }
    }
}
