/// Implement common helpers for command wrappers. This assumes that the command wrapper is a struct
/// containing a `cmd: Command` field. The provided helpers are:
///
/// 1. Generic argument acceptors: `arg` and `args` (delegated to [`Command`]). These are intended
///    to be *fallback* argument acceptors, when specific helpers don't make sense. Prefer to add
///    new specific helper methods over relying on these generic argument providers.
/// 2. Environment manipulation methods: `env`, `env_remove` and `env_clear`: these delegate to
///    methods of the same name on [`Command`].
/// 3. Output and execution: `run` and `run_fail` are provided. These are higher-level convenience
///    methods which wait for the command to finish running and assert that the command successfully
///    ran or failed as expected. They return [`CompletedProcess`], which can be used to assert the
///    stdout/stderr/exit code of the executed process.
///
/// Example usage:
///
/// ```ignore (illustrative)
/// struct CommandWrapper { cmd: Command } // <- required `cmd` field
///
/// crate::macros::impl_common_helpers!(CommandWrapper);
///
/// impl CommandWrapper {
///     // ... additional specific helper methods
/// }
/// ```
///
/// [`Command`]: crate::command::Command
/// [`CompletedProcess`]: crate::command::CompletedProcess
macro_rules! impl_common_helpers {
    ($wrapper: ident) => {
        impl $wrapper {
            /// Specify an environment variable.
            pub fn env<K, V>(&mut self, key: K, value: V) -> &mut Self
            where
                K: AsRef<::std::ffi::OsStr>,
                V: AsRef<::std::ffi::OsStr>,
            {
                self.cmd.env(key, value);
                self
            }

            /// Remove an environmental variable.
            pub fn env_remove<K>(&mut self, key: K) -> &mut Self
            where
                K: AsRef<::std::ffi::OsStr>,
            {
                self.cmd.env_remove(key);
                self
            }

            /// Generic command argument provider. Prefer specific helper methods if possible.
            /// Note that for some executables, arguments might be platform specific. For C/C++
            /// compilers, arguments might be platform *and* compiler specific.
            pub fn arg<S>(&mut self, arg: S) -> &mut Self
            where
                S: AsRef<::std::ffi::OsStr>,
            {
                self.cmd.arg(arg);
                self
            }

            /// Generic command arguments provider. Prefer specific helper methods if possible.
            /// Note that for some executables, arguments might be platform specific. For C/C++
            /// compilers, arguments might be platform *and* compiler specific.
            pub fn args<V, S>(&mut self, args: V) -> &mut Self
            where
                V: AsRef<[S]>,
                S: AsRef<::std::ffi::OsStr>,
            {
                self.cmd.args(args.as_ref());
                self
            }

            /// Configuration for the child process’s standard input (stdin) handle.
            ///
            /// See [`std::process::Command::stdin`].
            pub fn stdin<T: Into<::std::process::Stdio>>(&mut self, cfg: T) -> &mut Self {
                self.cmd.stdin(cfg);
                self
            }

            /// Configuration for the child process’s standard output (stdout) handle.
            ///
            /// See [`std::process::Command::stdout`].
            pub fn stdout<T: Into<::std::process::Stdio>>(&mut self, cfg: T) -> &mut Self {
                self.cmd.stdout(cfg);
                self
            }

            /// Configuration for the child process’s standard error (stderr) handle.
            ///
            /// See [`std::process::Command::stderr`].
            pub fn stderr<T: Into<::std::process::Stdio>>(&mut self, cfg: T) -> &mut Self {
                self.cmd.stderr(cfg);
                self
            }

            /// Inspect what the underlying [`Command`] is up to the
            /// current construction.
            pub fn inspect<I>(&mut self, inspector: I) -> &mut Self
            where
                I: FnOnce(&::std::process::Command),
            {
                self.cmd.inspect(inspector);
                self
            }

            /// Run the constructed command and assert that it is successfully run.
            #[track_caller]
            pub fn run(&mut self) -> crate::command::CompletedProcess {
                self.cmd.run()
            }

            /// Run the constructed command and assert that it does not successfully run.
            #[track_caller]
            pub fn run_fail(&mut self) -> crate::command::CompletedProcess {
                self.cmd.run_fail()
            }

            /// Run the command but do not check its exit status.
            /// Only use if you explicitly don't care about the exit status.
            /// Prefer to use [`Self::run`] and [`Self::run_fail`]
            /// whenever possible.
            #[track_caller]
            pub fn run_unchecked(&mut self) -> crate::command::CompletedProcess {
                self.cmd.run_unchecked()
            }

            /// Set the path where the command will be run.
            pub fn current_dir<P: AsRef<::std::path::Path>>(&mut self, path: P) -> &mut Self {
                self.cmd.current_dir(path);
                self
            }
        }
    };
}

pub(crate) use impl_common_helpers;
