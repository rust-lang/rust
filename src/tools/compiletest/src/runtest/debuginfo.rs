use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::process::{Command, Output, Stdio};

use camino::Utf8Path;
use tracing::debug;

use super::debugger::DebuggerCommands;
use super::{Debugger, Emit, ProcRes, TestCx, Truncated, WillExecute};
use crate::common::Config;
use crate::debuggers::{extract_gdb_version, is_android_gdb_target};
use crate::util::logv;

impl TestCx<'_> {
    pub(super) fn run_debuginfo_test(&self) {
        match self.config.debugger.unwrap() {
            Debugger::Cdb => self.run_debuginfo_cdb_test(),
            Debugger::Gdb => self.run_debuginfo_gdb_test(),
            Debugger::Lldb => self.run_debuginfo_lldb_test(),
        }
    }

    fn run_debuginfo_cdb_test(&self) {
        let config = Config {
            target_rustcflags: self.cleanup_debug_info_options(&self.config.target_rustcflags),
            host_rustcflags: self.cleanup_debug_info_options(&self.config.host_rustcflags),
            ..self.config.clone()
        };

        let test_cx = TestCx { config: &config, ..*self };

        test_cx.run_debuginfo_cdb_test_no_opt();
    }

    fn run_debuginfo_cdb_test_no_opt(&self) {
        let exe_file = self.make_exe_name();

        // Existing PDB files are update in-place. When changing the debuginfo
        // the compiler generates for something, this can lead to the situation
        // where both the old and the new version of the debuginfo for the same
        // type is present in the PDB, which is very confusing.
        // Therefore we delete any existing PDB file before compiling the test
        // case.
        // FIXME: If can reliably detect that MSVC's link.exe is used, then
        //        passing `/INCREMENTAL:NO` might be a cleaner way to do this.
        let pdb_file = exe_file.with_extension(".pdb");
        if pdb_file.exists() {
            std::fs::remove_file(pdb_file).unwrap();
        }

        // compile test file (it should have 'compile-flags:-g' in the header)
        let should_run = self.run_if_enabled();
        let compile_result = self.compile_test(should_run, Emit::None);
        if !compile_result.status.success() {
            self.fatal_proc_rec("compilation failed!", &compile_result);
        }
        if let WillExecute::Disabled = should_run {
            return;
        }

        // Parse debugger commands etc from test files
        let dbg_cmds = DebuggerCommands::parse_from(&self.testpaths.file, self.config, "cdb")
            .unwrap_or_else(|e| self.fatal(&e));

        // https://docs.microsoft.com/en-us/windows-hardware/drivers/debugger/debugger-commands
        let mut script_str = String::with_capacity(2048);
        script_str.push_str("version\n"); // List CDB (and more) version info in test output
        script_str.push_str(".nvlist\n"); // List loaded `*.natvis` files, bulk of custom MSVC debug

        // If a .js file exists next to the source file being tested, then this is a JavaScript
        // debugging extension that needs to be loaded.
        let mut js_extension = self.testpaths.file.clone();
        js_extension.set_extension("cdb.js");
        if js_extension.exists() {
            script_str.push_str(&format!(".scriptload \"{}\"\n", js_extension));
        }

        // Set breakpoints on every line that contains the string "#break"
        let source_file_name = self.testpaths.file.file_name().unwrap();
        for line in &dbg_cmds.breakpoint_lines {
            script_str.push_str(&format!("bp `{}:{}`\n", source_file_name, line));
        }

        // Append the other `cdb-command:`s
        for line in &dbg_cmds.commands {
            script_str.push_str(line);
            script_str.push('\n');
        }

        script_str.push_str("qq\n"); // Quit the debugger (including remote debugger, if any)

        // Write the script into a file
        debug!("script_str = {}", script_str);
        self.dump_output_file(&script_str, "debugger.script");
        let debugger_script = self.make_out_name("debugger.script");

        let cdb_path = &self.config.cdb.as_ref().unwrap();
        let mut cdb = Command::new(cdb_path);
        cdb.arg("-lines") // Enable source line debugging.
            .arg("-cf")
            .arg(&debugger_script)
            .arg(&exe_file);

        let debugger_run_result = self.compose_and_run(
            cdb,
            self.config.run_lib_path.as_path(),
            None, // aux_path
            None, // input
        );

        if !debugger_run_result.status.success() {
            self.fatal_proc_rec("Error while running CDB", &debugger_run_result);
        }

        if let Err(e) = dbg_cmds.check_output(&debugger_run_result) {
            self.fatal_proc_rec(&e, &debugger_run_result);
        }
    }

    fn run_debuginfo_gdb_test(&self) {
        let config = Config {
            target_rustcflags: self.cleanup_debug_info_options(&self.config.target_rustcflags),
            host_rustcflags: self.cleanup_debug_info_options(&self.config.host_rustcflags),
            ..self.config.clone()
        };

        let test_cx = TestCx { config: &config, ..*self };

        test_cx.run_debuginfo_gdb_test_no_opt();
    }

    fn run_debuginfo_gdb_test_no_opt(&self) {
        let dbg_cmds = DebuggerCommands::parse_from(&self.testpaths.file, self.config, "gdb")
            .unwrap_or_else(|e| self.fatal(&e));
        let mut cmds = dbg_cmds.commands.join("\n");

        // compile test file (it should have 'compile-flags:-g' in the header)
        let should_run = self.run_if_enabled();
        let compiler_run_result = self.compile_test(should_run, Emit::None);
        if !compiler_run_result.status.success() {
            self.fatal_proc_rec("compilation failed!", &compiler_run_result);
        }
        if let WillExecute::Disabled = should_run {
            return;
        }

        let exe_file = self.make_exe_name();

        let debugger_run_result;
        if is_android_gdb_target(&self.config.target) {
            cmds = cmds.replace("run", "continue");

            // write debugger script
            let mut script_str = String::with_capacity(2048);
            script_str.push_str(&format!("set charset {}\n", Self::charset()));
            script_str.push_str(&format!("set sysroot {}\n", &self.config.android_cross_path));
            script_str.push_str(&format!("file {}\n", exe_file));
            script_str.push_str("target remote :5039\n");
            script_str.push_str(&format!(
                "set solib-search-path \
                 ./{}/stage2/lib/rustlib/{}/lib/\n",
                self.config.host, self.config.target
            ));
            for line in &dbg_cmds.breakpoint_lines {
                script_str.push_str(
                    format!("break {}:{}\n", self.testpaths.file.file_name().unwrap(), *line)
                        .as_str(),
                );
            }
            script_str.push_str(&cmds);
            script_str.push_str("\nquit\n");

            debug!("script_str = {}", script_str);
            self.dump_output_file(&script_str, "debugger.script");

            let adb_path = &self.config.adb_path;

            Command::new(adb_path)
                .arg("push")
                .arg(&exe_file)
                .arg(&self.config.adb_test_dir)
                .status()
                .unwrap_or_else(|e| panic!("failed to exec `{adb_path:?}`: {e:?}"));

            Command::new(adb_path)
                .args(&["forward", "tcp:5039", "tcp:5039"])
                .status()
                .unwrap_or_else(|e| panic!("failed to exec `{adb_path:?}`: {e:?}"));

            let adb_arg = format!(
                "export LD_LIBRARY_PATH={}; \
                 gdbserver{} :5039 {}/{}",
                self.config.adb_test_dir.clone(),
                if self.config.target.contains("aarch64") { "64" } else { "" },
                self.config.adb_test_dir.clone(),
                exe_file.file_name().unwrap()
            );

            debug!("adb arg: {}", adb_arg);
            let mut adb = Command::new(adb_path)
                .args(&["shell", &adb_arg])
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap_or_else(|e| panic!("failed to exec `{adb_path:?}`: {e:?}"));

            // Wait for the gdbserver to print out "Listening on port ..."
            // at which point we know that it's started and then we can
            // execute the debugger below.
            let mut stdout = BufReader::new(adb.stdout.take().unwrap());
            let mut line = String::new();
            loop {
                line.truncate(0);
                stdout.read_line(&mut line).unwrap();
                if line.starts_with("Listening on port 5039") {
                    break;
                }
            }
            drop(stdout);

            let mut debugger_script = OsString::from("-command=");
            debugger_script.push(self.make_out_name("debugger.script"));
            let debugger_opts: &[&OsStr] =
                &["-quiet".as_ref(), "-batch".as_ref(), "-nx".as_ref(), &debugger_script];

            let gdb_path = self.config.gdb.as_ref().unwrap();
            let Output { status, stdout, stderr } = Command::new(&gdb_path)
                .args(debugger_opts)
                .output()
                .unwrap_or_else(|e| panic!("failed to exec `{gdb_path:?}`: {e:?}"));
            let cmdline = {
                let mut gdb = Command::new(&format!("{}-gdb", self.config.target));
                gdb.args(debugger_opts);
                // FIXME(jieyouxu): don't pass an empty Path
                let cmdline = self.make_cmdline(&gdb, Utf8Path::new(""));
                logv(self.config, format!("executing {}", cmdline));
                cmdline
            };

            debugger_run_result = ProcRes {
                status,
                stdout: String::from_utf8(stdout).unwrap(),
                stderr: String::from_utf8(stderr).unwrap(),
                truncated: Truncated::No,
                cmdline,
            };
            if adb.kill().is_err() {
                println!("Adb process is already finished.");
            }
        } else {
            let rust_pp_module_abs_path = self.config.src_root.join("src").join("etc");
            // write debugger script
            let mut script_str = String::with_capacity(2048);
            script_str.push_str(&format!("set charset {}\n", Self::charset()));
            script_str.push_str("show version\n");

            match self.config.gdb_version {
                Some(version) => {
                    println!("NOTE: compiletest thinks it is using GDB version {}", version);

                    if version > extract_gdb_version("7.4").unwrap() {
                        // Add the directory containing the pretty printers to
                        // GDB's script auto loading safe path
                        script_str.push_str(&format!(
                            "add-auto-load-safe-path {}\n",
                            rust_pp_module_abs_path.as_str().replace(r"\", r"\\")
                        ));

                        // Add the directory containing the output binary to
                        // include embedded pretty printers to GDB's script
                        // auto loading safe path
                        script_str.push_str(&format!(
                            "add-auto-load-safe-path {}\n",
                            self.output_base_dir().as_str().replace(r"\", r"\\")
                        ));
                    }
                }
                _ => {
                    println!(
                        "NOTE: compiletest does not know which version of \
                         GDB it is using"
                    );
                }
            }

            // The following line actually doesn't have to do anything with
            // pretty printing, it just tells GDB to print values on one line:
            script_str.push_str("set print pretty off\n");

            // Add the pretty printer directory to GDB's source-file search path
            script_str.push_str(&format!(
                "directory {}\n",
                rust_pp_module_abs_path.as_str().replace(r"\", r"\\")
            ));

            // Load the target executable
            script_str.push_str(&format!("file {}\n", exe_file.as_str().replace(r"\", r"\\")));

            // Force GDB to print values in the Rust format.
            script_str.push_str("set language rust\n");

            // Add line breakpoints
            for line in &dbg_cmds.breakpoint_lines {
                script_str.push_str(&format!(
                    "break '{}':{}\n",
                    self.testpaths.file.file_name().unwrap(),
                    *line
                ));
            }

            script_str.push_str(&cmds);
            script_str.push_str("\nquit\n");

            debug!("script_str = {}", script_str);
            self.dump_output_file(&script_str, "debugger.script");

            let mut debugger_script = OsString::from("-command=");
            debugger_script.push(self.make_out_name("debugger.script"));

            let debugger_opts: &[&OsStr] =
                &["-quiet".as_ref(), "-batch".as_ref(), "-nx".as_ref(), &debugger_script];

            let mut gdb = Command::new(self.config.gdb.as_ref().unwrap());
            let pythonpath = if let Ok(pp) = std::env::var("PYTHONPATH") {
                format!("{pp}:{rust_pp_module_abs_path}")
            } else {
                rust_pp_module_abs_path.to_string()
            };
            gdb.args(debugger_opts).env("PYTHONPATH", pythonpath);

            debugger_run_result =
                self.compose_and_run(gdb, self.config.run_lib_path.as_path(), None, None);
        }

        if !debugger_run_result.status.success() {
            self.fatal_proc_rec("gdb failed to execute", &debugger_run_result);
        }

        if let Err(e) = dbg_cmds.check_output(&debugger_run_result) {
            self.fatal_proc_rec(&e, &debugger_run_result);
        }
    }

    fn run_debuginfo_lldb_test(&self) {
        if self.config.lldb_python_dir.is_none() {
            self.fatal("Can't run LLDB test because LLDB's python path is not set.");
        }

        let config = Config {
            target_rustcflags: self.cleanup_debug_info_options(&self.config.target_rustcflags),
            host_rustcflags: self.cleanup_debug_info_options(&self.config.host_rustcflags),
            ..self.config.clone()
        };

        let test_cx = TestCx { config: &config, ..*self };

        test_cx.run_debuginfo_lldb_test_no_opt();
    }

    fn run_debuginfo_lldb_test_no_opt(&self) {
        // compile test file (it should have 'compile-flags:-g' in the header)
        let should_run = self.run_if_enabled();
        let compile_result = self.compile_test(should_run, Emit::None);
        if !compile_result.status.success() {
            self.fatal_proc_rec("compilation failed!", &compile_result);
        }
        if let WillExecute::Disabled = should_run {
            return;
        }

        let exe_file = self.make_exe_name();

        match self.config.lldb_version {
            Some(ref version) => {
                println!("NOTE: compiletest thinks it is using LLDB version {}", version);
            }
            _ => {
                println!(
                    "NOTE: compiletest does not know which version of \
                     LLDB it is using"
                );
            }
        }

        // Parse debugger commands etc from test files
        let dbg_cmds = DebuggerCommands::parse_from(&self.testpaths.file, self.config, "lldb")
            .unwrap_or_else(|e| self.fatal(&e));

        // Write debugger script:
        // We don't want to hang when calling `quit` while the process is still running
        let mut script_str = String::from("settings set auto-confirm true\n");

        // Make LLDB emit its version, so we have it documented in the test output
        script_str.push_str("version\n");

        // Switch LLDB into "Rust mode".
        let rust_pp_module_abs_path = self.config.src_root.join("src/etc");

        script_str.push_str(&format!(
            "command script import {}/lldb_lookup.py\n",
            rust_pp_module_abs_path
        ));
        File::open(rust_pp_module_abs_path.join("lldb_commands"))
            .and_then(|mut file| file.read_to_string(&mut script_str))
            .expect("Failed to read lldb_commands");

        // Set breakpoints on every line that contains the string "#break"
        let source_file_name = self.testpaths.file.file_name().unwrap();
        for line in &dbg_cmds.breakpoint_lines {
            script_str.push_str(&format!(
                "breakpoint set --file '{}' --line {}\n",
                source_file_name, line
            ));
        }

        // Append the other commands
        for line in &dbg_cmds.commands {
            script_str.push_str(line);
            script_str.push('\n');
        }

        // Finally, quit the debugger
        script_str.push_str("\nquit\n");

        // Write the script into a file
        debug!("script_str = {}", script_str);
        self.dump_output_file(&script_str, "debugger.script");
        let debugger_script = self.make_out_name("debugger.script");

        // Let LLDB execute the script via lldb_batchmode.py
        let debugger_run_result = self.run_lldb(&exe_file, &debugger_script);

        if !debugger_run_result.status.success() {
            self.fatal_proc_rec("Error while running LLDB", &debugger_run_result);
        }

        if let Err(e) = dbg_cmds.check_output(&debugger_run_result) {
            self.fatal_proc_rec(&e, &debugger_run_result);
        }
    }

    fn run_lldb(&self, test_executable: &Utf8Path, debugger_script: &Utf8Path) -> ProcRes {
        // Prepare the lldb_batchmode which executes the debugger script
        let lldb_script_path = self.config.src_root.join("src/etc/lldb_batchmode.py");
        let pythonpath = if let Ok(pp) = std::env::var("PYTHONPATH") {
            format!("{pp}:{}", self.config.lldb_python_dir.as_ref().unwrap())
        } else {
            self.config.lldb_python_dir.clone().unwrap()
        };
        self.run_command_to_procres(
            Command::new(&self.config.python)
                .arg(&lldb_script_path)
                .arg(test_executable)
                .arg(debugger_script)
                .env("PYTHONUNBUFFERED", "1") // Help debugging #78665
                .env("PYTHONPATH", pythonpath),
        )
    }

    fn cleanup_debug_info_options(&self, options: &Vec<String>) -> Vec<String> {
        // Remove options that are either unwanted (-O) or may lead to duplicates due to RUSTFLAGS.
        let options_to_remove = ["-O".to_owned(), "-g".to_owned(), "--debuginfo".to_owned()];

        options.iter().filter(|x| !options_to_remove.contains(x)).cloned().collect()
    }
}
