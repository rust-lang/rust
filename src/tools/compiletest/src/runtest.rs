use crate::common::Debugger;
use crate::common::{incremental_dir, Incremental, Ui};
use crate::common::{Config, TestPaths};
use crate::header::TestProps;
use crate::json;

use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::env;
use std::fs::create_dir_all;
use std::hash::{Hash, Hasher};
use std::io::prelude::*;
use std::io::{self};
use std::path::PathBuf;
use std::process::{Child, ExitStatus, Output};
use std::str;

use tracing::*;

mod test_cx;
use test_cx::TestCx;

#[cfg(test)]
mod tests;

#[cfg(windows)]
fn disable_error_reporting<F: FnOnce() -> R, R>(f: F) -> R {
    use std::sync::Mutex;
    use winapi::um::errhandlingapi::SetErrorMode;
    use winapi::um::winbase::SEM_NOGPFAULTERRORBOX;

    lazy_static! {
        static ref LOCK: Mutex<()> = Mutex::new(());
    }
    // Error mode is a global variable, so lock it so only one thread will change it
    let _lock = LOCK.lock().unwrap();

    // Tell Windows to not show any UI on errors (such as terminating abnormally).
    // This is important for running tests, since some of them use abnormal
    // termination by design. This mode is inherited by all child processes.
    unsafe {
        let old_mode = SetErrorMode(SEM_NOGPFAULTERRORBOX); // read inherited flags
        SetErrorMode(old_mode | SEM_NOGPFAULTERRORBOX);
        let r = f();
        SetErrorMode(old_mode);
        r
    }
}

#[cfg(not(windows))]
fn disable_error_reporting<F: FnOnce() -> R, R>(f: F) -> R {
    f()
}

/// The name of the environment variable that holds dynamic library locations.
pub fn dylib_env_var() -> &'static str {
    if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else if cfg!(target_os = "haiku") {
        "LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    }
}

/// The platform-specific library name
pub fn get_lib_name(lib: &str, dylib: bool) -> String {
    // In some casess (e.g. MUSL), we build a static
    // library, rather than a dynamic library.
    // In this case, the only path we can pass
    // with '--extern-meta' is the '.lib' file
    if !dylib {
        return format!("lib{}.rlib", lib);
    }

    if cfg!(windows) {
        format!("{}.dll", lib)
    } else if cfg!(target_os = "macos") {
        format!("lib{}.dylib", lib)
    } else {
        format!("lib{}.so", lib)
    }
}

#[derive(Debug, PartialEq)]
pub enum DiffLine {
    Context(String),
    Expected(String),
    Resulting(String),
}

#[derive(Debug, PartialEq)]
pub struct Mismatch {
    pub line_number: u32,
    pub lines: Vec<DiffLine>,
}

impl Mismatch {
    fn new(line_number: u32) -> Mismatch {
        Mismatch { line_number, lines: Vec::new() }
    }
}

// Produces a diff between the expected output and actual output.
pub fn make_diff(expected: &str, actual: &str, context_size: usize) -> Vec<Mismatch> {
    let mut line_number = 1;
    let mut context_queue: VecDeque<&str> = VecDeque::with_capacity(context_size);
    let mut lines_since_mismatch = context_size + 1;
    let mut results = Vec::new();
    let mut mismatch = Mismatch::new(0);

    for result in diff::lines(expected, actual) {
        match result {
            diff::Result::Left(str) => {
                if lines_since_mismatch >= context_size && lines_since_mismatch > 0 {
                    results.push(mismatch);
                    mismatch = Mismatch::new(line_number - context_queue.len() as u32);
                }

                while let Some(line) = context_queue.pop_front() {
                    mismatch.lines.push(DiffLine::Context(line.to_owned()));
                }

                mismatch.lines.push(DiffLine::Expected(str.to_owned()));
                line_number += 1;
                lines_since_mismatch = 0;
            }
            diff::Result::Right(str) => {
                if lines_since_mismatch >= context_size && lines_since_mismatch > 0 {
                    results.push(mismatch);
                    mismatch = Mismatch::new(line_number - context_queue.len() as u32);
                }

                while let Some(line) = context_queue.pop_front() {
                    mismatch.lines.push(DiffLine::Context(line.to_owned()));
                }

                mismatch.lines.push(DiffLine::Resulting(str.to_owned()));
                lines_since_mismatch = 0;
            }
            diff::Result::Both(str, _) => {
                if context_queue.len() >= context_size {
                    let _ = context_queue.pop_front();
                }

                if lines_since_mismatch < context_size {
                    mismatch.lines.push(DiffLine::Context(str.to_owned()));
                } else if context_size > 0 {
                    context_queue.push_back(str);
                }

                line_number += 1;
                lines_since_mismatch += 1;
            }
        }
    }

    results.push(mismatch);
    results.remove(0);

    results
}

fn write_diff(expected: &str, actual: &str, context_size: usize) -> String {
    use std::fmt::Write;
    let mut output = String::new();
    let diff_results = make_diff(expected, actual, context_size);
    for result in diff_results {
        let mut line_number = result.line_number;
        for line in result.lines {
            match line {
                DiffLine::Expected(e) => {
                    writeln!(output, "-\t{}", e).unwrap();
                    line_number += 1;
                }
                DiffLine::Context(c) => {
                    writeln!(output, "{}\t{}", line_number, c).unwrap();
                    line_number += 1;
                }
                DiffLine::Resulting(r) => {
                    writeln!(output, "+\t{}", r).unwrap();
                }
            }
        }
        writeln!(output).unwrap();
    }
    output
}

pub fn run(config: Config, testpaths: &TestPaths, revision: Option<&str>) {
    match &*config.target {
        "arm-linux-androideabi"
        | "armv7-linux-androideabi"
        | "thumbv7neon-linux-androideabi"
        | "aarch64-linux-android" => {
            if !config.adb_device_status {
                panic!("android device not available");
            }
        }

        _ => {
            // android has its own gdb handling
            if config.debugger == Some(Debugger::Gdb) && config.gdb.is_none() {
                panic!("gdb not available but debuginfo gdb debuginfo test requested");
            }
        }
    }

    if config.verbose {
        // We're going to be dumping a lot of info. Start on a new line.
        print!("\n\n");
    }
    debug!("running {:?}", testpaths.file.display());
    let mut props = TestProps::from_file(&testpaths.file, revision, &config);
    if props.incremental {
        props.incremental_dir = Some(incremental_dir(&config, testpaths));
    }

    let cx = TestCx { config: &config, props: &props, testpaths, revision };
    create_dir_all(&cx.output_base_dir()).unwrap();
    if props.incremental {
        cx.init_incremental_test();
    }

    if config.mode == Incremental {
        // Incremental tests are special because they cannot be run in
        // parallel.
        assert!(!props.revisions.is_empty(), "Incremental tests require revisions.");
        for revision in &props.revisions {
            let mut revision_props = TestProps::from_file(&testpaths.file, Some(revision), &config);
            revision_props.incremental_dir = props.incremental_dir.clone();
            let rev_cx = TestCx {
                config: &config,
                props: &revision_props,
                testpaths,
                revision: Some(revision),
            };
            rev_cx.run_revision();
        }
    } else {
        cx.run_revision();
    }

    cx.create_stamp();
}

pub fn compute_stamp_hash(config: &Config) -> String {
    let mut hash = DefaultHasher::new();
    config.stage_id.hash(&mut hash);
    config.run.hash(&mut hash);

    match config.debugger {
        Some(Debugger::Cdb) => {
            config.cdb.hash(&mut hash);
        }

        Some(Debugger::Gdb) => {
            config.gdb.hash(&mut hash);
            env::var_os("PATH").hash(&mut hash);
            env::var_os("PYTHONPATH").hash(&mut hash);
        }

        Some(Debugger::Lldb) => {
            config.lldb_python.hash(&mut hash);
            config.lldb_python_dir.hash(&mut hash);
            env::var_os("PATH").hash(&mut hash);
            env::var_os("PYTHONPATH").hash(&mut hash);
        }

        None => {}
    }

    if let Ui = config.mode {
        config.force_pass_mode.hash(&mut hash);
    }

    format!("{:x}", hash.finish())
}

struct DebuggerCommands {
    commands: Vec<String>,
    check_lines: Vec<String>,
    breakpoint_lines: Vec<usize>,
}

enum ReadFrom {
    Path,
    Stdin(String),
}

enum TestOutput {
    Compile,
    Run,
}

/// Will this test be executed? Should we use `make_exe_name`?
#[derive(Copy, Clone, PartialEq)]
enum WillExecute {
    Yes,
    No,
    Disabled,
}

/// Should `--emit metadata` be used?
#[derive(Copy, Clone)]
enum EmitMetadata {
    Yes,
    No,
}

struct ProcArgs {
    prog: String,
    args: Vec<String>,
}

pub struct ProcRes {
    status: ExitStatus,
    stdout: String,
    stderr: String,
    cmdline: String,
}

impl ProcRes {
    pub fn fatal(&self, err: Option<&str>, on_failure: impl FnOnce()) -> ! {
        if let Some(e) = err {
            println!("\nerror: {}", e);
        }
        print!(
            "\
             status: {}\n\
             command: {}\n\
             stdout:\n\
             ------------------------------------------\n\
             {}\n\
             ------------------------------------------\n\
             stderr:\n\
             ------------------------------------------\n\
             {}\n\
             ------------------------------------------\n\
             \n",
            self.status,
            self.cmdline,
            json::extract_rendered(&self.stdout),
            json::extract_rendered(&self.stderr),
        );
        on_failure();
        // Use resume_unwind instead of panic!() to prevent a panic message + backtrace from
        // compiletest, which is unnecessary noise.
        std::panic::resume_unwind(Box::new(()));
    }
}

#[derive(Debug)]
enum TargetLocation {
    ThisFile(PathBuf),
    ThisDirectory(PathBuf),
}

enum AllowUnused {
    Yes,
    No,
}

fn read2_abbreviated(mut child: Child) -> io::Result<Output> {
    use crate::read2::read2;
    use std::mem::replace;

    const HEAD_LEN: usize = 160 * 1024;
    const TAIL_LEN: usize = 256 * 1024;

    enum ProcOutput {
        Full(Vec<u8>),
        Abbreviated { head: Vec<u8>, skipped: usize, tail: Box<[u8]> },
    }

    impl ProcOutput {
        fn extend(&mut self, data: &[u8]) {
            let new_self = match *self {
                ProcOutput::Full(ref mut bytes) => {
                    bytes.extend_from_slice(data);
                    let new_len = bytes.len();
                    if new_len <= HEAD_LEN + TAIL_LEN {
                        return;
                    }
                    let tail = bytes.split_off(new_len - TAIL_LEN).into_boxed_slice();
                    let head = replace(bytes, Vec::new());
                    let skipped = new_len - HEAD_LEN - TAIL_LEN;
                    ProcOutput::Abbreviated { head, skipped, tail }
                }
                ProcOutput::Abbreviated { ref mut skipped, ref mut tail, .. } => {
                    *skipped += data.len();
                    if data.len() <= TAIL_LEN {
                        tail[..data.len()].copy_from_slice(data);
                        tail.rotate_left(data.len());
                    } else {
                        tail.copy_from_slice(&data[(data.len() - TAIL_LEN)..]);
                    }
                    return;
                }
            };
            *self = new_self;
        }

        fn into_bytes(self) -> Vec<u8> {
            match self {
                ProcOutput::Full(bytes) => bytes,
                ProcOutput::Abbreviated { mut head, skipped, tail } => {
                    write!(&mut head, "\n\n<<<<<< SKIPPED {} BYTES >>>>>>\n\n", skipped).unwrap();
                    head.extend_from_slice(&tail);
                    head
                }
            }
        }
    }

    let mut stdout = ProcOutput::Full(Vec::new());
    let mut stderr = ProcOutput::Full(Vec::new());

    drop(child.stdin.take());
    read2(
        child.stdout.take().unwrap(),
        child.stderr.take().unwrap(),
        &mut |is_stdout, data, _| {
            if is_stdout { &mut stdout } else { &mut stderr }.extend(data);
            data.clear();
        },
    )?;
    let status = child.wait()?;

    Ok(Output { status, stdout: stdout.into_bytes(), stderr: stderr.into_bytes() })
}
