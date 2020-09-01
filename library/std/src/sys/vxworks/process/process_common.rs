use crate::os::unix::prelude::*;

use crate::collections::BTreeMap;
use crate::ffi::{CStr, CString, OsStr, OsString};
use crate::fmt;
use crate::io;
use crate::ptr;
use crate::sys::fd::FileDesc;
use crate::sys::fs::{File, OpenOptions};
use crate::sys::pipe::{self, AnonPipe};
use crate::sys_common::process::CommandEnv;

use libc::{c_char, c_int, gid_t, uid_t, EXIT_FAILURE, EXIT_SUCCESS};

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

pub struct Command {
    // Currently we try hard to ensure that the call to `.exec()` doesn't
    // actually allocate any memory. While many platforms try to ensure that
    // memory allocation works after a fork in a multithreaded process, it's
    // been observed to be buggy and somewhat unreliable, so we do our best to
    // just not do it at all!
    //
    // Along those lines, the `argv` and `envp` raw pointers here are exactly
    // what's gonna get passed to `execvp`. The `argv` array starts with the
    // `program` and ends with a NULL, and the `envp` pointer, if present, is
    // also null-terminated.
    //
    // Right now we don't support removing arguments, so there's no much fancy
    // support there, but we support adding and removing environment variables,
    // so a side table is used to track where in the `envp` array each key is
    // located. Whenever we add a key we update it in place if it's already
    // present, and whenever we remove a key we update the locations of all
    // other keys.
    program: CString,
    args: Vec<CString>,
    argv: Argv,
    env: CommandEnv,

    cwd: Option<CString>,
    uid: Option<uid_t>,
    gid: Option<gid_t>,
    saw_nul: bool,
    closures: Vec<Box<dyn FnMut() -> io::Result<()> + Send + Sync>>,
    stdin: Option<Stdio>,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,
}

// Create a new type for `Argv`, so that we can make it `Send` and `Sync`
struct Argv(Vec<*const c_char>);

// It is safe to make `Argv` `Send` and `Sync`, because it contains
// pointers to memory owned by `Command.args`
unsafe impl Send for Argv {}
unsafe impl Sync for Argv {}

// passed back to std::process with the pipes connected to the child, if any
// were requested
pub struct StdioPipes {
    pub stdin: Option<AnonPipe>,
    pub stdout: Option<AnonPipe>,
    pub stderr: Option<AnonPipe>,
}

// passed to do_exec() with configuration of what the child stdio should look
// like
pub struct ChildPipes {
    pub stdin: ChildStdio,
    pub stdout: ChildStdio,
    pub stderr: ChildStdio,
}

pub enum ChildStdio {
    Inherit,
    Explicit(c_int),
    Owned(FileDesc),
}

pub enum Stdio {
    Inherit,
    Null,
    MakePipe,
    Fd(FileDesc),
}

impl Command {
    pub fn new(program: &OsStr) -> Command {
        let mut saw_nul = false;
        let program = os2c(program, &mut saw_nul);
        Command {
            argv: Argv(vec![program.as_ptr(), ptr::null()]),
            args: vec![program.clone()],
            program,
            env: Default::default(),
            cwd: None,
            uid: None,
            gid: None,
            saw_nul,
            closures: Vec::new(),
            stdin: None,
            stdout: None,
            stderr: None,
        }
    }

    pub fn set_arg_0(&mut self, arg: &OsStr) {
        // Set a new arg0
        let arg = os2c(arg, &mut self.saw_nul);
        debug_assert!(self.argv.0.len() > 1);
        self.argv.0[0] = arg.as_ptr();
        self.args[0] = arg;
    }

    pub fn arg(&mut self, arg: &OsStr) {
        // Overwrite the trailing NULL pointer in `argv` and then add a new null
        // pointer.
        let arg = os2c(arg, &mut self.saw_nul);
        self.argv.0[self.args.len()] = arg.as_ptr();
        self.argv.0.push(ptr::null());

        // Also make sure we keep track of the owned value to schedule a
        // destructor for this memory.
        self.args.push(arg);
    }

    pub fn cwd(&mut self, dir: &OsStr) {
        self.cwd = Some(os2c(dir, &mut self.saw_nul));
    }
    pub fn uid(&mut self, id: uid_t) {
        self.uid = Some(id);
    }
    pub fn gid(&mut self, id: gid_t) {
        self.gid = Some(id);
    }

    pub fn saw_nul(&self) -> bool {
        self.saw_nul
    }
    pub fn get_argv(&self) -> &Vec<*const c_char> {
        &self.argv.0
    }

    pub fn get_program(&self) -> &CStr {
        &*self.program
    }

    #[allow(dead_code)]
    pub fn get_cwd(&self) -> &Option<CString> {
        &self.cwd
    }
    #[allow(dead_code)]
    pub fn get_uid(&self) -> Option<uid_t> {
        self.uid
    }
    #[allow(dead_code)]
    pub fn get_gid(&self) -> Option<gid_t> {
        self.gid
    }

    pub fn get_closures(&mut self) -> &mut Vec<Box<dyn FnMut() -> io::Result<()> + Send + Sync>> {
        &mut self.closures
    }

    pub unsafe fn pre_exec(&mut self, _f: Box<dyn FnMut() -> io::Result<()> + Send + Sync>) {
        // Fork() is not supported in vxWorks so no way to run the closure in the new procecss.
        unimplemented!();
    }

    pub fn stdin(&mut self, stdin: Stdio) {
        self.stdin = Some(stdin);
    }

    pub fn stdout(&mut self, stdout: Stdio) {
        self.stdout = Some(stdout);
    }

    pub fn stderr(&mut self, stderr: Stdio) {
        self.stderr = Some(stderr);
    }

    pub fn env_mut(&mut self) -> &mut CommandEnv {
        &mut self.env
    }

    pub fn capture_env(&mut self) -> Option<CStringArray> {
        let maybe_env = self.env.capture_if_changed();
        maybe_env.map(|env| construct_envp(env, &mut self.saw_nul))
    }
    #[allow(dead_code)]
    pub fn env_saw_path(&self) -> bool {
        self.env.have_changed_path()
    }

    pub fn setup_io(
        &self,
        default: Stdio,
        needs_stdin: bool,
    ) -> io::Result<(StdioPipes, ChildPipes)> {
        let null = Stdio::Null;
        let default_stdin = if needs_stdin { &default } else { &null };
        let stdin = self.stdin.as_ref().unwrap_or(default_stdin);
        let stdout = self.stdout.as_ref().unwrap_or(&default);
        let stderr = self.stderr.as_ref().unwrap_or(&default);
        let (their_stdin, our_stdin) = stdin.to_child_stdio(true)?;
        let (their_stdout, our_stdout) = stdout.to_child_stdio(false)?;
        let (their_stderr, our_stderr) = stderr.to_child_stdio(false)?;
        let ours = StdioPipes { stdin: our_stdin, stdout: our_stdout, stderr: our_stderr };
        let theirs = ChildPipes { stdin: their_stdin, stdout: their_stdout, stderr: their_stderr };
        Ok((ours, theirs))
    }
}

fn os2c(s: &OsStr, saw_nul: &mut bool) -> CString {
    CString::new(s.as_bytes()).unwrap_or_else(|_e| {
        *saw_nul = true;
        CString::new("<string-with-nul>").unwrap()
    })
}

// Helper type to manage ownership of the strings within a C-style array.
pub struct CStringArray {
    items: Vec<CString>,
    ptrs: Vec<*const c_char>,
}

impl CStringArray {
    pub fn with_capacity(capacity: usize) -> Self {
        let mut result = CStringArray {
            items: Vec::with_capacity(capacity),
            ptrs: Vec::with_capacity(capacity + 1),
        };
        result.ptrs.push(ptr::null());
        result
    }
    pub fn push(&mut self, item: CString) {
        let l = self.ptrs.len();
        self.ptrs[l - 1] = item.as_ptr();
        self.ptrs.push(ptr::null());
        self.items.push(item);
    }
    pub fn as_ptr(&self) -> *const *const c_char {
        self.ptrs.as_ptr()
    }
}

fn construct_envp(env: BTreeMap<OsString, OsString>, saw_nul: &mut bool) -> CStringArray {
    let mut result = CStringArray::with_capacity(env.len());
    for (k, v) in env {
        let mut k: OsString = k.into();

        // Reserve additional space for '=' and null terminator
        k.reserve_exact(v.len() + 2);
        k.push("=");
        k.push(&v);

        // Add the new entry into the array
        if let Ok(item) = CString::new(k.into_vec()) {
            result.push(item);
        } else {
            *saw_nul = true;
        }
    }

    result
}

impl Stdio {
    pub fn to_child_stdio(&self, readable: bool) -> io::Result<(ChildStdio, Option<AnonPipe>)> {
        match *self {
            Stdio::Inherit => Ok((ChildStdio::Inherit, None)),

            // Make sure that the source descriptors are not an stdio
            // descriptor, otherwise the order which we set the child's
            // descriptors may blow away a descriptor which we are hoping to
            // save. For example, suppose we want the child's stderr to be the
            // parent's stdout, and the child's stdout to be the parent's
            // stderr. No matter which we dup first, the second will get
            // overwritten prematurely.
            Stdio::Fd(ref fd) => {
                if fd.raw() >= 0 && fd.raw() <= libc::STDERR_FILENO {
                    Ok((ChildStdio::Owned(fd.duplicate()?), None))
                } else {
                    Ok((ChildStdio::Explicit(fd.raw()), None))
                }
            }

            Stdio::MakePipe => {
                let (reader, writer) = pipe::anon_pipe()?;
                let (ours, theirs) = if readable { (writer, reader) } else { (reader, writer) };
                Ok((ChildStdio::Owned(theirs.into_fd()), Some(ours)))
            }

            Stdio::Null => {
                let mut opts = OpenOptions::new();
                opts.read(readable);
                opts.write(!readable);
                let path = unsafe { CStr::from_ptr("/null\0".as_ptr() as *const _) };
                let fd = File::open_c(&path, &opts)?;
                Ok((ChildStdio::Owned(fd.into_fd()), None))
            }
        }
    }
}

impl From<AnonPipe> for Stdio {
    fn from(pipe: AnonPipe) -> Stdio {
        Stdio::Fd(pipe.into_fd())
    }
}

impl From<File> for Stdio {
    fn from(file: File) -> Stdio {
        Stdio::Fd(file.into_fd())
    }
}

impl ChildStdio {
    pub fn fd(&self) -> Option<c_int> {
        match *self {
            ChildStdio::Inherit => None,
            ChildStdio::Explicit(fd) => Some(fd),
            ChildStdio::Owned(ref fd) => Some(fd.raw()),
        }
    }
}

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.program != self.args[0] {
            write!(f, "[{:?}] ", self.program)?;
        }
        write!(f, "{:?}", self.args[0])?;

        for arg in &self.args[1..] {
            write!(f, " {:?}", arg)?;
        }
        Ok(())
    }
}

/// Unix exit statuses
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatus(c_int);

impl ExitStatus {
    pub fn new(status: c_int) -> ExitStatus {
        ExitStatus(status)
    }

    fn exited(&self) -> bool {
        libc::WIFEXITED(self.0)
    }

    pub fn success(&self) -> bool {
        self.code() == Some(0)
    }

    pub fn code(&self) -> Option<i32> {
        if self.exited() { Some(libc::WEXITSTATUS(self.0)) } else { None }
    }

    pub fn signal(&self) -> Option<i32> {
        if !self.exited() { Some(libc::WTERMSIG(self.0)) } else { None }
    }
}

/// Converts a raw `c_int` to a type-safe `ExitStatus` by wrapping it without copying.
impl From<c_int> for ExitStatus {
    fn from(a: c_int) -> ExitStatus {
        ExitStatus(a)
    }
}

impl fmt::Display for ExitStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(code) = self.code() {
            write!(f, "exit code: {}", code)
        } else {
            let signal = self.signal().unwrap();
            write!(f, "signal: {}", signal)
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitCode(u8);

impl ExitCode {
    pub const SUCCESS: ExitCode = ExitCode(EXIT_SUCCESS as _);
    pub const FAILURE: ExitCode = ExitCode(EXIT_FAILURE as _);

    #[inline]
    pub fn as_i32(&self) -> i32 {
        self.0 as i32
    }
}
