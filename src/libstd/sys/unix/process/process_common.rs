use crate::os::unix::prelude::*;

use crate::ffi::{OsString, OsStr, CString, CStr};
use crate::fmt;
use crate::io;
use crate::ptr;
use crate::sys::fd::FileDesc;
use crate::sys::fs::{File, OpenOptions};
use crate::sys::pipe::{self, AnonPipe};
use crate::sys_common::process::{CommandEnv, DefaultEnvKey};
use crate::collections::BTreeMap;

use libc::{c_int, gid_t, uid_t, c_char, EXIT_SUCCESS, EXIT_FAILURE};

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
    env: CommandEnv<DefaultEnvKey>,

    cwd: Option<CString>,
    uid: Option<uid_t>,
    gid: Option<gid_t>,
    saw_nul: bool,
    closures: Vec<Box<dyn FnMut() -> io::Result<()> + Send + Sync>>,
    stdin: Option<Stdio>,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,
}

// Create a new type for argv, so that we can make it `Send`
struct Argv(Vec<*const c_char>);

// It is safe to make Argv Send, because it contains pointers to memory owned by `Command.args`
unsafe impl Send for Argv {}

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
            program,
            args: Vec::new(),
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

    pub fn arg(&mut self, arg: &OsStr) {
        // Overwrite the trailing NULL pointer in `argv` and then add a new null
        // pointer.
        let arg = os2c(arg, &mut self.saw_nul);
        self.argv.0[self.args.len() + 1] = arg.as_ptr();
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

    pub unsafe fn pre_exec(
        &mut self,
        f: Box<dyn FnMut() -> io::Result<()> + Send + Sync>,
    ) {
        self.closures.push(f);
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

    pub fn env_mut(&mut self) -> &mut CommandEnv<DefaultEnvKey> {
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

    pub fn setup_io(&self, default: Stdio, needs_stdin: bool)
                -> io::Result<(StdioPipes, ChildPipes)> {
        let null = Stdio::Null;
        let default_stdin = if needs_stdin {&default} else {&null};
        let stdin = self.stdin.as_ref().unwrap_or(default_stdin);
        let stdout = self.stdout.as_ref().unwrap_or(&default);
        let stderr = self.stderr.as_ref().unwrap_or(&default);
        let (their_stdin, our_stdin) = stdin.to_child_stdio(true)?;
        let (their_stdout, our_stdout) = stdout.to_child_stdio(false)?;
        let (their_stderr, our_stderr) = stderr.to_child_stdio(false)?;
        let ours = StdioPipes {
            stdin: our_stdin,
            stdout: our_stdout,
            stderr: our_stderr,
        };
        let theirs = ChildPipes {
            stdin: their_stdin,
            stdout: their_stdout,
            stderr: their_stderr,
        };
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
    ptrs: Vec<*const c_char>
}

impl CStringArray {
    pub fn with_capacity(capacity: usize) -> Self {
        let mut result = CStringArray {
            items: Vec::with_capacity(capacity),
            ptrs: Vec::with_capacity(capacity+1)
        };
        result.ptrs.push(ptr::null());
        result
    }
    pub fn push(&mut self, item: CString) {
        let l = self.ptrs.len();
        self.ptrs[l-1] = item.as_ptr();
        self.ptrs.push(ptr::null());
        self.items.push(item);
    }
    pub fn as_ptr(&self) -> *const *const c_char {
        self.ptrs.as_ptr()
    }
}

fn construct_envp(env: BTreeMap<DefaultEnvKey, OsString>, saw_nul: &mut bool) -> CStringArray {
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
    pub fn to_child_stdio(&self, readable: bool)
                      -> io::Result<(ChildStdio, Option<AnonPipe>)> {
        match *self {
            Stdio::Inherit => {
                Ok((ChildStdio::Inherit, None))
            },

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
                let (ours, theirs) = if readable {
                    (writer, reader)
                } else {
                    (reader, writer)
                };
                Ok((ChildStdio::Owned(theirs.into_fd()), Some(ours)))
            }

            Stdio::Null => {
                let mut opts = OpenOptions::new();
                opts.read(readable);
                opts.write(!readable);
                let path = unsafe {
                    CStr::from_ptr("/dev/null\0".as_ptr() as *const _)
                };
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
        write!(f, "{:?}", self.program)?;
        for arg in &self.args {
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
        unsafe { libc::WIFEXITED(self.0) }
    }

    pub fn success(&self) -> bool {
        self.code() == Some(0)
    }

    pub fn code(&self) -> Option<i32> {
        if self.exited() {
            Some(unsafe { libc::WEXITSTATUS(self.0) })
        } else {
            None
        }
    }

    pub fn signal(&self) -> Option<i32> {
        if !self.exited() {
            Some(unsafe { libc::WTERMSIG(self.0) })
        } else {
            None
        }
    }
}

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

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use super::*;

    use crate::ffi::OsStr;
    use crate::mem;
    use crate::ptr;
    use crate::sys::cvt;

    macro_rules! t {
        ($e:expr) => {
            match $e {
                Ok(t) => t,
                Err(e) => panic!("received error for `{}`: {}", stringify!($e), e),
            }
        }
    }

    // Android with api less than 21 define sig* functions inline, so it is not
    // available for dynamic link. Implementing sigemptyset and sigaddset allow us
    // to support older Android version (independent of libc version).
    // The following implementations are based on https://git.io/vSkNf

    #[cfg(not(target_os = "android"))]
    extern {
        #[cfg_attr(target_os = "netbsd", link_name = "__sigemptyset14")]
        fn sigemptyset(set: *mut libc::sigset_t) -> libc::c_int;

        #[cfg_attr(target_os = "netbsd", link_name = "__sigaddset14")]
        fn sigaddset(set: *mut libc::sigset_t, signum: libc::c_int) -> libc::c_int;
    }

    #[cfg(target_os = "android")]
    unsafe fn sigemptyset(set: *mut libc::sigset_t) -> libc::c_int {
        set.write_bytes(0u8, 1);
        return 0;
    }

    #[cfg(target_os = "android")]
    unsafe fn sigaddset(set: *mut libc::sigset_t, signum: libc::c_int) -> libc::c_int {
        use crate::slice;

        let raw = slice::from_raw_parts_mut(set as *mut u8, mem::size_of::<libc::sigset_t>());
        let bit = (signum - 1) as usize;
        raw[bit / 8] |= 1 << (bit % 8);
        return 0;
    }

    // See #14232 for more information, but it appears that signal delivery to a
    // newly spawned process may just be raced in the macOS, so to prevent this
    // test from being flaky we ignore it on macOS.
    #[test]
    #[cfg_attr(target_os = "macos", ignore)]
    // When run under our current QEMU emulation test suite this test fails,
    // although the reason isn't very clear as to why. For now this test is
    // ignored there.
    #[cfg_attr(target_arch = "arm", ignore)]
    #[cfg_attr(target_arch = "aarch64", ignore)]
    fn test_process_mask() {
        unsafe {
            // Test to make sure that a signal mask does not get inherited.
            let mut cmd = Command::new(OsStr::new("cat"));

            let mut set = mem::MaybeUninit::<libc::sigset_t>::uninit();
            let mut old_set = mem::MaybeUninit::<libc::sigset_t>::uninit();
            t!(cvt(sigemptyset(set.as_mut_ptr())));
            t!(cvt(sigaddset(set.as_mut_ptr(), libc::SIGINT)));
            t!(cvt(libc::pthread_sigmask(libc::SIG_SETMASK, set.as_ptr(), old_set.as_mut_ptr())));

            cmd.stdin(Stdio::MakePipe);
            cmd.stdout(Stdio::MakePipe);

            let (mut cat, mut pipes) = t!(cmd.spawn(Stdio::Null, true));
            let stdin_write = pipes.stdin.take().unwrap();
            let stdout_read = pipes.stdout.take().unwrap();

            t!(cvt(libc::pthread_sigmask(libc::SIG_SETMASK, old_set.as_ptr(),
                                         ptr::null_mut())));

            t!(cvt(libc::kill(cat.id() as libc::pid_t, libc::SIGINT)));
            // We need to wait until SIGINT is definitely delivered. The
            // easiest way is to write something to cat, and try to read it
            // back: if SIGINT is unmasked, it'll get delivered when cat is
            // next scheduled.
            let _ = stdin_write.write(b"Hello");
            drop(stdin_write);

            // Either EOF or failure (EPIPE) is okay.
            let mut buf = [0; 5];
            if let Ok(ret) = stdout_read.read(&mut buf) {
                assert_eq!(ret, 0);
            }

            t!(cat.wait());
        }
    }
}
