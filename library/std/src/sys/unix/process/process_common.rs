#[cfg(all(test, not(target_os = "emscripten")))]
mod tests;

use crate::os::unix::prelude::*;

use crate::collections::BTreeMap;
use crate::ffi::{CStr, CString, OsStr, OsString};
use crate::fmt;
use crate::io;
use crate::path::Path;
use crate::ptr;
use crate::sys::fd::FileDesc;
use crate::sys::fs::File;
use crate::sys::pipe::{self, AnonPipe};
use crate::sys_common::process::{CommandEnv, CommandEnvs};
use crate::sys_common::IntoInner;

#[cfg(not(target_os = "fuchsia"))]
use crate::sys::fs::OpenOptions;

use libc::{c_char, c_int, gid_t, uid_t, EXIT_FAILURE, EXIT_SUCCESS};

cfg_if::cfg_if! {
    if #[cfg(target_os = "fuchsia")] {
        // fuchsia doesn't have /dev/null
    } else if #[cfg(target_os = "redox")] {
        const DEV_NULL: &str = "null:\0";
    } else if #[cfg(target_os = "vxworks")] {
        const DEV_NULL: &str = "/null\0";
    } else {
        const DEV_NULL: &str = "/dev/null\0";
    }
}

// Android with api less than 21 define sig* functions inline, so it is not
// available for dynamic link. Implementing sigemptyset and sigaddset allow us
// to support older Android version (independent of libc version).
// The following implementations are based on https://git.io/vSkNf
cfg_if::cfg_if! {
    if #[cfg(target_os = "android")] {
        pub unsafe fn sigemptyset(set: *mut libc::sigset_t) -> libc::c_int {
            set.write_bytes(0u8, 1);
            return 0;
        }
        #[allow(dead_code)]
        pub unsafe fn sigaddset(set: *mut libc::sigset_t, signum: libc::c_int) -> libc::c_int {
            use crate::{slice, mem};

            let raw = slice::from_raw_parts_mut(set as *mut u8, mem::size_of::<libc::sigset_t>());
            let bit = (signum - 1) as usize;
            raw[bit / 8] |= 1 << (bit % 8);
            return 0;
        }
    } else {
        pub use libc::{sigemptyset, sigaddset};
    }
}

////////////////////////////////////////////////////////////////////////////////
// Command
////////////////////////////////////////////////////////////////////////////////

pub struct Command {
    program: CString,
    args: Vec<CString>,
    /// Exactly what will be passed to `execvp`.
    ///
    /// First element is a pointer to `program`, followed by pointers to
    /// `args`, followed by a `null`. Be careful when modifying `program` or
    /// `args` to properly update this as well.
    argv: Argv,
    env: CommandEnv,

    cwd: Option<CString>,
    uid: Option<uid_t>,
    gid: Option<gid_t>,
    saw_nul: bool,
    closures: Vec<Box<dyn FnMut() -> io::Result<()> + Send + Sync>>,
    groups: Option<Box<[gid_t]>>,
    stdin: Option<Stdio>,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,
    #[cfg(target_os = "linux")]
    create_pidfd: bool,
}

// Create a new type for argv, so that we can make it `Send` and `Sync`
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

    // On Fuchsia, null stdio is the default, so we simply don't specify
    // any actions at the time of spawning.
    #[cfg(target_os = "fuchsia")]
    Null,
}

pub enum Stdio {
    Inherit,
    Null,
    MakePipe,
    Fd(FileDesc),
}

impl Command {
    #[cfg(not(target_os = "linux"))]
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
            groups: None,
            stdin: None,
            stdout: None,
            stderr: None,
        }
    }

    #[cfg(target_os = "linux")]
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
            groups: None,
            stdin: None,
            stdout: None,
            stderr: None,
            create_pidfd: false,
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
        // Overwrite the trailing null pointer in `argv` and then add a new null
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
    pub fn groups(&mut self, groups: &[gid_t]) {
        self.groups = Some(Box::from(groups));
    }

    #[cfg(target_os = "linux")]
    pub fn create_pidfd(&mut self, val: bool) {
        self.create_pidfd = val;
    }

    #[cfg(not(target_os = "linux"))]
    #[allow(dead_code)]
    pub fn get_create_pidfd(&self) -> bool {
        false
    }

    #[cfg(target_os = "linux")]
    pub fn get_create_pidfd(&self) -> bool {
        self.create_pidfd
    }

    pub fn saw_nul(&self) -> bool {
        self.saw_nul
    }

    pub fn get_program(&self) -> &OsStr {
        OsStr::from_bytes(self.program.as_bytes())
    }

    pub fn get_args(&self) -> CommandArgs<'_> {
        let mut iter = self.args.iter();
        iter.next();
        CommandArgs { iter }
    }

    pub fn get_envs(&self) -> CommandEnvs<'_> {
        self.env.iter()
    }

    pub fn get_current_dir(&self) -> Option<&Path> {
        self.cwd.as_ref().map(|cs| Path::new(OsStr::from_bytes(cs.as_bytes())))
    }

    pub fn get_argv(&self) -> &Vec<*const c_char> {
        &self.argv.0
    }

    pub fn get_program_cstr(&self) -> &CStr {
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
    #[allow(dead_code)]
    pub fn get_groups(&self) -> Option<&[gid_t]> {
        self.groups.as_deref()
    }

    pub fn get_closures(&mut self) -> &mut Vec<Box<dyn FnMut() -> io::Result<()> + Send + Sync>> {
        &mut self.closures
    }

    pub unsafe fn pre_exec(&mut self, f: Box<dyn FnMut() -> io::Result<()> + Send + Sync>) {
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

    #[allow(dead_code)]
    pub fn program_is_path(&self) -> bool {
        self.program.to_bytes().contains(&b'/')
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
    for (mut k, v) in env {
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
                if fd.as_raw_fd() >= 0 && fd.as_raw_fd() <= libc::STDERR_FILENO {
                    Ok((ChildStdio::Owned(fd.duplicate()?), None))
                } else {
                    Ok((ChildStdio::Explicit(fd.as_raw_fd()), None))
                }
            }

            Stdio::MakePipe => {
                let (reader, writer) = pipe::anon_pipe()?;
                let (ours, theirs) = if readable { (writer, reader) } else { (reader, writer) };
                Ok((ChildStdio::Owned(theirs.into_inner()), Some(ours)))
            }

            #[cfg(not(target_os = "fuchsia"))]
            Stdio::Null => {
                let mut opts = OpenOptions::new();
                opts.read(readable);
                opts.write(!readable);
                let path = unsafe { CStr::from_ptr(DEV_NULL.as_ptr() as *const _) };
                let fd = File::open_c(&path, &opts)?;
                Ok((ChildStdio::Owned(fd.into_inner()), None))
            }

            #[cfg(target_os = "fuchsia")]
            Stdio::Null => Ok((ChildStdio::Null, None)),
        }
    }
}

impl From<AnonPipe> for Stdio {
    fn from(pipe: AnonPipe) -> Stdio {
        Stdio::Fd(pipe.into_inner())
    }
}

impl From<File> for Stdio {
    fn from(file: File) -> Stdio {
        Stdio::Fd(file.into_inner())
    }
}

impl ChildStdio {
    pub fn fd(&self) -> Option<c_int> {
        match *self {
            ChildStdio::Inherit => None,
            ChildStdio::Explicit(fd) => Some(fd),
            ChildStdio::Owned(ref fd) => Some(fd.as_raw_fd()),

            #[cfg(target_os = "fuchsia")]
            ChildStdio::Null => None,
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

pub struct CommandArgs<'a> {
    iter: crate::slice::Iter<'a, CString>,
}

impl<'a> Iterator for CommandArgs<'a> {
    type Item = &'a OsStr;
    fn next(&mut self) -> Option<&'a OsStr> {
        self.iter.next().map(|cs| OsStr::from_bytes(cs.as_bytes()))
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a> ExactSizeIterator for CommandArgs<'a> {
    fn len(&self) -> usize {
        self.iter.len()
    }
    fn is_empty(&self) -> bool {
        self.iter.is_empty()
    }
}

impl<'a> fmt::Debug for CommandArgs<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.iter.clone()).finish()
    }
}
