// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use os::unix::prelude::*;

use collections::hash_map::{HashMap, Entry};
use env;
use ffi::{OsString, OsStr, CString, CStr};
use fmt;
use io;
use libc::{self, c_int, gid_t, uid_t, c_char};
use ptr;
use sys::fd::FileDesc;
use sys::fs::{File, OpenOptions};
use sys::pipe::{self, AnonPipe};

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
    env: Option<HashMap<OsString, (usize, CString)>>,
    argv: Vec<*const c_char>,
    envp: Option<Vec<*const c_char>>,

    cwd: Option<CString>,
    uid: Option<uid_t>,
    gid: Option<gid_t>,
    saw_nul: bool,
    closures: Vec<Box<FnMut() -> io::Result<()> + Send + Sync>>,
    stdin: Option<Stdio>,
    stdout: Option<Stdio>,
    stderr: Option<Stdio>,
}

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
            argv: vec![program.as_ptr(), ptr::null()],
            program: program,
            args: Vec::new(),
            env: None,
            envp: None,
            cwd: None,
            uid: None,
            gid: None,
            saw_nul: saw_nul,
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
        self.argv[self.args.len() + 1] = arg.as_ptr();
        self.argv.push(ptr::null());

        // Also make sure we keep track of the owned value to schedule a
        // destructor for this memory.
        self.args.push(arg);
    }

    fn init_env_map(&mut self) -> (&mut HashMap<OsString, (usize, CString)>,
                                   &mut Vec<*const c_char>) {
        if self.env.is_none() {
            let mut map = HashMap::new();
            let mut envp = Vec::new();
            for (k, v) in env::vars_os() {
                let s = pair_to_key(&k, &v, &mut self.saw_nul);
                envp.push(s.as_ptr());
                map.insert(k, (envp.len() - 1, s));
            }
            envp.push(ptr::null());
            self.env = Some(map);
            self.envp = Some(envp);
        }
        (self.env.as_mut().unwrap(), self.envp.as_mut().unwrap())
    }

    pub fn env(&mut self, key: &OsStr, val: &OsStr) {
        let new_key = pair_to_key(key, val, &mut self.saw_nul);
        let (map, envp) = self.init_env_map();

        // If `key` is already present then we just update `envp` in place
        // (and store the owned value), but if it's not there we override the
        // trailing NULL pointer, add a new NULL pointer, and store where we
        // were located.
        match map.entry(key.to_owned()) {
            Entry::Occupied(mut e) => {
                let (i, ref mut s) = *e.get_mut();
                envp[i] = new_key.as_ptr();
                *s = new_key;
            }
            Entry::Vacant(e) => {
                let len = envp.len();
                envp[len - 1] = new_key.as_ptr();
                envp.push(ptr::null());
                e.insert((len - 1, new_key));
            }
        }
    }

    pub fn env_remove(&mut self, key: &OsStr) {
        let (map, envp) = self.init_env_map();

        // If we actually ended up removing a key, then we need to update the
        // position of all keys that come after us in `envp` because they're all
        // one element sooner now.
        if let Some((i, _)) = map.remove(key) {
            envp.remove(i);

            for (_, &mut (ref mut j, _)) in map.iter_mut() {
                if *j >= i {
                    *j -= 1;
                }
            }
        }
    }

    pub fn env_clear(&mut self) {
        self.env = Some(HashMap::new());
        self.envp = Some(vec![ptr::null()]);
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
    pub fn get_envp(&self) -> &Option<Vec<*const c_char>> {
        &self.envp
    }
    pub fn get_argv(&self) -> &Vec<*const c_char> {
        &self.argv
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

    pub fn get_closures(&mut self) -> &mut Vec<Box<FnMut() -> io::Result<()> + Send + Sync>> {
        &mut self.closures
    }

    pub fn before_exec(&mut self,
                       f: Box<FnMut() -> io::Result<()> + Send + Sync>) {
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

impl ChildStdio {
    pub fn fd(&self) -> Option<c_int> {
        match *self {
            ChildStdio::Inherit => None,
            ChildStdio::Explicit(fd) => Some(fd),
            ChildStdio::Owned(ref fd) => Some(fd.raw()),
        }
    }
}

fn pair_to_key(key: &OsStr, value: &OsStr, saw_nul: &mut bool) -> CString {
    let (key, value) = (key.as_bytes(), value.as_bytes());
    let mut v = Vec::with_capacity(key.len() + value.len() + 1);
    v.extend(key);
    v.push(b'=');
    v.extend(value);
    CString::new(v).unwrap_or_else(|_e| {
        *saw_nul = true;
        CString::new("foo=bar").unwrap()
    })
}

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
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
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(code) = self.code() {
            write!(f, "exit code: {}", code)
        } else {
            let signal = self.signal().unwrap();
            write!(f, "signal: {}", signal)
        }
    }
}

#[cfg(all(test, not(target_os = "emscripten")))]
mod tests {
    use super::*;

    use ffi::OsStr;
    use mem;
    use ptr;
    use libc;
    use sys::cvt;

    macro_rules! t {
        ($e:expr) => {
            match $e {
                Ok(t) => t,
                Err(e) => panic!("received error for `{}`: {}", stringify!($e), e),
            }
        }
    }

    #[cfg(not(target_os = "android"))]
    extern {
        #[cfg_attr(target_os = "netbsd", link_name = "__sigaddset14")]
        fn sigaddset(set: *mut libc::sigset_t, signum: libc::c_int) -> libc::c_int;
    }

    #[cfg(target_os = "android")]
    unsafe fn sigaddset(set: *mut libc::sigset_t, signum: libc::c_int) -> libc::c_int {
        use slice;

        let raw = slice::from_raw_parts_mut(set as *mut u8, mem::size_of::<libc::sigset_t>());
        let bit = (signum - 1) as usize;
        raw[bit / 8] |= 1 << (bit % 8);
        return 0;
    }

    // See #14232 for more information, but it appears that signal delivery to a
    // newly spawned process may just be raced in the OSX, so to prevent this
    // test from being flaky we ignore it on OSX.
    #[test]
    #[cfg_attr(target_os = "macos", ignore)]
    #[cfg_attr(target_os = "nacl", ignore)] // no signals on NaCl.
    fn test_process_mask() {
        unsafe {
            // Test to make sure that a signal mask does not get inherited.
            let mut cmd = Command::new(OsStr::new("cat"));

            let mut set: libc::sigset_t = mem::uninitialized();
            let mut old_set: libc::sigset_t = mem::uninitialized();
            t!(cvt(libc::sigemptyset(&mut set)));
            t!(cvt(sigaddset(&mut set, libc::SIGINT)));
            t!(cvt(libc::pthread_sigmask(libc::SIG_SETMASK, &set, &mut old_set)));

            cmd.stdin(Stdio::MakePipe);
            cmd.stdout(Stdio::MakePipe);

            let (mut cat, mut pipes) = t!(cmd.spawn(Stdio::Null, true));
            let stdin_write = pipes.stdin.take().unwrap();
            let stdout_read = pipes.stdout.take().unwrap();

            t!(cvt(libc::pthread_sigmask(libc::SIG_SETMASK, &old_set,
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
                assert!(ret == 0);
            }

            t!(cat.wait());
        }
    }
}
