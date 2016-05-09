// Copyright 2014-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;
use os::unix::prelude::*;

use collections::hash_map::{HashMap, Entry};
use env;
use ffi::{OsString, OsStr, CString, CStr};
use fmt;
use io::{self, Error, ErrorKind};
use libc::{self, pid_t, c_int, gid_t, uid_t, c_char};
use mem;
use ptr;
use sys::fd::FileDesc;
use sys::fs::{File, OpenOptions};
use sys::pipe::{self, AnonPipe};
use sys::{self, cvt, cvt_r};

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
    session_leader: bool,
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
struct ChildPipes {
    stdin: ChildStdio,
    stdout: ChildStdio,
    stderr: ChildStdio,
}

enum ChildStdio {
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
            argv: vec![program.as_ptr(), 0 as *const _],
            program: program,
            args: Vec::new(),
            env: None,
            envp: None,
            cwd: None,
            uid: None,
            gid: None,
            session_leader: false,
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
        self.argv.push(0 as *const _);

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
            envp.push(0 as *const _);
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
                envp.push(0 as *const _);
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
        self.envp = Some(vec![0 as *const _]);
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
    pub fn session_leader(&mut self, session_leader: bool) {
        self.session_leader = session_leader;
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

    pub fn spawn(&mut self, default: Stdio, needs_stdin: bool)
                 -> io::Result<(Process, StdioPipes)> {
        const CLOEXEC_MSG_FOOTER: &'static [u8] = b"NOEX";

        if self.saw_nul {
            return Err(io::Error::new(ErrorKind::InvalidInput,
                                      "nul byte found in provided data"));
        }

        let (ours, theirs) = self.setup_io(default, needs_stdin)?;
        let (input, output) = sys::pipe::anon_pipe()?;

        let pid = unsafe {
            match cvt(libc::fork())? {
                0 => {
                    drop(input);
                    let err = self.do_exec(theirs);
                    let errno = err.raw_os_error().unwrap_or(libc::EINVAL) as u32;
                    let bytes = [
                        (errno >> 24) as u8,
                        (errno >> 16) as u8,
                        (errno >>  8) as u8,
                        (errno >>  0) as u8,
                        CLOEXEC_MSG_FOOTER[0], CLOEXEC_MSG_FOOTER[1],
                        CLOEXEC_MSG_FOOTER[2], CLOEXEC_MSG_FOOTER[3]
                    ];
                    // pipe I/O up to PIPE_BUF bytes should be atomic, and then
                    // we want to be sure we *don't* run at_exit destructors as
                    // we're being torn down regardless
                    assert!(output.write(&bytes).is_ok());
                    libc::_exit(1)
                }
                n => n,
            }
        };

        let mut p = Process { pid: pid, status: None };
        drop(output);
        let mut bytes = [0; 8];

        // loop to handle EINTR
        loop {
            match input.read(&mut bytes) {
                Ok(0) => return Ok((p, ours)),
                Ok(8) => {
                    assert!(combine(CLOEXEC_MSG_FOOTER) == combine(&bytes[4.. 8]),
                            "Validation on the CLOEXEC pipe failed: {:?}", bytes);
                    let errno = combine(&bytes[0.. 4]);
                    assert!(p.wait().is_ok(),
                            "wait() should either return Ok or panic");
                    return Err(Error::from_raw_os_error(errno))
                }
                Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => {
                    assert!(p.wait().is_ok(),
                            "wait() should either return Ok or panic");
                    panic!("the CLOEXEC pipe failed: {:?}", e)
                },
                Ok(..) => { // pipe I/O up to PIPE_BUF bytes should be atomic
                    assert!(p.wait().is_ok(),
                            "wait() should either return Ok or panic");
                    panic!("short read on the CLOEXEC pipe")
                }
            }
        }

        fn combine(arr: &[u8]) -> i32 {
            let a = arr[0] as u32;
            let b = arr[1] as u32;
            let c = arr[2] as u32;
            let d = arr[3] as u32;

            ((a << 24) | (b << 16) | (c << 8) | (d << 0)) as i32
        }
    }

    pub fn exec(&mut self, default: Stdio) -> io::Error {
        if self.saw_nul {
            return io::Error::new(ErrorKind::InvalidInput,
                                  "nul byte found in provided data")
        }

        match self.setup_io(default, true) {
            Ok((_, theirs)) => unsafe { self.do_exec(theirs) },
            Err(e) => e,
        }
    }

    // And at this point we've reached a special time in the life of the
    // child. The child must now be considered hamstrung and unable to
    // do anything other than syscalls really. Consider the following
    // scenario:
    //
    //      1. Thread A of process 1 grabs the malloc() mutex
    //      2. Thread B of process 1 forks(), creating thread C
    //      3. Thread C of process 2 then attempts to malloc()
    //      4. The memory of process 2 is the same as the memory of
    //         process 1, so the mutex is locked.
    //
    // This situation looks a lot like deadlock, right? It turns out
    // that this is what pthread_atfork() takes care of, which is
    // presumably implemented across platforms. The first thing that
    // threads to *before* forking is to do things like grab the malloc
    // mutex, and then after the fork they unlock it.
    //
    // Despite this information, libnative's spawn has been witnessed to
    // deadlock on both OSX and FreeBSD. I'm not entirely sure why, but
    // all collected backtraces point at malloc/free traffic in the
    // child spawned process.
    //
    // For this reason, the block of code below should contain 0
    // invocations of either malloc of free (or their related friends).
    //
    // As an example of not having malloc/free traffic, we don't close
    // this file descriptor by dropping the FileDesc (which contains an
    // allocation). Instead we just close it manually. This will never
    // have the drop glue anyway because this code never returns (the
    // child will either exec() or invoke libc::exit)
    unsafe fn do_exec(&mut self, stdio: ChildPipes) -> io::Error {
        macro_rules! t {
            ($e:expr) => (match $e {
                Ok(e) => e,
                Err(e) => return e,
            })
        }

        if let Some(fd) = stdio.stdin.fd() {
            t!(cvt_r(|| libc::dup2(fd, libc::STDIN_FILENO)));
        }
        if let Some(fd) = stdio.stdout.fd() {
            t!(cvt_r(|| libc::dup2(fd, libc::STDOUT_FILENO)));
        }
        if let Some(fd) = stdio.stderr.fd() {
            t!(cvt_r(|| libc::dup2(fd, libc::STDERR_FILENO)));
        }

        if let Some(u) = self.gid {
            t!(cvt(libc::setgid(u as gid_t)));
        }
        if let Some(u) = self.uid {
            // When dropping privileges from root, the `setgroups` call
            // will remove any extraneous groups. If we don't call this,
            // then even though our uid has dropped, we may still have
            // groups that enable us to do super-user things. This will
            // fail if we aren't root, so don't bother checking the
            // return value, this is just done as an optimistic
            // privilege dropping function.
            let _ = libc::setgroups(0, ptr::null());

            t!(cvt(libc::setuid(u as uid_t)));
        }
        if self.session_leader {
            // Don't check the error of setsid because it fails if we're the
            // process leader already. We just forked so it shouldn't return
            // error, but ignore it anyway.
            let _ = libc::setsid();
        }
        if let Some(ref cwd) = self.cwd {
            t!(cvt(libc::chdir(cwd.as_ptr())));
        }
        if let Some(ref envp) = self.envp {
            *sys::os::environ() = envp.as_ptr();
        }

        // NaCl has no signal support.
        if cfg!(not(target_os = "nacl")) {
            // Reset signal handling so the child process starts in a
            // standardized state. libstd ignores SIGPIPE, and signal-handling
            // libraries often set a mask. Child processes inherit ignored
            // signals and the signal mask from their parent, but most
            // UNIX programs do not reset these things on their own, so we
            // need to clean things up now to avoid confusing the program
            // we're about to run.
            let mut set: libc::sigset_t = mem::uninitialized();
            t!(cvt(libc::sigemptyset(&mut set)));
            t!(cvt(libc::pthread_sigmask(libc::SIG_SETMASK, &set,
                                         ptr::null_mut())));
            let ret = super::signal(libc::SIGPIPE, libc::SIG_DFL);
            if ret == libc::SIG_ERR {
                return io::Error::last_os_error()
            }
        }

        for callback in self.closures.iter_mut() {
            t!(callback());
        }

        libc::execvp(self.argv[0], self.argv.as_ptr());
        io::Error::last_os_error()
    }


    fn setup_io(&self, default: Stdio, needs_stdin: bool)
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
    fn to_child_stdio(&self, readable: bool)
                      -> io::Result<(ChildStdio, Option<AnonPipe>)> {
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
    fn fd(&self) -> Option<c_int> {
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

////////////////////////////////////////////////////////////////////////////////
// Processes
////////////////////////////////////////////////////////////////////////////////

/// Unix exit statuses
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatus(c_int);

impl ExitStatus {
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

/// The unique id of the process (this should never be negative).
pub struct Process {
    pid: pid_t,
    status: Option<ExitStatus>,
}

impl Process {
    pub fn id(&self) -> u32 {
        self.pid as u32
    }

    pub fn kill(&mut self) -> io::Result<()> {
        // If we've already waited on this process then the pid can be recycled
        // and used for another process, and we probably shouldn't be killing
        // random processes, so just return an error.
        if self.status.is_some() {
            Err(Error::new(ErrorKind::InvalidInput,
                           "invalid argument: can't kill an exited process"))
        } else {
            cvt(unsafe { libc::kill(self.pid, libc::SIGKILL) }).map(|_| ())
        }
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        if let Some(status) = self.status {
            return Ok(status)
        }
        let mut status = 0 as c_int;
        cvt_r(|| unsafe { libc::waitpid(self.pid, &mut status, 0) })?;
        self.status = Some(ExitStatus(status));
        Ok(ExitStatus(status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prelude::v1::*;

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
