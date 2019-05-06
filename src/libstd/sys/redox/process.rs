use crate::env::{self, split_paths};
use crate::ffi::{CStr, OsStr};
use crate::fmt;
use crate::fs::File;
use crate::io::{self, prelude::*, BufReader, Error, ErrorKind, SeekFrom};
use crate::os::unix::ffi::OsStrExt;
use crate::path::{Path, PathBuf};
use crate::ptr;
use crate::sys::ext::fs::MetadataExt;
use crate::sys::ext::io::AsRawFd;
use crate::sys::fd::FileDesc;
use crate::sys::fs::{File as SysFile, OpenOptions};
use crate::sys::os::{ENV_LOCK, environ};
use crate::sys::pipe::{self, AnonPipe};
use crate::sys::{cvt, syscall};
use crate::sys_common::process::{CommandEnv, DefaultEnvKey};

use libc::{EXIT_SUCCESS, EXIT_FAILURE};

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
    program: String,
    args: Vec<String>,
    env: CommandEnv<DefaultEnvKey>,

    cwd: Option<String>,
    uid: Option<u32>,
    gid: Option<u32>,
    saw_nul: bool,
    closures: Vec<Box<dyn FnMut() -> io::Result<()> + Send + Sync>>,
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
    Explicit(usize),
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
        Command {
            program: program.to_str().unwrap().to_owned(),
            args: Vec::new(),
            env: Default::default(),
            cwd: None,
            uid: None,
            gid: None,
            saw_nul: false,
            closures: Vec::new(),
            stdin: None,
            stdout: None,
            stderr: None,
        }
    }

    pub fn arg(&mut self, arg: &OsStr) {
        self.args.push(arg.to_str().unwrap().to_owned());
    }

    pub fn env_mut(&mut self) -> &mut CommandEnv<DefaultEnvKey> {
        &mut self.env
    }

    pub fn cwd(&mut self, dir: &OsStr) {
        self.cwd = Some(dir.to_str().unwrap().to_owned());
    }
    pub fn uid(&mut self, id: u32) {
        self.uid = Some(id);
    }
    pub fn gid(&mut self, id: u32) {
        self.gid = Some(id);
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

    pub fn spawn(&mut self, default: Stdio, needs_stdin: bool)
                 -> io::Result<(Process, StdioPipes)> {
         const CLOEXEC_MSG_FOOTER: &[u8] = b"NOEX";

         if self.saw_nul {
             return Err(io::Error::new(ErrorKind::InvalidInput,
                                       "nul byte found in provided data"));
         }

         let (ours, theirs) = self.setup_io(default, needs_stdin)?;
         let (input, output) = pipe::anon_pipe()?;

         let pid = unsafe {
             match cvt(syscall::clone(0))? {
                 0 => {
                     drop(input);
                     let Err(err) = self.do_exec(theirs);
                     let errno = err.raw_os_error().unwrap_or(syscall::EINVAL) as u32;
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
                     let _ = syscall::exit(1);
                     panic!("failed to exit");
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
            Ok((_, theirs)) => unsafe {
                let Err(e) = self.do_exec(theirs);
                e
            },
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
    // deadlock on both macOS and FreeBSD. I'm not entirely sure why, but
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
    // child will either exec() or invoke syscall::exit)
    unsafe fn do_exec(&mut self, stdio: ChildPipes) -> Result<!, io::Error> {
        if let Some(fd) = stdio.stderr.fd() {
            cvt(syscall::dup2(fd, 2, &[]))?;
            let mut flags = cvt(syscall::fcntl(2, syscall::F_GETFD, 0))?;
            flags &= ! syscall::O_CLOEXEC;
            cvt(syscall::fcntl(2, syscall::F_SETFD, flags))?;
        }
        if let Some(fd) = stdio.stdout.fd() {
            cvt(syscall::dup2(fd, 1, &[]))?;
            let mut flags = cvt(syscall::fcntl(1, syscall::F_GETFD, 0))?;
            flags &= ! syscall::O_CLOEXEC;
            cvt(syscall::fcntl(1, syscall::F_SETFD, flags))?;
        }
        if let Some(fd) = stdio.stdin.fd() {
            cvt(syscall::dup2(fd, 0, &[]))?;
            let mut flags = cvt(syscall::fcntl(0, syscall::F_GETFD, 0))?;
            flags &= ! syscall::O_CLOEXEC;
            cvt(syscall::fcntl(0, syscall::F_SETFD, flags))?;
        }

        if let Some(g) = self.gid {
            cvt(syscall::setregid(g as usize, g as usize))?;
        }
        if let Some(u) = self.uid {
            cvt(syscall::setreuid(u as usize, u as usize))?;
        }
        if let Some(ref cwd) = self.cwd {
            cvt(syscall::chdir(cwd))?;
        }

        for callback in self.closures.iter_mut() {
            callback()?;
        }

        self.env.apply();

        let program = if self.program.contains(':') || self.program.contains('/') {
            Some(PathBuf::from(&self.program))
        } else if let Ok(path_env) = env::var("PATH") {
            let mut program = None;
            for mut path in split_paths(&path_env) {
                path.push(&self.program);
                if path.exists() {
                    program = Some(path);
                    break;
                }
            }
            program
        } else {
            None
        };

        let mut file = if let Some(program) = program {
            File::open(program.as_os_str())?
        } else {
            return Err(io::Error::from_raw_os_error(syscall::ENOENT));
        };

        // Push all the arguments
        let mut args: Vec<[usize; 2]> = Vec::with_capacity(1 + self.args.len());

        let interpreter = {
            let mut reader = BufReader::new(&file);

            let mut shebang = [0; 2];
            let mut read = 0;
            loop {
                match reader.read(&mut shebang[read..])? {
                    0 => break,
                    n => read += n,
                }
            }

            if &shebang == b"#!" {
                // This is an interpreted script.
                // First of all, since we'll be passing another file to
                // fexec(), we need to manually check that we have permission
                // to execute this file:
                let uid = cvt(syscall::getuid())?;
                let gid = cvt(syscall::getgid())?;
                let meta = file.metadata()?;

                let mode = if uid == meta.uid() as usize {
                    meta.mode() >> 3*2 & 0o7
                } else if gid == meta.gid() as usize {
                    meta.mode() >> 3*1 & 0o7
                } else {
                    meta.mode() & 0o7
                };
                if mode & 1 == 0 {
                    return Err(io::Error::from_raw_os_error(syscall::EPERM));
                }

                // Second of all, we need to actually read which interpreter it wants
                let mut interpreter = Vec::new();
                reader.read_until(b'\n', &mut interpreter)?;
                // Pop one trailing newline, if any
                if interpreter.ends_with(&[b'\n']) {
                    interpreter.pop().unwrap();
                }

                // FIXME: Here we could just reassign `file` directly, if it
                // wasn't for lexical lifetimes. Remove the whole `let
                // interpreter = { ... };` hack once NLL lands.
                // NOTE: Although DO REMEMBER to make sure the interpreter path
                // still lives long enough to reach fexec.
                Some(interpreter)
            } else {
                None
            }
        };
        if let Some(ref interpreter) = interpreter {
            let path: &OsStr = OsStr::from_bytes(&interpreter);
            file = File::open(path)?;

            args.push([interpreter.as_ptr() as usize, interpreter.len()]);
        } else {
            file.seek(SeekFrom::Start(0))?;
        }

        args.push([self.program.as_ptr() as usize, self.program.len()]);
        args.extend(self.args.iter().map(|arg| [arg.as_ptr() as usize, arg.len()]));

        // Push all the variables
        let mut vars: Vec<[usize; 2]> = Vec::new();
        {
            let _guard = ENV_LOCK.lock();
            let mut environ = *environ();
            while *environ != ptr::null() {
                let var = CStr::from_ptr(*environ).to_bytes();
                vars.push([var.as_ptr() as usize, var.len()]);
                environ = environ.offset(1);
            }
        }

        if let Err(err) = syscall::fexec(file.as_raw_fd(), &args, &vars) {
            Err(io::Error::from_raw_os_error(err.errno as i32))
        } else {
            panic!("return from exec without err");
        }
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
                if fd.raw() <= 2 {
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
                let fd = SysFile::open(Path::new("null:"), &opts)?;
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

impl From<SysFile> for Stdio {
    fn from(file: SysFile) -> Stdio {
        Stdio::Fd(file.into_fd())
    }
}

impl ChildStdio {
    fn fd(&self) -> Option<usize> {
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

////////////////////////////////////////////////////////////////////////////////
// Processes
////////////////////////////////////////////////////////////////////////////////

/// Unix exit statuses
#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub struct ExitStatus(i32);

impl ExitStatus {
    fn exited(&self) -> bool {
        self.0 & 0x7F == 0
    }

    pub fn success(&self) -> bool {
        self.code() == Some(0)
    }

    pub fn code(&self) -> Option<i32> {
        if self.exited() {
            Some((self.0 >> 8) & 0xFF)
        } else {
            None
        }
    }

    pub fn signal(&self) -> Option<i32> {
        if !self.exited() {
            Some(self.0 & 0x7F)
        } else {
            None
        }
    }
}

impl From<i32> for ExitStatus {
    fn from(a: i32) -> ExitStatus {
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

    pub fn as_i32(&self) -> i32 {
        self.0 as i32
    }
}

/// The unique ID of the process (this should never be negative).
pub struct Process {
    pid: usize,
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
            cvt(syscall::kill(self.pid, syscall::SIGKILL))?;
            Ok(())
        }
    }

    pub fn wait(&mut self) -> io::Result<ExitStatus> {
        if let Some(status) = self.status {
            return Ok(status)
        }
        let mut status = 0;
        cvt(syscall::waitpid(self.pid, &mut status, 0))?;
        self.status = Some(ExitStatus(status as i32));
        Ok(ExitStatus(status as i32))
    }

    pub fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        if let Some(status) = self.status {
            return Ok(Some(status))
        }
        let mut status = 0;
        let pid = cvt(syscall::waitpid(self.pid, &mut status, syscall::WNOHANG))?;
        if pid == 0 {
            Ok(None)
        } else {
            self.status = Some(ExitStatus(status as i32));
            Ok(Some(ExitStatus(status as i32)))
        }
    }
}
