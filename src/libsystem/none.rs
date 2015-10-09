use io;

#[inline(never)]
fn unimplemented() -> io::Error {
    io::Error::new(io::ErrorKind::Other, "unimplemented")
}

pub mod thread_local {
    use ptr::null_mut;
    use cell::UnsafeCell;
    use marker::Sync;
    use sys::thread_local as sys;

    pub struct StaticOsKey(UnsafeCell<*mut u8>, Option<unsafe extern fn(*mut u8)>);
    pub struct OsKey(UnsafeCell<*mut u8>, Option<unsafe extern fn(*mut u8)>);
    pub struct Key<T>(UnsafeCell<Option<T>>);
    unsafe impl<T> Sync for Key<T> { }
    unsafe impl Sync for OsKey { }
    unsafe impl Sync for StaticOsKey { }

    impl<T> sys::Key<T> for Key<T> {
        unsafe fn get(&'static self) -> Option<&'static UnsafeCell<Option<T>>> {
            Some(&self.0)
        }
    }

    impl<T> Key<T> {
        pub const fn new() -> Self { Key(UnsafeCell::new(None)) }
    }

    impl sys::StaticOsKey for StaticOsKey {
        const INIT: StaticOsKey = StaticOsKey(UnsafeCell::new(null_mut()), None);

        unsafe fn get(&self) -> *mut u8 { *self.0.get() }
        unsafe fn set(&self, val: *mut u8) { *self.0.get() = val }
        unsafe fn destroy(&self) {
            if let Some(dtor) = self.1 {
                dtor(self.get())
            }
        }
    }

    impl sys::OsKey for OsKey {
        fn new(dtor: Option<unsafe extern fn(*mut u8)>) -> Self { OsKey(UnsafeCell::new(null_mut()), dtor) }
        fn get(&self) -> *mut u8 { unsafe { *self.0.get() } }
        fn set(&self, val: *mut u8) { unsafe { *self.0.get() = val } }
    }

    impl StaticOsKey {
        pub const fn new(dtor: Option<unsafe extern fn(*mut u8)>) -> Self { StaticOsKey(UnsafeCell::new(null_mut()), dtor) }
    }
}

pub mod unwind {
    use any::Any;
    use boxed::Box;
    use fmt;

    extern {
        fn rust_none_begin_unwind_fmt(msg: &fmt::Arguments, file_line: &(&'static str, u32)) -> !;
        fn rust_none_begin_unwind(msg: &Any, file_line: &(&'static str, u32)) -> !;
        fn rust_none_panicking() -> bool;
        fn rust_none_try(f: unsafe extern fn (*mut u8), data: *mut u8) -> Result<(), Box<Any + Send>>;
    }

    #[inline(never)] #[cold]
    pub fn begin_unwind_fmt(msg: fmt::Arguments, file_line: &(&'static str, u32)) -> ! {
        unsafe { rust_none_begin_unwind_fmt(&msg, file_line) }
    }

    #[inline(always)]
    pub fn begin_unwind<M: Any + Send>(msg: M, file_line: &(&'static str, u32)) -> ! {
        unsafe { rust_none_begin_unwind(&msg, file_line) }
    }

    #[inline(always)]
    pub fn panicking() -> bool {
        unsafe { rust_none_panicking() }
    }

    pub unsafe fn try<F: FnOnce()>(f: F) -> Result<(), Box<Any + Send>> {
        let mut f = Some(f);
        unsafe extern fn __call<F: FnOnce()>(f: *mut u8) {
            let f = f as *mut Option<F>;
            (*f).take().expect("f must only be called once")()
        }
        unsafe { rust_none_try(__call::<F>, &mut f as *mut _ as *mut u8) }
    }
}

pub mod error {
    use string::String;
    use io;

    #[inline(always)]
    pub fn errno() -> i32 {
        0
    }

    #[inline(always)]
    pub fn error_string(code: i32) -> String {
        String::new()
    }

    #[inline(always)]
    pub fn decode_error_kind(code: i32) -> io::ErrorKind {
        io::ErrorKind::Other
    }
}

pub mod time {
    use time::Duration;
    use sys::time as sys;

    pub struct SteadyTime(usize);

    extern {
        fn rust_none_time_now() -> SteadyTime;
        fn rust_none_time_delta(lhs: &SteadyTime, rhs: &SteadyTime) -> Duration;
    }

    impl sys::SteadyTime for SteadyTime {
        fn now() -> Self {
            unsafe { rust_none_time_now() }
        }

        fn delta(&self, rhs: &SteadyTime) -> Duration {
            unsafe { rust_none_time_delta(self, rhs) }
        }
    }
}

pub mod sync {
    use time::Duration;
    use sys::sync as sys;

    pub struct Mutex;
    pub struct ReentrantMutex;
    pub struct Condvar;
    pub struct RwLock;

    impl sys::Lock for Mutex {
        unsafe fn lock(&self) { }
        unsafe fn unlock(&self) { }
        unsafe fn try_lock(&self) -> bool { true }
        unsafe fn destroy(&self) { }
    }

    impl sys::Mutex for Mutex { }

    impl sys::Lock for ReentrantMutex {
        unsafe fn lock(&self) { }
        unsafe fn unlock(&self) { }
        unsafe fn try_lock(&self) -> bool { true }
        unsafe fn destroy(&self) { }
    }

    impl sys::ReentrantMutex for ReentrantMutex {
        unsafe fn uninitialized() -> Self { ReentrantMutex }

        unsafe fn init(&mut self) { }
    }

    impl sys::RwLock for RwLock {
        unsafe fn read(&self) { }
        unsafe fn try_read(&self) -> bool { true }
        unsafe fn write(&self) { }
        unsafe fn try_write(&self) -> bool { true }
        unsafe fn read_unlock(&self) { }
        unsafe fn write_unlock(&self) { }
        unsafe fn destroy(&self) { }
    }

    impl sys::Condvar for Condvar {
        type Mutex = Mutex;

        unsafe fn notify_one(&self) { }
        unsafe fn notify_all(&self) { }
        unsafe fn wait(&self, _mutex: &Mutex) { }
        unsafe fn wait_timeout(&self, _mutex: &Mutex, _dur: Duration) -> bool { true }
        unsafe fn destroy(&self) { }
    }

    impl Condvar {
        pub const fn new() -> Self { Condvar }
    }

    impl Mutex {
        pub const fn new() -> Self { Mutex }
    }

    impl RwLock {
        pub const fn new() -> Self { RwLock }
    }
}

pub mod thread {
    use boxed::{Box, FnBox};
    use time::Duration;
    use io;
    use libc;
    use sys::thread as sys;

    extern {
        fn sleep(d: libc::c_uint) -> libc::c_uint;
        fn usleep(d: libc::useconds_t) -> libc::c_int;
    }

    pub struct Thread;

    impl sys::Thread for Thread {
        unsafe fn new<'a>(_stack: usize, _p: Box<FnBox() + 'a>) -> io::Result<Self> { Err(super::unimplemented()) }
        fn set_name(_name: &str) { }
        fn yield_now() { }
        fn sleep(dur: Duration) {
            unsafe { sleep(dur.as_secs() as libc::c_uint); }
            unsafe { usleep((dur.subsec_nanos() / 1000) as libc::useconds_t); }
        }

        fn join(self) { }

        unsafe fn guard_current() -> Option<usize> { None }
        unsafe fn guard_init() -> Option<usize> { None }
    }
}

pub mod backtrace {
    use io;

    pub fn write(_: &mut io::Write) -> io::Result<()> { Ok(()) }
}

pub mod env {
    use io;
    use fmt;
    use iter;
    use error;
    use marker;
    use convert;
    use ffi::{OsString, OsStr};
    use path::{PathBuf, Path};
    use sys::env as sys;
    pub struct Env;

    impl sys::Env for Env {
        type Args = iter::Empty<OsString>;
        type Vars = iter::Empty<(OsString, OsString)>;

        fn getcwd() -> io::Result<PathBuf> { Err(super::unimplemented()) }
        fn chdir(_p: &Path) -> io::Result<()> { Err(super::unimplemented()) }

        fn getenv(k: &OsStr) -> Option<OsString> { None }
        fn setenv(_k: &OsStr, _v: &OsStr) { }
        fn unsetenv(_k: &OsStr) { }

        fn home_dir() -> Option<PathBuf> { None }
        fn temp_dir() -> PathBuf { PathBuf::new() }
        fn current_exe() -> io::Result<PathBuf> { Err(super::unimplemented()) }

        fn env() -> Self::Vars { iter::empty() }
        fn args() -> Self::Args { iter::empty() }

        const FAMILY: &'static str = "none";
        const OS: &'static str = "none";
        const DLL_PREFIX: &'static str = "";
        const DLL_SUFFIX: &'static str = "";
        const DLL_EXTENSION: &'static str = "";
        const EXE_SUFFIX: &'static str = "";
        const EXE_EXTENSION: &'static str = "";
    }

    pub struct SplitPaths<'a>(marker::PhantomData<&'a [u8]>);

    pub fn split_paths(unparsed: &OsStr) -> SplitPaths {
        SplitPaths(marker::PhantomData)
    }

    impl<'a> iter::Iterator for SplitPaths<'a> {
        type Item = PathBuf;
        fn next(&mut self) -> Option<PathBuf> { None }
        fn size_hint(&self) -> (usize, Option<usize>) { (0, None) }
    }

    #[derive(Debug)]
    pub struct JoinPathsError;

    pub fn join_paths<I: iter::Iterator<Item=T>, T: convert::AsRef<OsStr>>(paths: I) -> Result<OsString, JoinPathsError> {
        Err(JoinPathsError)
    }

    impl fmt::Display for JoinPathsError {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            "unimplemented".fmt(f)
        }
    }

    impl error::Error for JoinPathsError {
        fn description(&self) -> &str { "unimplemented" }
    }
}

pub mod stdio {
    use io;
    use sys::stdio as sys;

    pub struct Stdin;
    pub struct Stdout;
    pub struct Stderr;

    impl sys::In for Stdin {
        fn new() -> io::Result<Self> { Ok(Stdin) }

        fn read(&self, data: &mut [u8]) -> io::Result<usize> { Err(super::unimplemented()) }
    }

    impl sys::Out for Stdout {
        fn new() -> io::Result<Self> { Ok(Stdout) }

        fn write(&self, data: &[u8]) -> io::Result<usize> { Err(super::unimplemented()) }
    }

    impl sys::Out for Stderr {
        fn new() -> io::Result<Self> { Ok(Stderr) }

        fn write(&self, data: &[u8]) -> io::Result<usize> { Err(super::unimplemented()) }
    }

    pub fn handle_ebadf<T>(r: io::Result<T>, default: T) -> io::Result<T> {
        r
    }
}

pub mod stack_overflow {
    use sys::stack_overflow as sys;

    pub struct Handler;

    impl sys::Handler for Handler {
        unsafe fn new() -> Self { Handler }
    }
}

pub mod rand {
    use io;
    use rand;
    use sys::rand as sys;

    pub struct Rng;

    impl sys::Rng for Rng {
        fn new() -> io::Result<Self> { Err(super::unimplemented()) }
    }

    impl rand::Rng for Rng {
        fn next_u32(&mut self) -> u32 { 0 }
        fn next_u64(&mut self) -> u64 { 0 }
        fn fill_bytes(&mut self, v: &mut [u8]) { }
    }
}

pub mod fs {
    use io;
    use ffi::{OsString, CStr};
    use path::{Path, PathBuf};
    use sys::fs as sys;

    pub struct Fs;
    pub struct ReadDir;
    #[derive(Clone)]
    pub struct OpenOptions;
    #[derive(Copy, Clone, PartialEq, Eq, Hash)]
    pub struct FileType;
    pub struct DirBuilder;
    pub struct DirEntry;
    #[derive(Debug)]
    pub struct File(());
    pub struct FileAttr;
    #[derive(Clone, PartialEq, Eq, Debug)]
    pub struct FilePermissions;
    impl sys::ReadDir<Fs> for ReadDir { }

    impl Iterator for ReadDir {
        type Item = io::Result<DirEntry>;
        fn next(&mut self) -> Option<io::Result<DirEntry>> { None }
    }

    impl sys::DirEntry for DirEntry {
        type Fs = Fs;

        fn path(&self) -> PathBuf { PathBuf::new() }
        fn file_name(&self) -> OsString { OsString::new() }
        fn metadata(&self) -> io::Result<FileAttr> { Err(super::unimplemented()) }
        fn file_type(&self) -> io::Result<FileType> { Err(super::unimplemented()) }
        fn ino(&self) -> <Self::Fs as sys::Fs>::INode { () }
    }

    impl sys::OpenOptions for OpenOptions {
        type Fs = Fs;

        fn new() -> Self where Self: Sized { OpenOptions }

        fn read(&mut self, read: bool) { }
        fn write(&mut self, write: bool) { }
        fn append(&mut self, append: bool) { }
        fn truncate(&mut self, truncate: bool) { }
        fn create(&mut self, create: bool) { }
        fn mode(&mut self, mode: <Self::Fs as sys::Fs>::Mode) { }
    }

    impl sys::FileType for FileType {
        type Fs = Fs;

        fn is_dir(&self) -> bool { false }
        fn is_file(&self) -> bool { false }
        fn is_symlink(&self) -> bool { false }

        fn is(&self, mode: <Self::Fs as sys::Fs>::Mode) -> bool { false }
    }

    impl sys::DirBuilder for DirBuilder {
        type Fs = Fs;

        fn new() -> Self { DirBuilder }
        fn mkdir(&self, p: &Path) -> io::Result<()> { Err(super::unimplemented()) }

        fn set_mode(&mut self, mode: <Self::Fs as sys::Fs>::Mode) { }
    }

    impl sys::File for File {
        type Fs = Fs;

        fn fsync(&self) -> io::Result<()> { Err(super::unimplemented()) }
        fn datasync(&self) -> io::Result<()> { Err(super::unimplemented()) }
        fn truncate(&self, sz: u64) -> io::Result<()> { Err(super::unimplemented()) }
        fn file_attr(&self) -> io::Result<FileAttr> { Err(super::unimplemented()) }

        fn open(path: &Path, opts: &OpenOptions) -> io::Result<Self> { Err(super::unimplemented()) }
        fn open_c(path: &CStr, opts: &OpenOptions) -> io::Result<Self> { Err(super::unimplemented()) }

        fn read(&self, buf: &mut [u8]) -> io::Result<usize> { Err(super::unimplemented()) }
        fn write(&self, buf: &[u8]) -> io::Result<usize> { Err(super::unimplemented()) }
        fn flush(&self) -> io::Result<()> { Err(super::unimplemented()) }
        fn seek(&self, pos: io::SeekFrom) -> io::Result<u64> { Err(super::unimplemented()) }

        fn fd(&self) -> &<Self::Fs as sys::Fs>::FileDesc { &self.0 }
        fn into_fd(self) -> <Self::Fs as sys::Fs>::FileDesc { () }
    }

    impl sys::FileAttr for FileAttr {
        type Fs = Fs;

        fn size(&self) -> u64 { 0 }
        fn perm(&self) -> FilePermissions { FilePermissions }
        fn file_type(&self) -> FileType { FileType }
    }

    impl sys::FilePermissions for FilePermissions {
        type Fs = Fs;

        fn readonly(&self) -> bool { false }
        fn set_readonly(&mut self, readonly: bool) { }
        fn mode(&self) -> <Self::Fs as sys::Fs>::Mode { () }
    }

    impl sys::Fs for Fs {
        type ReadDir = ReadDir;
        type File = File;
        type FileAttr = FileAttr;
        type DirEntry = DirEntry;
        type OpenOptions = OpenOptions;
        type FilePermissions = FilePermissions;
        type FileType = FileType;
        type DirBuilder = DirBuilder;
        type FileDesc = ();
        type Mode = ();
        type INode = ();

        fn unlink(p: &Path) -> io::Result<()> { Err(super::unimplemented()) }
        fn stat(p: &Path) -> io::Result<Self::FileAttr> { Err(super::unimplemented()) }
        fn lstat(p: &Path) -> io::Result<Self::FileAttr> { Err(super::unimplemented()) }
        fn rename(from: &Path, to: &Path) -> io::Result<()> { Err(super::unimplemented()) }
        fn copy(from: &Path, to: &Path) -> io::Result<u64> { Err(super::unimplemented()) }
        fn link(src: &Path, dst: &Path) -> io::Result<()> { Err(super::unimplemented()) }
        fn symlink(src: &Path, dst: &Path) -> io::Result<()> { Err(super::unimplemented()) }
        fn readlink(p: &Path) -> io::Result<PathBuf> { Err(super::unimplemented()) }
        fn canonicalize(p: &Path) -> io::Result<PathBuf> { Err(super::unimplemented()) }
        fn rmdir(p: &Path) -> io::Result<()> { Err(super::unimplemented()) }
        fn readdir(p: &Path) -> io::Result<ReadDir> { Err(super::unimplemented()) }
        fn set_perm(p: &Path, perm: FilePermissions) -> io::Result<()> { Err(super::unimplemented()) }
    }
}

pub mod dynamic_lib {
    use io;
    use ffi::OsStr;
    use sys::dynamic_lib as sys;

    pub struct Dl;
    impl sys::Dl for Dl {
        type Error = io::Error;

        fn open(filename: Option<&OsStr>) -> Result<Self, Self::Error> { Err(super::unimplemented()) }

        fn symbol(&self, _symbol: &str) -> Result<*mut u8, Self::Error> { Err(super::unimplemented()) }
        fn close(&self) -> Result<(), Self::Error> { Err(super::unimplemented()) }

        fn envvar() -> &'static str { "" }
        fn separator() -> &'static str { "" }
    }
}

pub mod process {
    use io;
    use fmt;
    use iter;
    use ffi::OsStr;
    use sys::process as sys;

    #[derive(Clone, Debug)]
    pub struct Command;
    #[derive(PartialEq, Eq, Clone, Copy, Debug)]
    pub struct ExitStatus;
    pub struct Pipe;
    pub struct Process(Option<Pipe>);

    impl sys::Command for Command {
        fn new(program: &OsStr) -> Self where Self: Sized { Command }

        fn arg(&mut self, arg: &OsStr) { }
        fn args<'a, I: iter::Iterator<Item = &'a OsStr>>(&mut self, args: I) { }
        fn env(&mut self, key: &OsStr, val: &OsStr) { }
        fn env_remove(&mut self, key: &OsStr) { }
        fn env_clear(&mut self) { }
        fn cwd(&mut self, dir: &OsStr) { }
    }

    impl fmt::Display for ExitStatus {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            <Self as fmt::Debug>::fmt(self, f)
        }
    }

    impl sys::ExitStatus for ExitStatus {
        fn success(&self) -> bool { false }
        fn code(&self) -> Option<i32> { None }
    }

    impl sys::PipeRead for Pipe {
        fn read(&self, buf: &mut [u8]) -> io::Result<usize> { Err(super::unimplemented()) }
    }

    impl sys::PipeWrite for Pipe {
        fn write(&self, buf: &[u8]) -> io::Result<usize> { Err(super::unimplemented()) }
    }

    impl sys::Process for Process {
        type RawFd = ();
        type Command = Command;
        type ExitStatus = ExitStatus;
        type PipeRead = Pipe;
        type PipeWrite = Pipe;

        fn pipe() -> io::Result<(Self::PipeRead, Self::PipeWrite)> { Err(super::unimplemented()) }
        fn spawn(cfg: &Self::Command, stdio: sys::Stdio<Self>, stdout: sys::Stdio<Self>, stderr: sys::Stdio<Self>) -> io::Result<Self> { Err(super::unimplemented()) }
        fn exit(code: i32) -> ! { unimplemented!() }

        unsafe fn kill(&self) -> io::Result<()> { Err(super::unimplemented()) }
        fn id(&self) -> u32 { 0 }
        fn wait(&self) -> io::Result<Self::ExitStatus> { Err(super::unimplemented()) }
        fn try_wait(&self) -> Option<Self::ExitStatus> { None }

        fn stdin(&mut self) -> &mut Option<Self::PipeWrite> { &mut self.0 }
        fn stdout(&mut self) -> &mut Option<Self::PipeRead> { &mut self.0 }
        fn stderr(&mut self) -> &mut Option<Self::PipeRead> { &mut self.0 }
    }
}

pub mod net {
    use io;
    use str;
    use iter;
    use borrow::Cow;
    use time::Duration;
    use rt::{FromInner, AsInner, IntoInner};
    use sys::net as sys;

    pub struct Net;
    pub struct LookupHost;
    pub struct LookupAddr;

    #[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
    pub struct AddrV4([u8; 4]);

    #[derive(Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
    pub struct AddrV6([u16; 8]);

    #[derive(Copy, Clone, PartialEq, Eq, Hash)]
    pub struct SocketAddrV4 {
        addr: AddrV4,
        port: u16,
    }

    #[derive(Copy, Clone, PartialEq, Eq, Hash)]
    pub struct SocketAddrV6 {
        addr: AddrV6,
        port: u16,
        flowinfo: u32,
        scope_id: u32,
    }

    #[derive(Debug)]
    pub struct Socket;

    impl iter::Iterator for LookupHost {
        type Item = io::Result<sys::SocketAddr<Net>>;

        fn next(&mut self) -> Option<Self::Item> { None }
    }

    impl sys::LookupHost<Net> for LookupHost { }

    impl sys::LookupAddr for LookupAddr {
        fn to_string_lossy(&self) -> Cow<str> { "".into() }
        fn to_str(&self) -> Result<&str, str::Utf8Error> { Ok("") }
    }

    impl sys::AddrV4 for AddrV4 {
        fn new(a: u8, b: u8, c: u8, d: u8) -> Self {
            AddrV4([a, b, c, d])
        }

        fn octets(&self) -> [u8; 4] { self.0 }
    }

    impl sys::AddrV6 for AddrV6 {
        fn new(a: u16, b: u16, c: u16, d: u16, e: u16, f: u16, g: u16, h: u16) -> Self {
            AddrV6([a, b, c, d, e, f, g, h])
        }

        fn segments(&self) -> [u16; 8] { self.0 }
    }

    impl sys::SocketAddrV4 for SocketAddrV4 {
        type Addr = AddrV4;

        fn new(ip: Self::Addr, port: u16) -> Self {
            SocketAddrV4 {
                addr: ip,
                port: port,
            }
        }

        fn addr(&self) -> &Self::Addr { &self.addr }
        fn port(&self) -> u16 { self.port }
    }

    impl sys::SocketAddrV6 for SocketAddrV6 {
        type Addr = AddrV6;

        fn new(ip: Self::Addr, port: u16, flowinfo: u32, scope_id: u32) -> Self {
            SocketAddrV6 {
                addr: ip,
                port: port,
                flowinfo: flowinfo,
                scope_id: scope_id,
            }
        }

        fn addr(&self) -> &Self::Addr { &self.addr }
        fn port(&self) -> u16 { self.port }
        fn flowinfo(&self) -> u32 { self.flowinfo }
        fn scope_id(&self) -> u32 { self.scope_id }
    }

    impl sys::Net for Net {
        type SocketAddrV4 = SocketAddrV4;
        type SocketAddrV6 = SocketAddrV6;

        type LookupHost = LookupHost;
        type LookupAddr = LookupAddr;

        type Socket = Socket;
        type TcpStream = Socket;
        type TcpListener = Socket;
        type UdpSocket = Socket;

        fn lookup_host(host: &str) -> io::Result<Self::LookupHost> { Err(super::unimplemented()) }
        fn lookup_addr(addr: &sys::IpAddr<Self>) -> io::Result<Self::LookupAddr> { Err(super::unimplemented()) }

        fn connect_tcp(addr: &sys::SocketAddr<Self>) -> io::Result<Self::TcpStream> { Err(super::unimplemented()) }
        fn bind_tcp(addr: &sys::SocketAddr<Self>) -> io::Result<Self::TcpListener> { Err(super::unimplemented()) }
        fn bind_udp(addr: &sys::SocketAddr<Self>) -> io::Result<Self::UdpSocket> { Err(super::unimplemented()) }
    }

    impl sys::TcpStream<Net> for Socket {
        fn read(&self, buf: &mut [u8]) -> io::Result<usize> { Err(super::unimplemented()) }
        fn write(&self, buf: &[u8]) -> io::Result<usize> { Err(super::unimplemented()) }
        fn peer_addr(&self) -> io::Result<sys::SocketAddr<Net>> { Err(super::unimplemented()) }
        fn shutdown(&self, how: sys::Shutdown) -> io::Result<()> { Err(super::unimplemented()) }
    }

    impl sys::TcpListener<Net> for Socket {
        fn accept(&self) -> io::Result<(Socket, sys::SocketAddr<Net>)> { Err(super::unimplemented()) }
    }

    impl sys::UdpSocket<Net> for Socket {
        fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, sys::SocketAddr<Net>)> { Err(super::unimplemented()) }
        fn send_to(&self, buf: &[u8], dst: &sys::SocketAddr<Net>) -> io::Result<usize> { Err(super::unimplemented()) }
    }

    impl sys::Socket<Net> for Socket {
        fn socket(&self) -> &Socket { &self }
        fn into_socket(self) -> Socket where Self: Sized { Socket }

        fn socket_addr(&self) -> io::Result<sys::SocketAddr<Net>> { Err(super::unimplemented()) }
        fn set_read_timeout(&self, dur: Option<Duration>) -> io::Result<()> { Err(super::unimplemented()) }
        fn set_write_timeout(&self, dur: Option<Duration>) -> io::Result<()> { Err(super::unimplemented()) }
        fn read_timeout(&self) -> io::Result<Option<Duration>> { Err(super::unimplemented()) }
        fn write_timeout(&self) -> io::Result<Option<Duration>> { Err(super::unimplemented()) }

        fn duplicate(&self) -> io::Result<Self> { Err(super::unimplemented()) }
    }

    impl FromInner<Socket> for Socket {
        fn from_inner(s: Socket) -> Self { s }
    }

    impl IntoInner<Socket> for Socket {
        fn into_inner(self) -> Socket { self }
    }

    impl AsInner<Socket> for Socket {
        fn as_inner(&self) -> &Socket { self }
    }
}
