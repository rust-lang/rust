#![allow(improper_ctypes)]

pub type Handle = usize;
pub type NZHandle = ::core::nonzero::NonZero<usize>;

pub mod thread_local {
    use thread_local as sys;
    use core::ptr::null_mut;
    use core::cell::UnsafeCell;
    use core::marker::Sync;

    pub struct StaticOsKey(UnsafeCell<*mut u8>, Option<unsafe extern fn(*mut u8)>);
    pub struct OsKey(UnsafeCell<*mut u8>, Option<unsafe extern fn(*mut u8)>);
    pub struct Key<T>(UnsafeCell<Option<T>>);
    unsafe impl<T> Sync for Key<T> { }
    unsafe impl<T> Send for Key<T> { }
    unsafe impl Sync for OsKey { }
    unsafe impl Send for OsKey { }
    unsafe impl Sync for StaticOsKey { }
    unsafe impl Send for StaticOsKey { }

    impl<T> sys::Key<T> for Key<T> {
        unsafe fn get(&'static self) -> Option<&'static UnsafeCell<Option<T>>> {
            Some(&self.0)
        }
    }

    impl<T> Key<T> {
        pub const fn new() -> Self { Key(UnsafeCell::new(None)) }
    }

    impl StaticOsKey {
        pub const fn new(dtor: Option<unsafe extern fn(*mut u8)>) -> Self { StaticOsKey(UnsafeCell::new(null_mut()), dtor) }
    }

    impl sys::StaticOsKey for StaticOsKey {
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
}

pub mod error {
    use error as sys;
    use c::prelude as c;
    use error::prelude::{Result, sys_Error, sys_ErrorString};
    use core::nonzero::NonZero;
    use core::fmt;

    pub type ErrorCode = NonZero<i32>;

    extern {
        fn rust_system_error_last() -> Option<ErrorCode>;
        fn rust_system_error_string(code: i32, buffer: &mut [u8]) -> Option<usize>;
    }

    const BUFSIZE: usize = 0x80;

    pub struct Error(ErrorCode);
    pub struct ErrorString([u8; BUFSIZE], usize);
    
    impl sys::Error for Error {
        type ErrorString = ErrorString;

        fn from_code(code: i32) -> Self {
            debug_assert!(code != 0);
            unsafe { Error(NonZero::new(code)) }
        }

        fn last_error() -> Option<Self> {
            let code = unsafe {
                rust_system_error_last()
            };

            code.map(Error)
        }

        fn code(&self) -> i32 { *self.0 }
        fn description(&self) -> ErrorString {
            let mut s = [0u8; BUFSIZE];
            let len = unsafe {
                rust_system_error_string(self.code(), &mut s)
            };
            ErrorString(s, len.unwrap_or(0))
        }
    }

    impl fmt::Debug for Error {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            fmt::Debug::fmt(&(self.code(), self.description().to_str().unwrap_or("unknown error")), f)
        }
    }

    impl fmt::Display for Error {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            fmt::Display::fmt(self.description().to_str().unwrap_or("unknown error"), f)
        }
    }

    impl sys::ErrorString for ErrorString {
        fn as_bytes(&self) -> &[u8] {
            &self.0[..self.1]
        }
    }

    impl From<fmt::Error> for Error {
        fn from(_: fmt::Error) -> Self {
            Error::from_code(c::EIO)
        }
    }
}

pub mod time {
    use time as sys;

    use error::prelude::*;
    use core::time;

    extern {
        fn rust_system_time_steady_now() -> Result<u64>;
    }

    const NANOS_PER_SEC: u64 = 1_000_000_000;

    pub struct SteadyTime(u64);

    impl sys::SteadyTime for SteadyTime {
        fn now() -> Result<Self> {
            unsafe {
                rust_system_time_steady_now().map(SteadyTime)
            }
        }

        fn delta(&self, rhs: &Self) -> time::Duration {
            let diff = self.0 - rhs.0;
            let secs = diff / NANOS_PER_SEC;
            let nanos = diff % NANOS_PER_SEC;
            time::Duration::new(secs, nanos as u32)
        }
    }
}

pub mod sync {
    use sync as sys;
    use core::time;
    use core::marker;
    use core::cell::UnsafeCell;
    use super::Handle;

    extern {
        fn rust_system_sync_mutex_lock(handle: *mut Handle);
        fn rust_system_sync_mutex_unlock(handle: *mut Handle);
        fn rust_system_sync_mutex_try_lock(handle: *mut Handle) -> bool;
        fn rust_system_sync_mutex_destroy(handle: *mut Handle);

        fn rust_system_sync_remutex_init(handle: *mut Handle);
        fn rust_system_sync_remutex_lock(handle: *mut Handle);
        fn rust_system_sync_remutex_unlock(handle: *mut Handle);
        fn rust_system_sync_remutex_try_lock(handle: *mut Handle) -> bool;
        fn rust_system_sync_remutex_destroy(handle: *mut Handle);

        fn rust_system_sync_rwlock_read(handle: *mut Handle);
        fn rust_system_sync_rwlock_try_read(handle: *mut Handle) -> bool;
        fn rust_system_sync_rwlock_write(handle: *mut Handle);
        fn rust_system_sync_rwlock_try_write(handle: *mut Handle) -> bool;
        fn rust_system_sync_rwlock_read_unlock(handle: *mut Handle);
        fn rust_system_sync_rwlock_write_unlock(handle: *mut Handle);
        fn rust_system_sync_rwlock_destroy(handle: *mut Handle);

        fn rust_system_sync_condvar_notify_one(handle: *mut Handle);
        fn rust_system_sync_condvar_notify_all(handle: *mut Handle);
        fn rust_system_sync_condvar_wait(handle: *mut Handle, mutex: *mut Handle);
        fn rust_system_sync_condvar_wait_timeout(handle: *mut Handle, mutex: *mut Handle, dur: time::Duration) -> bool;
        fn rust_system_sync_condvar_destroy(handle: *mut Handle);

        fn rust_system_sync_once(handle: *mut Handle, f: unsafe extern fn(usize), data: usize);
    }

    pub struct Sync(());
    pub struct Mutex(UnsafeCell<Handle>);
    pub struct ReentrantMutex(UnsafeCell<Handle>);
    pub struct RwLock(UnsafeCell<Handle>);
    pub struct Condvar(UnsafeCell<Handle>);
    pub struct Once(UnsafeCell<Handle>);

    impl sys::Sync for Sync {
        type Mutex = Mutex;
        type ReentrantMutex = ReentrantMutex;
        type RwLock = RwLock;
        type Condvar = Condvar;
        type Once = Once;
    }

    impl sys::Lock for ReentrantMutex {
        unsafe fn lock(&self) { rust_system_sync_remutex_lock(self.0.get()) }
        unsafe fn unlock(&self) { rust_system_sync_remutex_unlock(self.0.get()) }
        unsafe fn try_lock(&self) -> bool { rust_system_sync_remutex_try_lock(self.0.get()) }
        unsafe fn destroy(&self) { rust_system_sync_remutex_destroy(self.0.get()) }
    }

    impl sys::Lock for Mutex {
        unsafe fn lock(&self) { rust_system_sync_mutex_lock(self.0.get()) }
        unsafe fn unlock(&self) { rust_system_sync_mutex_unlock(self.0.get()) }
        unsafe fn try_lock(&self) -> bool { rust_system_sync_mutex_try_lock(self.0.get()) }
        unsafe fn destroy(&self) { rust_system_sync_mutex_destroy(self.0.get()) }
    }

    impl Mutex {
        pub const fn new() -> Mutex { Mutex(UnsafeCell::new(0)) }
    }

    impl sys::Mutex for Mutex { }

    impl ReentrantMutex {
        pub const fn uninitialized() -> ReentrantMutex { ReentrantMutex(UnsafeCell::new(0)) }
    }

    impl sys::ReentrantMutex for ReentrantMutex {
        unsafe fn init(&mut self) { rust_system_sync_remutex_init(self.0.get()) }
    }

    impl RwLock {
        pub const fn new() -> RwLock { RwLock(UnsafeCell::new(0)) }
    }

    impl sys::RwLock for RwLock {
        unsafe fn read(&self) { rust_system_sync_rwlock_read(self.0.get()) }
        unsafe fn try_read(&self) -> bool { rust_system_sync_rwlock_try_read(self.0.get()) }
        unsafe fn write(&self) { rust_system_sync_rwlock_write(self.0.get()) }
        unsafe fn try_write(&self) -> bool { rust_system_sync_rwlock_try_write(self.0.get()) }
        unsafe fn read_unlock(&self) { rust_system_sync_rwlock_read_unlock(self.0.get()) }
        unsafe fn write_unlock(&self) { rust_system_sync_rwlock_write_unlock(self.0.get()) }
        unsafe fn destroy(&self) { rust_system_sync_rwlock_destroy(self.0.get()) }
    }

    impl Condvar {
        pub const fn new() -> Condvar { Condvar(UnsafeCell::new(0)) }
    }

    impl sys::Condvar for Condvar {
        type Mutex = Mutex;

        unsafe fn notify_one(&self) { rust_system_sync_condvar_notify_one(self.0.get()) }
        unsafe fn notify_all(&self) { rust_system_sync_condvar_notify_all(self.0.get()) }
        unsafe fn wait(&self, mutex: &Mutex) { rust_system_sync_condvar_wait(self.0.get(), mutex.0.get()) }
        unsafe fn wait_timeout(&self, mutex: &Mutex, dur: time::Duration) -> bool { rust_system_sync_condvar_wait_timeout(self.0.get(), mutex.0.get(), dur) }
        unsafe fn destroy(&self) { rust_system_sync_condvar_destroy(self.0.get()) }
    }

    impl Once {
        pub const fn new() -> Once { Once(UnsafeCell::new(0)) }
    }

    impl sys::Once for Once {
        fn call_once<F: FnOnce()>(&'static self, f: F) {
            let mut f = Some(f);
            unsafe extern fn __call<F: FnOnce()>(f: usize) {
                let f = f as *mut Option<F>;
                (*f).take().expect("f must only be called once")()
            }
            unsafe { rust_system_sync_once(self.0.get(), __call::<F>, &mut f as *mut _ as usize) }
        }
    }

    unsafe impl marker::Sync for Mutex { }
    unsafe impl marker::Send for Mutex { }
    unsafe impl marker::Sync for ReentrantMutex { }
    unsafe impl marker::Send for ReentrantMutex { }
    unsafe impl marker::Sync for RwLock { }
    unsafe impl marker::Send for RwLock { }
    unsafe impl marker::Sync for Condvar { }
    unsafe impl marker::Send for Condvar { }
    unsafe impl marker::Sync for Once { }
    unsafe impl marker::Send for Once { }
}

pub mod stdio {
    use stdio as sys;
    use error::prelude::*;
    use io;

    extern {
        fn rust_system_stdio_stdin(buf: &mut [u8]) -> Result<usize>;
        fn rust_system_stdio_stdout(buf: &[u8]) -> Result<usize>;
        fn rust_system_stdio_stderr(buf: &[u8]) -> Result<usize>;
    }

    pub struct Stdio(());
    pub struct Stdin(());
    pub struct Stdout(());
    pub struct Stderr(());

    impl io::Read for Stdin {
        fn read(&self, buf: &mut [u8]) -> Result<usize> {
            unsafe { rust_system_stdio_stdin(buf) }
        }
    }

    impl io::Write for Stdout {
        fn write(&self, buf: &[u8]) -> Result<usize> {
            unsafe { rust_system_stdio_stdout(buf) }
        }
    }

    impl io::Write for Stderr {
        fn write(&self, buf: &[u8]) -> Result<usize> {
            unsafe { rust_system_stdio_stderr(buf) }
        }
    }

    impl sys::Stdio for Stdio {
        fn ebadf() -> i32 { 0 }

        type Stdin = Stdin;
        type Stdout = Stdout;
        type Stderr = Stderr;

        fn stdin() -> Result<Stdin> { Ok(Stdin(())) }
        fn stdout() -> Result<Stdout> { Ok(Stdout(())) }
        fn stderr() -> Result<Stderr> { Ok(Stderr(())) }
    }
}

pub mod stack_overflow {
    use stack_overflow as sys;
    use super::NZHandle as Handle;

    extern {
        fn rust_system_stack_overflow_handler() -> Option<Handle>;
        fn rust_system_stack_overflow_handler_drop(handle: Handle);
        fn rust_system_stack_overflow_report();
    }

    pub struct Handler(Option<Handle>);

    impl sys::Handler for Handler {
        unsafe fn new() -> Self {
            Handler(rust_system_stack_overflow_handler())
        }
    }

    impl Drop for Handler {
        fn drop(&mut self) {
            if let Some(ref handle) = self.0 {
                unsafe { rust_system_stack_overflow_handler_drop(*handle) }
            }
        }
    }

    pub unsafe fn report_overflow() {
        rust_system_stack_overflow_report();
    }
}

pub mod rand {
    use rand as sys;
    use core_rand as rand;
    use core::mem;
    use error::prelude::*;
    use super::NZHandle as Handle;

    extern {
        fn rust_system_rand_new() -> Result<Handle>;
        fn rust_system_rand_fill(handle: Handle, buf: &mut [u8]) -> Result<()>;
        fn rust_system_rand_drop(handle: Handle) -> Result<()>;
    }

    pub struct Rng(Handle);

    impl sys::Rng for Rng {
        fn new() -> Result<Self> { unsafe { rust_system_rand_new().map(Rng) } }
    }

    impl rand::Rng for Rng {
        fn next_u32(&mut self) -> u32 {
            let mut data = [0u8; 4];
            self.fill_bytes(&mut data);
            unsafe { mem::transmute(data) }
        }

        fn next_u64(&mut self) -> u64 {
            let mut data = [0u8; 8];
            self.fill_bytes(&mut data);
            unsafe { mem::transmute(data) }
        }

        fn fill_bytes(&mut self, v: &mut [u8]) {
            let res = unsafe { rust_system_rand_fill(self.0, v) };
            res.expect("Rng failed to produce bytes")
        }
    }

    impl Drop for Rng {
        fn drop(&mut self) {
            let _ = unsafe { rust_system_rand_drop(self.0) };
        }
    }
}

pub mod thread {
    use thread as sys;
    use error::prelude::*;
    use core::time::Duration;
    use super::NZHandle as Handle;

    extern {
        fn rust_system_thread_new(stack_size: usize, f: unsafe extern fn(usize) -> usize, data: usize) -> Result<Handle>;
        fn rust_system_thread_join(handle: Handle) -> Result<()>;
        fn rust_system_thread_drop(handle: Handle);
        fn rust_system_thread_set_name(name: &str) -> Result<()>;
        fn rust_system_thread_yield();
        fn rust_system_thread_sleep(dur: Duration) -> Result<()>;
    }

    pub struct Thread(Handle);

    impl sys::Thread for Thread {
        unsafe fn new(stack: usize, f: unsafe extern fn(usize) -> usize, data: usize) -> Result<Self> {
            rust_system_thread_new(stack, f, data).map(Thread)
        }

        fn join(self) -> Result<()> { unsafe { rust_system_thread_join(self.0) } }

        fn set_name(name: &str) -> Result<()> { unsafe { rust_system_thread_set_name(name) } }
        fn yield_() { unsafe { rust_system_thread_yield() } }
        fn sleep(dur: Duration) -> Result<()> { unsafe { rust_system_thread_sleep(dur) } }
    }

    impl Drop for Thread {
        fn drop(&mut self) {
            let _ = unsafe { rust_system_thread_drop(self.0) };
        }
    }
}

pub mod unwind {
    use unwind as sys;
    use core::any::Any;
    use alloc::boxed::Box;
    use core::fmt;

    extern {
        fn rust_system_unwind_begin_unwind_fmt(msg: fmt::Arguments, file_line: &(&'static str, u32)) -> !;
        fn rust_system_unwind_begin_unwind(msg: &Any, file_line: &(&'static str, u32)) -> !;
        fn rust_system_unwind_panic_inc() -> usize;
        fn rust_system_unwind_is_panicking() -> bool;
        fn rust_system_unwind_try(f: unsafe extern fn(usize), data: usize) -> Result<(), Box<Any + Send>>;
    }

    pub struct Unwind(());

    impl sys::Unwind for Unwind {
        fn begin_unwind_fmt(msg: fmt::Arguments, file_line: &(&'static str, u32)) -> ! {
            unsafe { rust_system_unwind_begin_unwind_fmt(msg, file_line) }
        }

        fn begin_unwind<M: Any + Send>(msg: M, file_line: &(&'static str, u32)) -> ! {
            unsafe { rust_system_unwind_begin_unwind(&msg, file_line) }
        }

        fn panic_inc() -> usize {
            unsafe { rust_system_unwind_panic_inc() }
        }

        fn is_panicking() -> bool {
            unsafe { rust_system_unwind_is_panicking() }
        }

        unsafe fn try<F: FnOnce()>(f: F) -> Result<(), Box<Any + Send>> {
            let mut f = Some(f);
            unsafe extern fn __call<F: FnOnce()>(f: usize) {
                let f = f as *mut Option<F>;
                (*f).take().expect("f must only be called once")()
            }
            rust_system_unwind_try(__call::<F>, &mut f as *mut _ as usize)
        }
    }
}

pub mod backtrace {
    use backtrace as sys;
    use error::prelude::*;
    use io;

    extern {
        fn rust_system_backtrace_write(w: &mut sys::BacktraceOutput) -> Result<()>;
        fn rust_system_backtrace_log_enabled() -> bool;
    }

    pub struct Backtrace(());

    impl Backtrace {
        pub const fn new() -> Self { Backtrace(()) }
    }

    impl sys::Backtrace for Backtrace {
        fn write<O: sys::BacktraceOutput>(&mut self, w: &mut O) -> Result<()> {
            unsafe { rust_system_backtrace_write(w) }
        }

        fn log_enabled() -> bool {
            unsafe { rust_system_backtrace_log_enabled() }
        }
    }
}

pub mod net {
    use net as sys;
    use error::prelude::*;
    use inner::prelude::*;
    use io;
    use core::iter;
    use core::mem;
    use core::time::Duration;
    use super::NZHandle as Handle;

    const BUFSIZE: usize = 0x400;

    pub struct Net(());
    pub struct LookupHost(());
    pub struct LookupAddr([u8; BUFSIZE], usize);

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
    pub struct Socket(Handle);

    impl iter::Iterator for LookupHost {
        type Item = Result<sys::SocketAddr<Net>>;

        fn next(&mut self) -> Option<Self::Item> { None }
    }

    impl sys::LookupHost<Net> for LookupHost { }

    impl sys::LookupAddr for LookupAddr {
        fn as_bytes(&self) -> &[u8] {
            &self.0[..self.1]
        }
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

    extern {
        //fn rust_system_net_lookup_host(host: &str) -> Result<LookupHost>;
        fn rust_system_net_lookup_addr(addr: &sys::IpAddr<Net>, buf: &mut [u8]) -> Result<usize>;

        fn rust_system_net_tcp_connect(addr: &sys::SocketAddr<Net>) -> Result<Handle>;
        fn rust_system_net_tcp_bind(addr: &sys::SocketAddr<Net>) -> Result<Handle>;
        fn rust_system_net_udp_bind(addr: &sys::SocketAddr<Net>) -> Result<Handle>;

        fn rust_system_net_socket_read(handle: Handle, buf: &mut [u8]) -> Result<usize>;
        fn rust_system_net_socket_write(handle: Handle, buf: &[u8]) -> Result<usize>;
        fn rust_system_net_socket_peer_addr(handle: Handle) -> Result<sys::SocketAddr<Net>>;
        fn rust_system_net_socket_shutdown(handle: Handle, how: sys::Shutdown) -> Result<()>;

        fn rust_system_net_socket_accept(handle: Handle) -> Result<(Handle, sys::SocketAddr<Net>)>;
        fn rust_system_net_socket_recv_from(handle: Handle, buf: &mut [u8]) -> Result<(usize, sys::SocketAddr<Net>)>;
        fn rust_system_net_socket_send_to(handle: Handle, buf: &[u8], dst: &sys::SocketAddr<Net>) -> Result<usize>;

        fn rust_system_net_socket_local_addr(handle: Handle) -> Result<sys::SocketAddr<Net>>;
        fn rust_system_net_socket_set_read_timeout(handle: Handle, dur: Option<Duration>) -> Result<()>;
        fn rust_system_net_socket_set_write_timeout(handle: Handle, dur: Option<Duration>) -> Result<()>;
        fn rust_system_net_socket_get_read_timeout(handle: Handle) -> Result<Option<Duration>>;
        fn rust_system_net_socket_get_write_timeout(handle: Handle) -> Result<Option<Duration>>;
        fn rust_system_net_socket_duplicate(handle: Handle) -> Result<Handle>;
        fn rust_system_net_socket_drop(handle: Handle) -> Result<()>;
    }

    impl sys::Net for Net {
        type SocketAddrV4 = SocketAddrV4;
        type SocketAddrV6 = SocketAddrV6;

        type LookupHost = LookupHost;
        type LookupAddr = LookupAddr;

        type Socket = Handle;
        type TcpStream = Socket;
        type TcpListener = Socket;
        type UdpSocket = Socket;

        fn lookup_host(host: &str) -> Result<Self::LookupHost> { panic!() }

        fn lookup_addr(addr: &sys::IpAddr<Self>) -> Result<Self::LookupAddr> {
            let mut s = [0u8; BUFSIZE];
            let len = unsafe {
                rust_system_net_lookup_addr(addr, &mut s)
            };
            len.map(|len| LookupAddr(s, len))
        }

        fn connect_tcp(addr: &sys::SocketAddr<Self>) -> Result<Self::TcpStream> {
            unsafe { rust_system_net_tcp_connect(addr).map(Socket) }
        }

        fn bind_tcp(addr: &sys::SocketAddr<Self>) -> Result<Self::TcpListener> {
            unsafe { rust_system_net_tcp_bind(addr).map(Socket) }
        }

        fn bind_udp(addr: &sys::SocketAddr<Self>) -> Result<Self::UdpSocket> {
            unsafe { rust_system_net_udp_bind(addr).map(Socket) }
        }
    }

    impl io::Read for Socket {
        fn read(&self, buf: &mut [u8]) -> Result<usize> {
            unsafe { rust_system_net_socket_read(self.0, buf) }
        }
    }

    impl io::Write for Socket {
        fn write(&self, buf: &[u8]) -> Result<usize> {
            unsafe { rust_system_net_socket_write(self.0, buf) }
        }
    }

    impl sys::TcpStream<Net> for Socket {
        fn peer_addr(&self) -> Result<sys::SocketAddr<Net>> {
            unsafe { rust_system_net_socket_peer_addr(self.0) }
        }

        fn shutdown(&self, how: sys::Shutdown) -> Result<()> {
            unsafe { rust_system_net_socket_shutdown(self.0, how) }
        }
    }

    impl sys::TcpListener<Net> for Socket {
        fn accept(&self) -> Result<(Socket, sys::SocketAddr<Net>)> {
            unsafe { rust_system_net_socket_accept(self.0).map(|(s, a)| (Socket(s), a)) }
        }
    }

    impl sys::UdpSocket<Net> for Socket {
        fn recv_from(&self, buf: &mut [u8]) -> Result<(usize, sys::SocketAddr<Net>)> {
            unsafe { rust_system_net_socket_recv_from(self.0, buf) }
        }
        fn send_to(&self, buf: &[u8], dst: &sys::SocketAddr<Net>) -> Result<usize> {
            unsafe { rust_system_net_socket_send_to(self.0, buf, dst) }
        }
    }

    impl sys::Socket<Net> for Socket {
        fn socket(&self) -> &Handle { &self.0 }
        fn into_socket(self) -> Handle {
            let h = self.0;
            mem::forget(self);
            h
        }

        fn socket_addr(&self) -> Result<sys::SocketAddr<Net>> {
            unsafe { rust_system_net_socket_local_addr(self.0) }
        }

        fn set_read_timeout(&self, dur: Option<Duration>) -> Result<()> {
            unsafe { rust_system_net_socket_set_read_timeout(self.0, dur) }
        }

        fn set_write_timeout(&self, dur: Option<Duration>) -> Result<()> {
            unsafe { rust_system_net_socket_set_write_timeout(self.0, dur) }
        }

        fn read_timeout(&self) -> Result<Option<Duration>> {
            unsafe { rust_system_net_socket_get_read_timeout(self.0) }
        }

        fn write_timeout(&self) -> Result<Option<Duration>> {
            unsafe { rust_system_net_socket_get_write_timeout(self.0) }
        }

        fn duplicate(&self) -> Result<Self> {
            unsafe { rust_system_net_socket_duplicate(self.0).map(Socket) }
        }
    }

    impl Drop for Socket {
        fn drop(&mut self) {
            let _ = unsafe { rust_system_net_socket_drop(self.0) };
        }
    }

    impl FromInner<Handle> for Socket {
        fn from_inner(s: Handle) -> Self { Socket(s) }
    }

    impl IntoInner<Handle> for Socket {
        fn into_inner(self) -> Handle { sys::Socket::into_socket(self) }
    }

    impl AsInner<Handle> for Socket {
        fn as_inner(&self) -> &Handle { &self.0 }
    }
}

pub mod dynamic_lib {
    use dynamic_lib as sys;
    use os_str::prelude::*;
    use error::prelude::*;
    use super::NZHandle as Handle;

    extern {
        fn rust_system_dynamic_lib_open(filename: Option<&OsStr>) -> Result<Handle>;
        fn rust_system_dynamic_lib_symbol(handle: Handle, name: &str) -> Result<*mut u8>;
        fn rust_system_dynamic_lib_close(handle: Handle) -> Result<()>;
        fn rust_system_dynamic_lib_envvar() -> &'static str;
        fn rust_system_dynamic_lib_separator() -> &'static str;
    }

    pub struct DynamicLibrary(Handle);

    impl sys::DynamicLibrary for DynamicLibrary {
        type Error = Error;

        fn open(filename: Option<&OsStr>) -> Result<Self> {
            unsafe { rust_system_dynamic_lib_open(filename).map(DynamicLibrary) }
        }

        fn symbol(&self, symbol: &str) -> Result<*mut u8> {
            unsafe { rust_system_dynamic_lib_symbol(self.0, symbol) }
        }
        fn close(&self) -> Result<()> {
            unsafe { rust_system_dynamic_lib_close(self.0) }
        }

        fn envvar() -> &'static str {
            unsafe { rust_system_dynamic_lib_envvar() }
        }

        fn separator() -> &'static str {
            unsafe { rust_system_dynamic_lib_separator() }
        }
    }
}

pub mod env {
    use env as sys;
    use os_str::prelude::*;
    use error::prelude::*;
    use core::result;
    use core::iter;

    extern {
        fn rust_system_env_getcwd() -> Result<OsString>;
        fn rust_system_env_chdir(p: &OsStr) -> Result<()>;
        fn rust_system_env_getenv(k: &OsStr) -> Result<Option<OsString>>;
        fn rust_system_env_setenv(k: &OsStr, v: &OsStr) -> Result<()>;
        fn rust_system_env_unsetenv(k: &OsStr) -> Result<()>;
        fn rust_system_env_home_dir() -> Result<OsString>;
        fn rust_system_env_temp_dir() -> Result<OsString>;
        fn rust_system_env_current_exe() -> Result<OsString>;
    }

    pub struct Env(());
    pub type SplitPaths<'a> = iter::Empty<&'a OsStr>;

    impl sys::Env for Env {
        type Args = iter::Empty<OsString>;
        type Vars = iter::Empty<(OsString, OsString)>;

        fn getcwd() -> Result<OsString> {
            unsafe { rust_system_env_getcwd() }
        }

        fn chdir(p: &OsStr) -> Result<()> {
            unsafe { rust_system_env_chdir(p) }
        }

        fn getenv(k: &OsStr) -> Result<Option<OsString>> {
            unsafe { rust_system_env_getenv(k) }
        }

        fn setenv(k: &OsStr, v: &OsStr) -> Result<()> {
            unsafe { rust_system_env_setenv(k, v) }
        }

        fn unsetenv(k: &OsStr) -> Result<()> {
            unsafe { rust_system_env_unsetenv(k) }
        }

        fn home_dir() -> Result<OsString> {
            unsafe { rust_system_env_home_dir() }
        }

        fn temp_dir() -> Result<OsString> {
            unsafe { rust_system_env_temp_dir() }
        }

        fn current_exe() -> Result<OsString> {
            unsafe { rust_system_env_current_exe() }
        }

        fn env() -> Result<Self::Vars> { panic!() }
        fn args() -> Result<Self::Args> { panic!() }

        fn join_paths<'a, I: Iterator<Item=T>, T: AsRef<OsStr>>(paths: I) -> result::Result<OsString, sys::JoinPathsError<Self>> { panic!() }
        fn join_paths_error() -> &'static str { "" }

        const ARCH: &'static str = sys::ARCH;
        const FAMILY: &'static str = "";
        const OS: &'static str = "";
        const DLL_PREFIX: &'static str = "";
        const DLL_SUFFIX: &'static str = "";
        const DLL_EXTENSION: &'static str = "";
        const EXE_SUFFIX: &'static str = "";
        const EXE_EXTENSION: &'static str = "";
    }

    pub fn split_paths(unparsed: &OsStr) -> SplitPaths { panic!() }
}

pub mod process {
    use process as sys;
    use io;
    use error::prelude::*;
    use core::iter;
    use core::fmt;
    use os_str::prelude::*;
    use super::NZHandle as Handle;

    extern {
        fn rust_system_process_spawn(cmd: &Command, stdin: sys::Stdio<Process>, stdout: sys::Stdio<Process>, stderr: sys::Stdio<Process>) -> Result<(Handle, Option<AnonPipe>, Option<AnonPipe>, Option<AnonPipe>)>;
        fn rust_system_process_exit(code: i32) -> !;
        fn rust_system_process_kill(handle: Handle) -> Result<()>;
        fn rust_system_process_id(handle: Handle) -> Result<u32>;
        fn rust_system_process_wait(handle: Handle) -> Result<ExitStatus>;
        fn rust_system_process_try_wait(handle: Handle) -> Option<ExitStatus>;
        fn rust_system_process_pipe_read(handle: Handle, buf: &mut [u8]) -> Result<usize>;
        fn rust_system_process_pipe_write(handle: Handle, buf: &[u8]) -> Result<usize>;
        fn rust_system_process_exit_status_display(status: &ExitStatus, f: &mut fmt::Formatter) -> fmt::Result;
    }

    pub struct Process {
        handle: Handle,
        stdin: Option<AnonPipe>,
        stdout: Option<AnonPipe>,
        stderr: Option<AnonPipe>,
    }
    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    pub struct ExitStatus(Option<i32>, Option<Handle>);
    pub struct AnonPipe(Handle);
    #[derive(Clone, Debug)]
    pub struct Command(());

    impl io::Read for AnonPipe {
        fn read(&self, buf: &mut [u8]) -> Result<usize> {
            unsafe { rust_system_process_pipe_read(self.0, buf) }
        }
    }

    impl io::Write for AnonPipe {
        fn write(&self, buf: &[u8]) -> Result<usize> {
            unsafe { rust_system_process_pipe_write(self.0, buf) }
        }
    }

    impl sys::Command for Command {
        fn new(program: &OsStr) -> Result<Self> { panic!() }

        fn arg(&mut self, arg: &OsStr) { panic!() }
        fn args<'a, I: iter::Iterator<Item = &'a OsStr>>(&mut self, args: I) { panic!() }
        fn env(&mut self, key: &OsStr, val: &OsStr) { panic!() }
        fn env_remove(&mut self, key: &OsStr) { panic!() }
        fn env_clear(&mut self) { panic!() }
        fn cwd(&mut self, dir: &OsStr) { panic!() }
    }

    impl sys::ExitStatus for ExitStatus {
        fn success(&self) -> bool { self.0 == Some(0) }
        fn code(&self) -> Option<i32> { self.0 }
    }

    impl fmt::Display for ExitStatus {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            unsafe {
                rust_system_process_exit_status_display(self, f)
            }
        }
    }

    impl sys::Process for Process {
        type RawFd = Handle;
        type Command = Command;
        type ExitStatus = ExitStatus;
        type PipeRead = AnonPipe;
        type PipeWrite = AnonPipe;

        fn spawn(cfg: &Self::Command, stdin: sys::Stdio<Self>, stdout: sys::Stdio<Self>, stderr: sys::Stdio<Self>) -> Result<Self> {
            unsafe { rust_system_process_spawn(cfg, stdin, stdout, stderr)
                .map(|(h, sin, sout, serr)| Process {
                    handle: h,
                    stdin: sin,
                    stdout: sout,
                    stderr: serr,
                })
            }
        }

        fn exit(code: i32) -> ! {
            unsafe { rust_system_process_exit(code) }
        }

        unsafe fn kill(&self) -> Result<()> {
            rust_system_process_kill(self.handle)
        }

        fn id(&self) -> Result<u32> {
            unsafe { rust_system_process_id(self.handle) }
        }

        fn wait(&self) -> Result<Self::ExitStatus> {
            unsafe { rust_system_process_wait(self.handle) }
        }

        fn try_wait(&self) -> Option<Self::ExitStatus> {
            unsafe { rust_system_process_try_wait(self.handle) }
        }

        fn stdin(&mut self) -> &mut Option<Self::PipeWrite> { &mut self.stdin }
        fn stdout(&mut self) -> &mut Option<Self::PipeRead> { &mut self.stdout }
        fn stderr(&mut self) -> &mut Option<Self::PipeRead> { &mut self.stderr }
    }
}

pub mod fs {
    use fs as sys;
    use inner::prelude::*;
    use error::prelude::*;
    use c_str::CStr;
    use io;
    use core::mem;
    use os_str::imp::{OsStr, OsString};
    use super::NZHandle as Handle;

    extern {
        fn rust_system_fs_unlink(p: &OsStr) -> Result<()>;
        fn rust_system_fs_stat(p: &OsStr) -> Result<FileAttr>;
        fn rust_system_fs_lstat(p: &OsStr) -> Result<FileAttr>;
        fn rust_system_fs_rename(from: &OsStr, to: &OsStr) -> Result<()>;
        fn rust_system_fs_copy(from: &OsStr, to: &OsStr) -> Result<u64>;
        fn rust_system_fs_link(src: &OsStr, dst: &OsStr) -> Result<()>;
        fn rust_system_fs_symlink(src: &OsStr, dst: &OsStr) -> Result<()>;
        fn rust_system_fs_readlink(p: &OsStr) -> Result<OsString>;
        fn rust_system_fs_canonicalize(p: &OsStr) -> Result<OsString>;
        fn rust_system_fs_rmdir(p: &OsStr) -> Result<()>;
        fn rust_system_fs_set_perm(p: &OsStr, perm: FilePermissions) -> Result<()>;

        fn rust_system_fs_file_open(p: &OsStr, opt: &OpenOptions) -> Result<Handle>;
        fn rust_system_fs_file_read(handle: Handle, buf: &mut [u8]) -> Result<usize>;
        fn rust_system_fs_file_write(handle: Handle, buf: &[u8]) -> Result<usize>;
        fn rust_system_fs_file_flush(handle: Handle) -> Result<()>;
        fn rust_system_fs_file_seek(handle: Handle, pos: io::SeekFrom) -> Result<u64>;
        fn rust_system_fs_file_fsync(handle: Handle) -> Result<()>;
        fn rust_system_fs_file_datasync(handle: Handle) -> Result<()>;
        fn rust_system_fs_file_truncate(handle: Handle, sz: u64) -> Result<()>;
        fn rust_system_fs_file_stat(handle: Handle) -> Result<FileAttr>;
        fn rust_system_fs_file_close(handle: Handle) -> Result<()>;
    }

    pub struct Fs(());
    pub struct ReadDir;
    #[derive(Clone)]
    pub struct OpenOptions(OpenFlags, Mode);
    #[derive(Copy, Clone, PartialEq, Eq, Hash)]
    pub struct FileType;
    pub struct DirBuilder;
    pub struct DirEntry;
    #[derive(Debug)]
    pub struct File(Handle);
    pub struct FileAttr;
    #[derive(Clone, PartialEq, Eq, Debug)]
    pub struct FilePermissions(u32);
    pub type Mode = usize;
    pub type INode = usize;

    bitflags! {
        flags OpenFlags: u8 {
            const OPEN_FLAG_READ        = 0b00000001,
            const OPEN_FLAG_WRITE       = 0b00000010,
            const OPEN_FLAG_APPEND      = 0b00000100,
            const OPEN_FLAG_TRUNCATE    = 0b00001000,
            const OPEN_FLAG_CREATE      = 0b00010000,
        }
    }

    impl Iterator for ReadDir {
        type Item = Result<DirEntry>;
        fn next(&mut self) -> Option<Result<DirEntry>> { panic!() }
    }

    impl sys::ReadDir<Fs> for ReadDir { }

    impl sys::DirEntry<Fs> for DirEntry {
        fn file_name(&self) -> &OsStr { panic!() }
        fn root(&self) -> &OsStr { panic!() }
        fn metadata(&self) -> Result<FileAttr> { panic!() }
        fn file_type(&self) -> Result<FileType> { panic!() }
        fn ino(&self) -> INode { panic!() }
    }

    impl sys::OpenOptions<Fs> for OpenOptions {
        fn new() -> Self { OpenOptions(OpenFlags { bits: 0 }, 0) }

        fn read(&mut self, read: bool) { self.0 = self.0 | OpenFlags::OPEN_FLAG_READ }
        fn write(&mut self, write: bool) { self.0 = self.0 | OpenFlags::OPEN_FLAG_WRITE }
        fn append(&mut self, append: bool) { self.0 = self.0 | OpenFlags::OPEN_FLAG_APPEND }
        fn truncate(&mut self, truncate: bool) { self.0 = self.0 | OpenFlags::OPEN_FLAG_TRUNCATE }
        fn create(&mut self, create: bool) { self.0 = self.0 | OpenFlags::OPEN_FLAG_CREATE }
        fn mode(&mut self, mode: Mode) { self.1 = mode }
    }

    impl sys::FileType<Fs> for FileType {
        fn is_dir(&self) -> bool { panic!() }
        fn is_file(&self) -> bool { panic!() }
        fn is_symlink(&self) -> bool { panic!() }

        fn is(&self, mode: Mode) -> bool { panic!() }
    }

    impl sys::DirBuilder<Fs> for DirBuilder {
        fn new() -> Self { panic!() }
        fn mkdir(&self, p: &OsStr) -> Result<()> { panic!() }

        fn set_mode(&mut self, mode: Mode) { panic!() }
    }

    impl sys::File<Fs> for File {
        fn open(path: &OsStr, opts: &OpenOptions) -> Result<Self> {
            unsafe { rust_system_fs_file_open(path, opts).map(File) }
        }

        fn open_c(path: &CStr, opts: &OpenOptions) -> Result<Self> {
            unsafe { rust_system_fs_file_open(OsStr::from_bytes(path.to_bytes()), opts).map(File) }
        }

        fn fsync(&self) -> Result<()> {
            unsafe { rust_system_fs_file_fsync(self.0) }
        }

        fn datasync(&self) -> Result<()> {
            unsafe { rust_system_fs_file_datasync(self.0) }
        }

        fn truncate(&self, sz: u64) -> Result<()> {
            unsafe { rust_system_fs_file_truncate(self.0, sz) }
        }

        fn file_attr(&self) -> Result<FileAttr> {
            unsafe { rust_system_fs_file_stat(self.0) }
        }
    }

    impl Drop for File {
        fn drop(&mut self) {
            let _ = unsafe { rust_system_fs_file_close(self.0) };
        }
    }

    impl io::Read for File {
        fn read(&self, buf: &mut [u8]) -> Result<usize> {
            unsafe { rust_system_fs_file_read(self.0, buf) }
        }
    }

    impl io::Write for File {
        fn write(&self, buf: &[u8]) -> Result<usize> {
            unsafe { rust_system_fs_file_write(self.0, buf) }
        }

        fn flush(&self) -> Result<()> {
            unsafe { rust_system_fs_file_flush(self.0) }
        }
    }

    impl io::Seek for File {
        fn seek(&self, pos: io::SeekFrom) -> Result<u64> {
            unsafe { rust_system_fs_file_seek(self.0, pos) }
        }
    }

    impl AsInner<Handle> for File {
        fn as_inner(&self) -> &Handle { &self.0 }
    }

    impl IntoInner<Handle> for File {
        fn into_inner(self) -> Handle {
            let h = self.0;
            mem::forget(self);
            h
        }
    }

    impl sys::FileAttr<Fs> for FileAttr {
        fn size(&self) -> u64 { panic!() }
        fn perm(&self) -> FilePermissions { panic!() }
        fn file_type(&self) -> FileType { panic!() }
    }

    impl sys::FilePermissions<Fs> for FilePermissions {
        fn readonly(&self) -> bool { panic!() }
        fn set_readonly(&mut self, readonly: bool) { panic!() }
        fn mode(&self) -> Mode { panic!() }
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
        type FileHandle = Handle;
        type Mode = Mode;
        type INode = INode;

        fn unlink(p: &OsStr) -> Result<()> {
            unsafe { rust_system_fs_unlink(p) }
        }

        fn stat(p: &OsStr) -> Result<Self::FileAttr> {
            unsafe { rust_system_fs_stat(p) }
        }

        fn lstat(p: &OsStr) -> Result<Self::FileAttr> {
            unsafe { rust_system_fs_lstat(p) }
        }

        fn rename(from: &OsStr, to: &OsStr) -> Result<()> {
            unsafe { rust_system_fs_rename(from, to) }
        }

        const COPY_IMP: bool = true;
        fn copy(from: &OsStr, to: &OsStr) -> Result<u64> {
            unsafe { rust_system_fs_copy(from, to) }
        }

        fn link(src: &OsStr, dst: &OsStr) -> Result<()> {
            unsafe { rust_system_fs_link(src, dst) }
        }

        fn symlink(src: &OsStr, dst: &OsStr) -> Result<()> {
            unsafe { rust_system_fs_symlink(src, dst) }
        }

        fn readlink(p: &OsStr) -> Result<OsString> {
            unsafe { rust_system_fs_readlink(p) }
        }

        fn canonicalize(p: &OsStr) -> Result<OsString> {
            unsafe { rust_system_fs_canonicalize(p) }
        }

        fn rmdir(p: &OsStr) -> Result<()> {
            unsafe { rust_system_fs_rmdir(p) }
        }

        fn set_perm(p: &OsStr, perm: FilePermissions) -> Result<()> {
            unsafe { rust_system_fs_set_perm(p, perm) }
        }

        fn readdir(p: &OsStr) -> Result<ReadDir> { panic!() }
    }
}

pub mod path {
    use os_str::prelude::*;
    use path as sys;

    pub struct PathInfo(());

    impl sys::PathInfo for PathInfo {
        #[inline]
        fn is_sep_byte(b: u8) -> bool {
            b == b'/'
        }

        #[inline]
        fn is_verbatim_sep(b: u8) -> bool {
            b == b'/'
        }

        const PREFIX_IMP: bool = false;

        fn parse_prefix(s: &OsStr) -> Option<sys::Prefix> {
            None
        }

        const MAIN_SEP_STR: &'static str = "/";
        const MAIN_SEP: char = '/';
    }
}

pub mod rt {
    use rt as sys;
    use core::any::Any;
    use core::fmt;

    pub struct Runtime(());

    impl sys::Runtime for Runtime {
        unsafe fn run_main<R, F: FnOnce() -> R>(f: F, argc: isize, argv: *const *const u8) -> R { panic!() }
        unsafe fn run_thread<R, F: FnOnce() -> R>(f: F) -> R { panic!() }
        unsafe fn thread_cleanup() { panic!() }
        unsafe fn cleanup() { panic!() }

        fn on_panic(msg: &(Any + Send), file: &'static str, line: u32) { panic!() }
        fn min_stack() -> usize { panic!() }
        fn abort(args: fmt::Arguments) -> ! { panic!() }
    }
}

pub mod c {
    extern {
        pub fn rust_system_rt_strlen(s: *const c_char) -> usize;
    }

    pub const EINVAL: i32 = 22;
    pub const EIO: i32 = 5;

    pub type c_char = i8;
    pub type c_int = i32;
    pub type c_float = f32;
    pub type c_double = f64;
    pub use self::rust_system_rt_strlen as strlen;
}
