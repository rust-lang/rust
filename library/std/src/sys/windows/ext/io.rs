//! Windows-specific extensions to general I/O primitives.

#![stable(feature = "rust1", since = "1.0.0")]

use crate::fs;
use crate::io;
use crate::net;
use crate::os::windows::raw;
use crate::sys;
use crate::sys::c;
use crate::sys_common::{self, AsInner, FromInner, IntoInner};

/// Raw HANDLEs.
#[stable(feature = "rust1", since = "1.0.0")]
pub type RawHandle = raw::HANDLE;

/// Raw SOCKETs.
#[stable(feature = "rust1", since = "1.0.0")]
pub type RawSocket = raw::SOCKET;

/// Extracts raw handles.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait AsRawHandle {
    /// Extracts the raw handle, without taking any ownership.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_raw_handle(&self) -> RawHandle;
}

/// Construct I/O objects from raw handles.
#[stable(feature = "from_raw_os", since = "1.1.0")]
pub trait FromRawHandle {
    /// Constructs a new I/O object from the specified raw handle.
    ///
    /// This function will **consume ownership** of the handle given,
    /// passing responsibility for closing the handle to the returned
    /// object.
    ///
    /// This function is also unsafe as the primitives currently returned
    /// have the contract that they are the sole owner of the file
    /// descriptor they are wrapping. Usage of this function could
    /// accidentally allow violating this contract which can cause memory
    /// unsafety in code that relies on it being true.
    #[stable(feature = "from_raw_os", since = "1.1.0")]
    unsafe fn from_raw_handle(handle: RawHandle) -> Self;
}

/// A trait to express the ability to consume an object and acquire ownership of
/// its raw `HANDLE`.
#[stable(feature = "into_raw_os", since = "1.4.0")]
pub trait IntoRawHandle {
    /// Consumes this object, returning the raw underlying handle.
    ///
    /// This function **transfers ownership** of the underlying handle to the
    /// caller. Callers are then the unique owners of the handle and must close
    /// it once it's no longer needed.
    #[stable(feature = "into_raw_os", since = "1.4.0")]
    fn into_raw_handle(self) -> RawHandle;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawHandle for fs::File {
    fn as_raw_handle(&self) -> RawHandle {
        self.as_inner().handle().raw() as RawHandle
    }
}

#[stable(feature = "asraw_stdio", since = "1.21.0")]
impl AsRawHandle for io::Stdin {
    fn as_raw_handle(&self) -> RawHandle {
        unsafe { c::GetStdHandle(c::STD_INPUT_HANDLE) as RawHandle }
    }
}

#[stable(feature = "asraw_stdio", since = "1.21.0")]
impl AsRawHandle for io::Stdout {
    fn as_raw_handle(&self) -> RawHandle {
        unsafe { c::GetStdHandle(c::STD_OUTPUT_HANDLE) as RawHandle }
    }
}

#[stable(feature = "asraw_stdio", since = "1.21.0")]
impl AsRawHandle for io::Stderr {
    fn as_raw_handle(&self) -> RawHandle {
        unsafe { c::GetStdHandle(c::STD_ERROR_HANDLE) as RawHandle }
    }
}

#[stable(feature = "asraw_stdio_locks", since = "1.35.0")]
impl<'a> AsRawHandle for io::StdinLock<'a> {
    fn as_raw_handle(&self) -> RawHandle {
        unsafe { c::GetStdHandle(c::STD_INPUT_HANDLE) as RawHandle }
    }
}

#[stable(feature = "asraw_stdio_locks", since = "1.35.0")]
impl<'a> AsRawHandle for io::StdoutLock<'a> {
    fn as_raw_handle(&self) -> RawHandle {
        unsafe { c::GetStdHandle(c::STD_OUTPUT_HANDLE) as RawHandle }
    }
}

#[stable(feature = "asraw_stdio_locks", since = "1.35.0")]
impl<'a> AsRawHandle for io::StderrLock<'a> {
    fn as_raw_handle(&self) -> RawHandle {
        unsafe { c::GetStdHandle(c::STD_ERROR_HANDLE) as RawHandle }
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawHandle for fs::File {
    unsafe fn from_raw_handle(handle: RawHandle) -> fs::File {
        let handle = handle as c::HANDLE;
        fs::File::from_inner(sys::fs::File::from_inner(handle))
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawHandle for fs::File {
    fn into_raw_handle(self) -> RawHandle {
        self.into_inner().into_handle().into_raw() as *mut _
    }
}

/// Extracts raw sockets.
#[stable(feature = "rust1", since = "1.0.0")]
pub trait AsRawSocket {
    /// Extracts the underlying raw socket from this object.
    #[stable(feature = "rust1", since = "1.0.0")]
    fn as_raw_socket(&self) -> RawSocket;
}

/// Creates I/O objects from raw sockets.
#[stable(feature = "from_raw_os", since = "1.1.0")]
pub trait FromRawSocket {
    /// Creates a new I/O object from the given raw socket.
    ///
    /// This function will **consume ownership** of the socket provided and
    /// it will be closed when the returned object goes out of scope.
    ///
    /// This function is also unsafe as the primitives currently returned
    /// have the contract that they are the sole owner of the file
    /// descriptor they are wrapping. Usage of this function could
    /// accidentally allow violating this contract which can cause memory
    /// unsafety in code that relies on it being true.
    #[stable(feature = "from_raw_os", since = "1.1.0")]
    unsafe fn from_raw_socket(sock: RawSocket) -> Self;
}

/// A trait to express the ability to consume an object and acquire ownership of
/// its raw `SOCKET`.
#[stable(feature = "into_raw_os", since = "1.4.0")]
pub trait IntoRawSocket {
    /// Consumes this object, returning the raw underlying socket.
    ///
    /// This function **transfers ownership** of the underlying socket to the
    /// caller. Callers are then the unique owners of the socket and must close
    /// it once it's no longer needed.
    #[stable(feature = "into_raw_os", since = "1.4.0")]
    fn into_raw_socket(self) -> RawSocket;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawSocket for net::TcpStream {
    fn as_raw_socket(&self) -> RawSocket {
        *self.as_inner().socket().as_inner()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawSocket for net::TcpListener {
    fn as_raw_socket(&self) -> RawSocket {
        *self.as_inner().socket().as_inner()
    }
}
#[stable(feature = "rust1", since = "1.0.0")]
impl AsRawSocket for net::UdpSocket {
    fn as_raw_socket(&self) -> RawSocket {
        *self.as_inner().socket().as_inner()
    }
}

#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawSocket for net::TcpStream {
    unsafe fn from_raw_socket(sock: RawSocket) -> net::TcpStream {
        let sock = sys::net::Socket::from_inner(sock);
        net::TcpStream::from_inner(sys_common::net::TcpStream::from_inner(sock))
    }
}
#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawSocket for net::TcpListener {
    unsafe fn from_raw_socket(sock: RawSocket) -> net::TcpListener {
        let sock = sys::net::Socket::from_inner(sock);
        net::TcpListener::from_inner(sys_common::net::TcpListener::from_inner(sock))
    }
}
#[stable(feature = "from_raw_os", since = "1.1.0")]
impl FromRawSocket for net::UdpSocket {
    unsafe fn from_raw_socket(sock: RawSocket) -> net::UdpSocket {
        let sock = sys::net::Socket::from_inner(sock);
        net::UdpSocket::from_inner(sys_common::net::UdpSocket::from_inner(sock))
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawSocket for net::TcpStream {
    fn into_raw_socket(self) -> RawSocket {
        self.into_inner().into_socket().into_inner()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawSocket for net::TcpListener {
    fn into_raw_socket(self) -> RawSocket {
        self.into_inner().into_socket().into_inner()
    }
}

#[stable(feature = "into_raw_os", since = "1.4.0")]
impl IntoRawSocket for net::UdpSocket {
    fn into_raw_socket(self) -> RawSocket {
        self.into_inner().into_socket().into_inner()
    }
}

#[unstable(feature = "is_atty", issue = "80937")]
impl io::IsAtty for sys::stdio::Stdin {
    fn is_atty() -> bool {
        use c::{
            STD_ERROR_HANDLE as STD_ERROR, STD_INPUT_HANDLE as STD_INPUT,
            STD_OUTPUT_HANDLE as STD_OUTPUT,
        };

        let fd = STD_INPUT;
        let others = [STD_ERROR, STD_OUTPUT];

        if unsafe { console_on_any(&[fd]) } {
            // False positives aren't possible. If we got a console then
            // we definitely have a tty on stdin.
            return true;
        }

        // At this point, we *could* have a false negative. We can determine that
        // this is true negative if we can detect the presence of a console on
        // any of the other streams. If another stream has a console, then we know
        // we're in a Windows console and can therefore trust the negative.
        if unsafe { console_on_any(&others) } {
            return false;
        }

        // Otherwise, we fall back to a very strange msys hack to see if we can
        // sneakily detect the presence of a tty.
        unsafe { msys_tty_on(fd) }
    }
}

#[unstable(feature = "is_atty", issue = "80937")]
impl io::IsAtty for sys::stdio::Stdout {
    fn is_atty() -> bool {
        use c::{
            STD_ERROR_HANDLE as STD_ERROR, STD_INPUT_HANDLE as STD_INPUT,
            STD_OUTPUT_HANDLE as STD_OUTPUT,
        };

        let fd = STD_OUTPUT;
        let others = [STD_INPUT, STD_ERROR];

        if unsafe { console_on_any(&[fd]) } {
            // False positives aren't possible. If we got a console then
            // we definitely have a tty on stdin.
            return true;
        }

        // At this point, we *could* have a false negative. We can determine that
        // this is true negative if we can detect the presence of a console on
        // any of the other streams. If another stream has a console, then we know
        // we're in a Windows console and can therefore trust the negative.
        if unsafe { console_on_any(&others) } {
            return false;
        }

        // Otherwise, we fall back to a very strange msys hack to see if we can
        // sneakily detect the presence of a tty.
        unsafe { msys_tty_on(fd) }
    }
}

#[unstable(feature = "is_atty", issue = "80937")]
impl io::IsAtty for sys::stdio::Stderr {
    fn is_atty() -> bool {
        use c::{
            STD_ERROR_HANDLE as STD_ERROR, STD_INPUT_HANDLE as STD_INPUT,
            STD_OUTPUT_HANDLE as STD_OUTPUT,
        };

        let fd = STD_ERROR;
        let others = [STD_INPUT, STD_OUTPUT];

        if unsafe { console_on_any(&[fd]) } {
            // False positives aren't possible. If we got a console then
            // we definitely have a tty on stdin.
            return true;
        }

        // At this point, we *could* have a false negative. We can determine that
        // this is true negative if we can detect the presence of a console on
        // any of the other streams. If another stream has a console, then we know
        // we're in a Windows console and can therefore trust the negative.
        if unsafe { console_on_any(&others) } {
            return false;
        }

        // Otherwise, we fall back to a very strange msys hack to see if we can
        // sneakily detect the presence of a tty.
        unsafe { msys_tty_on(fd) }
    }
}

#[unstable(feature = "is_atty", issue = "80937")]
unsafe fn console_on_any(fds: &[c::DWORD]) -> bool {
    use c::{GetConsoleMode, GetStdHandle};

    for &fd in fds {
        let mut out = 0;
        let handle = GetStdHandle(fd);
        if GetConsoleMode(handle, &mut out) != 0 {
            return true;
        }
    }
    false
}
#[unstable(feature = "is_atty", issue = "80937")]
unsafe fn msys_tty_on(fd: c::DWORD) -> bool {
    use std::{mem, slice};

    use c::{
        c_void, FileNameInfo, GetFileInformationByHandleEx, GetStdHandle, FILE_NAME_INFO, MAX_PATH,
    };

    let size = mem::size_of::<FILE_NAME_INFO>();
    let mut name_info_bytes = vec![0u8; size + MAX_PATH * mem::size_of::<WCHAR>()];
    let res = GetFileInformationByHandleEx(
        GetStdHandle(fd),
        FileNameInfo,
        &mut *name_info_bytes as *mut _ as *mut c_void,
        name_info_bytes.len() as u32,
    );
    if res == 0 {
        return false;
    }
    let name_info: &FILE_NAME_INFO = &*(name_info_bytes.as_ptr() as *const FILE_NAME_INFO);
    let s =
        slice::from_raw_parts(name_info.FileName.as_ptr(), name_info.FileNameLength as usize / 2);
    let name = String::from_utf16_lossy(s);
    // This checks whether 'pty' exists in the file name, which indicates that
    // a pseudo-terminal is attached. To mitigate against false positives
    // (e.g., an actual file name that contains 'pty'), we also require that
    // either the strings 'msys-' or 'cygwin-' are in the file name as well.)
    let is_msys = name.contains("msys-") || name.contains("cygwin-");
    let is_pty = name.contains("-pty");
    is_msys && is_pty
}
