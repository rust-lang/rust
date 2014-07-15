// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Named pipes implementation for windows
//!
//! If are unfortunate enough to be reading this code, I would like to first
//! apologize. This was my first encounter with windows named pipes, and it
//! didn't exactly turn out very cleanly. If you, too, are new to named pipes,
//! read on as I'll try to explain some fun things that I ran into.
//!
//! # Unix pipes vs Named pipes
//!
//! As with everything else, named pipes on windows are pretty different from
//! unix pipes on unix. On unix, you use one "server pipe" to accept new client
//! pipes. So long as this server pipe is active, new children pipes can
//! connect. On windows, you instead have a number of "server pipes", and each
//! of these server pipes can throughout their lifetime be attached to a client
//! or not. Once attached to a client, a server pipe may then disconnect at a
//! later date.
//!
//! # Accepting clients
//!
//! As with most other I/O interfaces, our Listener/Acceptor/Stream interfaces
//! are built around the unix flavors. This means that we have one "server
//! pipe" to which many clients can connect. In order to make this compatible
//! with the windows model, each connected client consumes ownership of a server
//! pipe, and then a new server pipe is created for the next client.
//!
//! Note that the server pipes attached to clients are never given back to the
//! listener for recycling. This could possibly be implemented with a channel so
//! the listener half can re-use server pipes, but for now I err'd on the simple
//! side of things. Each stream accepted by a listener will destroy the server
//! pipe after the stream is dropped.
//!
//! This model ends up having a small race or two, and you can find more details
//! on the `native_accept` method.
//!
//! # Simultaneous reads and writes
//!
//! In testing, I found that two simultaneous writes and two simultaneous reads
//! on a pipe ended up working out just fine, but problems were encountered when
//! a read was executed simultaneously with a write. After some googling around,
//! it sounded like named pipes just weren't built for this kind of interaction,
//! and the suggested solution was to use overlapped I/O.
//!
//! I don't really know what overlapped I/O is, but my basic understanding after
//! reading about it is that you have an external Event which is used to signal
//! I/O completion, passed around in some OVERLAPPED structures. As to what this
//! is, I'm not exactly sure.
//!
//! This problem implies that all named pipes are created with the
//! FILE_FLAG_OVERLAPPED option. This means that all of their I/O is
//! asynchronous. Each I/O operation has an associated OVERLAPPED structure, and
//! inside of this structure is a HANDLE from CreateEvent. After the I/O is
//! determined to be pending (may complete in the future), the
//! GetOverlappedResult function is used to block on the event, waiting for the
//! I/O to finish.
//!
//! This scheme ended up working well enough. There were two snags that I ran
//! into, however:
//!
//! * Each UnixStream instance needs its own read/write events to wait on. These
//!   can't be shared among clones of the same stream because the documentation
//!   states that it unsets the event when the I/O is started (would possibly
//!   corrupt other events simultaneously waiting). For convenience's sake,
//!   these events are lazily initialized.
//!
//! * Each server pipe needs to be created with FILE_FLAG_OVERLAPPED in addition
//!   to all pipes created through `connect`. Notably this means that the
//!   ConnectNamedPipe function is nonblocking, implying that the Listener needs
//!   to have yet another event to do the actual blocking.
//!
//! # Conclusion
//!
//! The conclusion here is that I probably don't know the best way to work with
//! windows named pipes, but the solution here seems to work well enough to get
//! the test suite passing (the suite is in libstd), and that's good enough for
//! me!

use alloc::arc::Arc;
use libc;
use std::c_str::CString;
use std::mem;
use std::os;
use std::ptr;
use std::rt::rtio;
use std::rt::rtio::{IoResult, IoError};
use std::sync::atomic;
use std::rt::mutex;

use super::c;
use super::util;
use super::file::to_utf16;

struct Event(libc::HANDLE);

impl Event {
    fn new(manual_reset: bool, initial_state: bool) -> IoResult<Event> {
        let event = unsafe {
            libc::CreateEventW(ptr::mut_null(),
                               manual_reset as libc::BOOL,
                               initial_state as libc::BOOL,
                               ptr::null())
        };
        if event as uint == 0 {
            Err(super::last_error())
        } else {
            Ok(Event(event))
        }
    }

    fn handle(&self) -> libc::HANDLE { let Event(handle) = *self; handle }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe { let _ = libc::CloseHandle(self.handle()); }
    }
}

struct Inner {
    handle: libc::HANDLE,
    lock: mutex::NativeMutex,
    read_closed: atomic::AtomicBool,
    write_closed: atomic::AtomicBool,
}

impl Inner {
    fn new(handle: libc::HANDLE) -> Inner {
        Inner {
            handle: handle,
            lock: unsafe { mutex::NativeMutex::new() },
            read_closed: atomic::AtomicBool::new(false),
            write_closed: atomic::AtomicBool::new(false),
        }
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            let _ = libc::FlushFileBuffers(self.handle);
            let _ = libc::CloseHandle(self.handle);
        }
    }
}

unsafe fn pipe(name: *const u16, init: bool) -> libc::HANDLE {
    libc::CreateNamedPipeW(
        name,
        libc::PIPE_ACCESS_DUPLEX |
            if init {libc::FILE_FLAG_FIRST_PIPE_INSTANCE} else {0} |
            libc::FILE_FLAG_OVERLAPPED,
        libc::PIPE_TYPE_BYTE | libc::PIPE_READMODE_BYTE |
            libc::PIPE_WAIT,
        libc::PIPE_UNLIMITED_INSTANCES,
        65536,
        65536,
        0,
        ptr::mut_null()
    )
}

pub fn await(handle: libc::HANDLE, deadline: u64,
             events: &[libc::HANDLE]) -> IoResult<uint> {
    use libc::consts::os::extra::{WAIT_FAILED, WAIT_TIMEOUT, WAIT_OBJECT_0};

    // If we've got a timeout, use WaitForSingleObject in tandem with CancelIo
    // to figure out if we should indeed get the result.
    let ms = if deadline == 0 {
        libc::INFINITE as u64
    } else {
        let now = ::io::timer::now();
        if deadline < now {0} else {deadline - now}
    };
    let ret = unsafe {
        c::WaitForMultipleObjects(events.len() as libc::DWORD,
                                  events.as_ptr(),
                                  libc::FALSE,
                                  ms as libc::DWORD)
    };
    match ret {
        WAIT_FAILED => Err(super::last_error()),
        WAIT_TIMEOUT => unsafe {
            let _ = c::CancelIo(handle);
            Err(util::timeout("operation timed out"))
        },
        n => Ok((n - WAIT_OBJECT_0) as uint)
    }
}

fn epipe() -> IoError {
    IoError {
        code: libc::ERROR_BROKEN_PIPE as uint,
        extra: 0,
        detail: None,
    }
}

////////////////////////////////////////////////////////////////////////////////
// Unix Streams
////////////////////////////////////////////////////////////////////////////////

pub struct UnixStream {
    inner: Arc<Inner>,
    write: Option<Event>,
    read: Option<Event>,
    read_deadline: u64,
    write_deadline: u64,
}

impl UnixStream {
    fn try_connect(p: *const u16) -> Option<libc::HANDLE> {
        // Note that most of this is lifted from the libuv implementation.
        // The idea is that if we fail to open a pipe in read/write mode
        // that we try afterwards in just read or just write
        let mut result = unsafe {
            libc::CreateFileW(p,
                libc::GENERIC_READ | libc::GENERIC_WRITE,
                0,
                ptr::mut_null(),
                libc::OPEN_EXISTING,
                libc::FILE_FLAG_OVERLAPPED,
                ptr::mut_null())
        };
        if result != libc::INVALID_HANDLE_VALUE {
            return Some(result)
        }

        let err = unsafe { libc::GetLastError() };
        if err == libc::ERROR_ACCESS_DENIED as libc::DWORD {
            result = unsafe {
                libc::CreateFileW(p,
                    libc::GENERIC_READ | libc::FILE_WRITE_ATTRIBUTES,
                    0,
                    ptr::mut_null(),
                    libc::OPEN_EXISTING,
                    libc::FILE_FLAG_OVERLAPPED,
                    ptr::mut_null())
            };
            if result != libc::INVALID_HANDLE_VALUE {
                return Some(result)
            }
        }
        let err = unsafe { libc::GetLastError() };
        if err == libc::ERROR_ACCESS_DENIED as libc::DWORD {
            result = unsafe {
                libc::CreateFileW(p,
                    libc::GENERIC_WRITE | libc::FILE_READ_ATTRIBUTES,
                    0,
                    ptr::mut_null(),
                    libc::OPEN_EXISTING,
                    libc::FILE_FLAG_OVERLAPPED,
                    ptr::mut_null())
            };
            if result != libc::INVALID_HANDLE_VALUE {
                return Some(result)
            }
        }
        None
    }

    pub fn connect(addr: &CString, timeout: Option<u64>) -> IoResult<UnixStream> {
        let addr = try!(to_utf16(addr));
        let start = ::io::timer::now();
        loop {
            match UnixStream::try_connect(addr.as_ptr()) {
                Some(handle) => {
                    let inner = Inner::new(handle);
                    let mut mode = libc::PIPE_TYPE_BYTE |
                                   libc::PIPE_READMODE_BYTE |
                                   libc::PIPE_WAIT;
                    let ret = unsafe {
                        libc::SetNamedPipeHandleState(inner.handle,
                                                      &mut mode,
                                                      ptr::mut_null(),
                                                      ptr::mut_null())
                    };
                    return if ret == 0 {
                        Err(super::last_error())
                    } else {
                        Ok(UnixStream {
                            inner: Arc::new(inner),
                            read: None,
                            write: None,
                            read_deadline: 0,
                            write_deadline: 0,
                        })
                    }
                }
                None => {}
            }

            // On windows, if you fail to connect, you may need to call the
            // `WaitNamedPipe` function, and this is indicated with an error
            // code of ERROR_PIPE_BUSY.
            let code = unsafe { libc::GetLastError() };
            if code as int != libc::ERROR_PIPE_BUSY as int {
                return Err(super::last_error())
            }

            match timeout {
                Some(timeout) => {
                    let now = ::io::timer::now();
                    let timed_out = (now - start) >= timeout || unsafe {
                        let ms = (timeout - (now - start)) as libc::DWORD;
                        libc::WaitNamedPipeW(addr.as_ptr(), ms) == 0
                    };
                    if timed_out {
                        return Err(util::timeout("connect timed out"))
                    }
                }

                // An example I found on Microsoft's website used 20
                // seconds, libuv uses 30 seconds, hence we make the
                // obvious choice of waiting for 25 seconds.
                None => {
                    if unsafe { libc::WaitNamedPipeW(addr.as_ptr(), 25000) } == 0 {
                        return Err(super::last_error())
                    }
                }
            }
        }
    }

    fn handle(&self) -> libc::HANDLE { self.inner.handle }

    fn read_closed(&self) -> bool {
        self.inner.read_closed.load(atomic::SeqCst)
    }

    fn write_closed(&self) -> bool {
        self.inner.write_closed.load(atomic::SeqCst)
    }

    fn cancel_io(&self) -> IoResult<()> {
        match unsafe { c::CancelIoEx(self.handle(), ptr::mut_null()) } {
            0 if os::errno() == libc::ERROR_NOT_FOUND as uint => {
                Ok(())
            }
            0 => Err(super::last_error()),
            _ => Ok(())
        }
    }
}

impl rtio::RtioPipe for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        if self.read.is_none() {
            self.read = Some(try!(Event::new(true, false)));
        }

        let mut bytes_read = 0;
        let mut overlapped: libc::OVERLAPPED = unsafe { mem::zeroed() };
        overlapped.hEvent = self.read.get_ref().handle();

        // Pre-flight check to see if the reading half has been closed. This
        // must be done before issuing the ReadFile request, but after we
        // acquire the lock.
        //
        // See comments in close_read() about why this lock is necessary.
        let guard = unsafe { self.inner.lock.lock() };
        if self.read_closed() {
            return Err(util::eof())
        }

        // Issue a nonblocking requests, succeeding quickly if it happened to
        // succeed.
        let ret = unsafe {
            libc::ReadFile(self.handle(),
                           buf.as_ptr() as libc::LPVOID,
                           buf.len() as libc::DWORD,
                           &mut bytes_read,
                           &mut overlapped)
        };
        if ret != 0 { return Ok(bytes_read as uint) }

        // If our errno doesn't say that the I/O is pending, then we hit some
        // legitimate error and return immediately.
        if os::errno() != libc::ERROR_IO_PENDING as uint {
            return Err(super::last_error())
        }

        // Now that we've issued a successful nonblocking request, we need to
        // wait for it to finish. This can all be done outside the lock because
        // we'll see any invocation of CancelIoEx. We also call this in a loop
        // because we're woken up if the writing half is closed, we just need to
        // realize that the reading half wasn't closed and we go right back to
        // sleep.
        drop(guard);
        loop {
            // Process a timeout if one is pending
            let wait_succeeded = await(self.handle(), self.read_deadline,
                                       [overlapped.hEvent]);

            let ret = unsafe {
                libc::GetOverlappedResult(self.handle(),
                                          &mut overlapped,
                                          &mut bytes_read,
                                          libc::TRUE)
            };
            // If we succeeded, or we failed for some reason other than
            // CancelIoEx, return immediately
            if ret != 0 { return Ok(bytes_read as uint) }
            if os::errno() != libc::ERROR_OPERATION_ABORTED as uint {
                return Err(super::last_error())
            }

            // If the reading half is now closed, then we're done. If we woke up
            // because the writing half was closed, keep trying.
            if wait_succeeded.is_err() {
                return Err(util::timeout("read timed out"))
            }
            if self.read_closed() {
                return Err(util::eof())
            }
        }
    }

    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        if self.write.is_none() {
            self.write = Some(try!(Event::new(true, false)));
        }

        let mut offset = 0;
        let mut overlapped: libc::OVERLAPPED = unsafe { mem::zeroed() };
        overlapped.hEvent = self.write.get_ref().handle();

        while offset < buf.len() {
            let mut bytes_written = 0;

            // This sequence below is quite similar to the one found in read().
            // Some careful looping is done to ensure that if close_write() is
            // invoked we bail out early, and if close_read() is invoked we keep
            // going after we woke up.
            //
            // See comments in close_read() about why this lock is necessary.
            let guard = unsafe { self.inner.lock.lock() };
            if self.write_closed() {
                return Err(epipe())
            }
            let ret = unsafe {
                libc::WriteFile(self.handle(),
                                buf.slice_from(offset).as_ptr() as libc::LPVOID,
                                (buf.len() - offset) as libc::DWORD,
                                &mut bytes_written,
                                &mut overlapped)
            };
            let err = os::errno();
            drop(guard);

            if ret == 0 {
                if err != libc::ERROR_IO_PENDING as uint {
                    return Err(IoError {
                        code: err as uint,
                        extra: 0,
                        detail: Some(os::error_string(err as uint)),
                    })
                }
                // Process a timeout if one is pending
                let wait_succeeded = await(self.handle(), self.write_deadline,
                                           [overlapped.hEvent]);
                let ret = unsafe {
                    libc::GetOverlappedResult(self.handle(),
                                              &mut overlapped,
                                              &mut bytes_written,
                                              libc::TRUE)
                };
                // If we weren't aborted, this was a legit error, if we were
                // aborted, then check to see if the write half was actually
                // closed or whether we woke up from the read half closing.
                if ret == 0 {
                    if os::errno() != libc::ERROR_OPERATION_ABORTED as uint {
                        return Err(super::last_error())
                    }
                    if !wait_succeeded.is_ok() {
                        let amt = offset + bytes_written as uint;
                        return if amt > 0 {
                            Err(IoError {
                                code: libc::ERROR_OPERATION_ABORTED as uint,
                                extra: amt,
                                detail: Some("short write during write".to_string()),
                            })
                        } else {
                            Err(util::timeout("write timed out"))
                        }
                    }
                    if self.write_closed() {
                        return Err(epipe())
                    }
                    continue // retry
                }
            }
            offset += bytes_written as uint;
        }
        Ok(())
    }

    fn clone(&self) -> Box<rtio::RtioPipe + Send> {
        box UnixStream {
            inner: self.inner.clone(),
            read: None,
            write: None,
            read_deadline: 0,
            write_deadline: 0,
        } as Box<rtio::RtioPipe + Send>
    }

    fn close_read(&mut self) -> IoResult<()> {
        // On windows, there's no actual shutdown() method for pipes, so we're
        // forced to emulate the behavior manually at the application level. To
        // do this, we need to both cancel any pending requests, as well as
        // prevent all future requests from succeeding. These two operations are
        // not atomic with respect to one another, so we must use a lock to do
        // so.
        //
        // The read() code looks like:
        //
        //      1. Make sure the pipe is still open
        //      2. Submit a read request
        //      3. Wait for the read request to finish
        //
        // The race this lock is preventing is if another thread invokes
        // close_read() between steps 1 and 2. By atomically executing steps 1
        // and 2 with a lock with respect to close_read(), we're guaranteed that
        // no thread will erroneously sit in a read forever.
        let _guard = unsafe { self.inner.lock.lock() };
        self.inner.read_closed.store(true, atomic::SeqCst);
        self.cancel_io()
    }

    fn close_write(&mut self) -> IoResult<()> {
        // see comments in close_read() for why this lock is necessary
        let _guard = unsafe { self.inner.lock.lock() };
        self.inner.write_closed.store(true, atomic::SeqCst);
        self.cancel_io()
    }

    fn set_timeout(&mut self, timeout: Option<u64>) {
        let deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
        self.read_deadline = deadline;
        self.write_deadline = deadline;
    }
    fn set_read_timeout(&mut self, timeout: Option<u64>) {
        self.read_deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
    }
    fn set_write_timeout(&mut self, timeout: Option<u64>) {
        self.write_deadline = timeout.map(|a| ::io::timer::now() + a).unwrap_or(0);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Unix Listener
////////////////////////////////////////////////////////////////////////////////

pub struct UnixListener {
    handle: libc::HANDLE,
    name: CString,
}

impl UnixListener {
    pub fn bind(addr: &CString) -> IoResult<UnixListener> {
        // Although we technically don't need the pipe until much later, we
        // create the initial handle up front to test the validity of the name
        // and such.
        let addr_v = try!(to_utf16(addr));
        let ret = unsafe { pipe(addr_v.as_ptr(), true) };
        if ret == libc::INVALID_HANDLE_VALUE {
            Err(super::last_error())
        } else {
            Ok(UnixListener { handle: ret, name: addr.clone() })
        }
    }

    pub fn native_listen(self) -> IoResult<UnixAcceptor> {
        Ok(UnixAcceptor {
            listener: self,
            event: try!(Event::new(true, false)),
            deadline: 0,
            inner: Arc::new(AcceptorState {
                abort: try!(Event::new(true, false)),
                closed: atomic::AtomicBool::new(false),
            }),
        })
    }
}

impl Drop for UnixListener {
    fn drop(&mut self) {
        unsafe { let _ = libc::CloseHandle(self.handle); }
    }
}

impl rtio::RtioUnixListener for UnixListener {
    fn listen(self: Box<UnixListener>)
              -> IoResult<Box<rtio::RtioUnixAcceptor + Send>> {
        self.native_listen().map(|a| {
            box a as Box<rtio::RtioUnixAcceptor + Send>
        })
    }
}

pub struct UnixAcceptor {
    inner: Arc<AcceptorState>,
    listener: UnixListener,
    event: Event,
    deadline: u64,
}

struct AcceptorState {
    abort: Event,
    closed: atomic::AtomicBool,
}

impl UnixAcceptor {
    pub fn native_accept(&mut self) -> IoResult<UnixStream> {
        // This function has some funky implementation details when working with
        // unix pipes. On windows, each server named pipe handle can be
        // connected to a one or zero clients. To the best of my knowledge, a
        // named server is considered active and present if there exists at
        // least one server named pipe for it.
        //
        // The model of this function is to take the current known server
        // handle, connect a client to it, and then transfer ownership to the
        // UnixStream instance. The next time accept() is invoked, it'll need a
        // different server handle to connect a client to.
        //
        // Note that there is a possible race here. Once our server pipe is
        // handed off to a `UnixStream` object, the stream could be closed,
        // meaning that there would be no active server pipes, hence even though
        // we have a valid `UnixAcceptor`, no one can connect to it. For this
        // reason, we generate the next accept call's server pipe at the end of
        // this function call.
        //
        // This provides us an invariant that we always have at least one server
        // connection open at a time, meaning that all connects to this acceptor
        // should succeed while this is active.
        //
        // The actual implementation of doing this is a little tricky. Once a
        // server pipe is created, a client can connect to it at any time. I
        // assume that which server a client connects to is nondeterministic, so
        // we also need to guarantee that the only server able to be connected
        // to is the one that we're calling ConnectNamedPipe on. This means that
        // we have to create the second server pipe *after* we've already
        // accepted a connection. In order to at least somewhat gracefully
        // handle errors, this means that if the second server pipe creation
        // fails that we disconnect the connected client and then just keep
        // using the original server pipe.
        let handle = self.listener.handle;

        // If we've had an artifical call to close_accept, be sure to never
        // proceed in accepting new clients in the future
        if self.inner.closed.load(atomic::SeqCst) { return Err(util::eof()) }

        let name = try!(to_utf16(&self.listener.name));

        // Once we've got a "server handle", we need to wait for a client to
        // connect. The ConnectNamedPipe function will block this thread until
        // someone on the other end connects. This function can "fail" if a
        // client connects after we created the pipe but before we got down
        // here. Thanks windows.
        let mut overlapped: libc::OVERLAPPED = unsafe { mem::zeroed() };
        overlapped.hEvent = self.event.handle();
        if unsafe { libc::ConnectNamedPipe(handle, &mut overlapped) == 0 } {
            let mut err = unsafe { libc::GetLastError() };

            if err == libc::ERROR_IO_PENDING as libc::DWORD {
                // Process a timeout if one is pending
                let wait_succeeded = await(handle, self.deadline,
                                           [self.inner.abort.handle(),
                                            overlapped.hEvent]);

                // This will block until the overlapped I/O is completed. The
                // timeout was previously handled, so this will either block in
                // the normal case or succeed very quickly in the timeout case.
                let ret = unsafe {
                    let mut transfer = 0;
                    libc::GetOverlappedResult(handle,
                                              &mut overlapped,
                                              &mut transfer,
                                              libc::TRUE)
                };
                if ret == 0 {
                    if wait_succeeded.is_ok() {
                        err = unsafe { libc::GetLastError() };
                    } else {
                        return Err(util::timeout("accept timed out"))
                    }
                } else {
                    // we succeeded, bypass the check below
                    err = libc::ERROR_PIPE_CONNECTED as libc::DWORD;
                }
            }
            if err != libc::ERROR_PIPE_CONNECTED as libc::DWORD {
                return Err(super::last_error())
            }
        }

        // Now that we've got a connected client to our handle, we need to
        // create a second server pipe. If this fails, we disconnect the
        // connected client and return an error (see comments above).
        let new_handle = unsafe { pipe(name.as_ptr(), false) };
        if new_handle == libc::INVALID_HANDLE_VALUE {
            let ret = Err(super::last_error());
            // If our disconnection fails, then there's not really a whole lot
            // that we can do, so fail the task.
            let err = unsafe { libc::DisconnectNamedPipe(handle) };
            assert!(err != 0);
            return ret;
        } else {
            self.listener.handle = new_handle;
        }

        // Transfer ownership of our handle into this stream
        Ok(UnixStream {
            inner: Arc::new(Inner::new(handle)),
            read: None,
            write: None,
            read_deadline: 0,
            write_deadline: 0,
        })
    }
}

impl rtio::RtioUnixAcceptor for UnixAcceptor {
    fn accept(&mut self) -> IoResult<Box<rtio::RtioPipe + Send>> {
        self.native_accept().map(|s| box s as Box<rtio::RtioPipe + Send>)
    }
    fn set_timeout(&mut self, timeout: Option<u64>) {
        self.deadline = timeout.map(|i| i + ::io::timer::now()).unwrap_or(0);
    }

    fn clone(&self) -> Box<rtio::RtioUnixAcceptor + Send> {
        let name = to_utf16(&self.listener.name).ok().unwrap();
        box UnixAcceptor {
            inner: self.inner.clone(),
            event: Event::new(true, false).ok().unwrap(),
            deadline: 0,
            listener: UnixListener {
                name: self.listener.name.clone(),
                handle: unsafe {
                    let p = pipe(name.as_ptr(), false) ;
                    assert!(p != libc::INVALID_HANDLE_VALUE as libc::HANDLE);
                    p
                },
            },
        } as Box<rtio::RtioUnixAcceptor + Send>
    }

    fn close_accept(&mut self) -> IoResult<()> {
        self.inner.closed.store(true, atomic::SeqCst);
        let ret = unsafe {
            c::SetEvent(self.inner.abort.handle())
        };
        if ret == 0 {
            Err(super::last_error())
        } else {
            Ok(())
        }
    }
}

