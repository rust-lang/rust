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
//! I don't realy know what overlapped I/O is, but my basic understanding after
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

use std::c_str::CString;
use libc;
use std::os::win32::as_utf16_p;
use std::ptr;
use std::rt::rtio;
use std::sync::arc::UnsafeArc;
use std::intrinsics;

use super::IoResult;
use super::c;
use super::util;

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
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            let _ = libc::FlushFileBuffers(self.handle);
            let _ = libc::CloseHandle(self.handle);
        }
    }
}

unsafe fn pipe(name: *u16, init: bool) -> libc::HANDLE {
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

////////////////////////////////////////////////////////////////////////////////
// Unix Streams
////////////////////////////////////////////////////////////////////////////////

pub struct UnixStream {
    inner: UnsafeArc<Inner>,
    write: Option<Event>,
    read: Option<Event>,
}

impl UnixStream {
    fn try_connect(p: *u16) -> Option<libc::HANDLE> {
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
        if result != libc::INVALID_HANDLE_VALUE as libc::HANDLE {
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
            if result != libc::INVALID_HANDLE_VALUE as libc::HANDLE {
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
            if result != libc::INVALID_HANDLE_VALUE as libc::HANDLE {
                return Some(result)
            }
        }
        None
    }

    pub fn connect(addr: &CString, timeout: Option<u64>) -> IoResult<UnixStream> {
        as_utf16_p(addr.as_str().unwrap(), |p| {
            let start = ::io::timer::now();
            loop {
                match UnixStream::try_connect(p) {
                    Some(handle) => {
                        let inner = Inner { handle: handle };
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
                                inner: UnsafeArc::new(inner),
                                read: None,
                                write: None,
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
                            libc::WaitNamedPipeW(p, ms) == 0
                        };
                        if timed_out {
                            return Err(util::timeout("connect timed out"))
                        }
                    }

                    // An example I found on microsoft's website used 20
                    // seconds, libuv uses 30 seconds, hence we make the
                    // obvious choice of waiting for 25 seconds.
                    None => {
                        if unsafe { libc::WaitNamedPipeW(p, 25000) } == 0 {
                            return Err(super::last_error())
                        }
                    }
                }
            }
        })
    }

    fn handle(&self) -> libc::HANDLE { unsafe { (*self.inner.get()).handle } }
}

impl rtio::RtioPipe for UnixStream {
    fn read(&mut self, buf: &mut [u8]) -> IoResult<uint> {
        if self.read.is_none() {
            self.read = Some(try!(Event::new(true, false)));
        }

        let mut bytes_read = 0;
        let mut overlapped: libc::OVERLAPPED = unsafe { intrinsics::init() };
        overlapped.hEvent = self.read.get_ref().handle();

        let ret = unsafe {
            libc::ReadFile(self.handle(),
                           buf.as_ptr() as libc::LPVOID,
                           buf.len() as libc::DWORD,
                           &mut bytes_read,
                           &mut overlapped)
        };
        if ret == 0 {
            let err = unsafe { libc::GetLastError() };
            if err == libc::ERROR_IO_PENDING as libc::DWORD {
                let ret = unsafe {
                    libc::GetOverlappedResult(self.handle(),
                                              &mut overlapped,
                                              &mut bytes_read,
                                              libc::TRUE)
                };
                if ret == 0 {
                    return Err(super::last_error())
                }
            } else {
                return Err(super::last_error())
            }
        }

        Ok(bytes_read as uint)
    }

    fn write(&mut self, buf: &[u8]) -> IoResult<()> {
        if self.write.is_none() {
            self.write = Some(try!(Event::new(true, false)));
        }

        let mut offset = 0;
        let mut overlapped: libc::OVERLAPPED = unsafe { intrinsics::init() };
        overlapped.hEvent = self.write.get_ref().handle();

        while offset < buf.len() {
            let mut bytes_written = 0;
            let ret = unsafe {
                libc::WriteFile(self.handle(),
                                buf.slice_from(offset).as_ptr() as libc::LPVOID,
                                (buf.len() - offset) as libc::DWORD,
                                &mut bytes_written,
                                &mut overlapped)
            };
            if ret == 0 {
                let err = unsafe { libc::GetLastError() };
                if err == libc::ERROR_IO_PENDING as libc::DWORD {
                    let ret = unsafe {
                        libc::GetOverlappedResult(self.handle(),
                                                  &mut overlapped,
                                                  &mut bytes_written,
                                                  libc::TRUE)
                    };
                    if ret == 0 {
                        return Err(super::last_error())
                    }
                } else {
                    return Err(super::last_error())
                }
            }
            offset += bytes_written as uint;
        }
        Ok(())
    }

    fn clone(&self) -> ~rtio::RtioPipe:Send {
        ~UnixStream {
            inner: self.inner.clone(),
            read: None,
            write: None,
        } as ~rtio::RtioPipe:Send
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
        as_utf16_p(addr.as_str().unwrap(), |p| {
            let ret = unsafe { pipe(p, true) };
            if ret == libc::INVALID_HANDLE_VALUE as libc::HANDLE {
                Err(super::last_error())
            } else {
                Ok(UnixListener { handle: ret, name: addr.clone() })
            }
        })
    }

    pub fn native_listen(self) -> IoResult<UnixAcceptor> {
        Ok(UnixAcceptor {
            listener: self,
            event: try!(Event::new(true, false)),
            deadline: 0,
        })
    }
}

impl Drop for UnixListener {
    fn drop(&mut self) {
        unsafe { let _ = libc::CloseHandle(self.handle); }
    }
}

impl rtio::RtioUnixListener for UnixListener {
    fn listen(~self) -> IoResult<~rtio::RtioUnixAcceptor:Send> {
        self.native_listen().map(|a| ~a as ~rtio::RtioUnixAcceptor:Send)
    }
}

pub struct UnixAcceptor {
    listener: UnixListener,
    event: Event,
    deadline: u64,
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

        // Once we've got a "server handle", we need to wait for a client to
        // connect. The ConnectNamedPipe function will block this thread until
        // someone on the other end connects. This function can "fail" if a
        // client connects after we created the pipe but before we got down
        // here. Thanks windows.
        let mut overlapped: libc::OVERLAPPED = unsafe { intrinsics::init() };
        overlapped.hEvent = self.event.handle();
        if unsafe { libc::ConnectNamedPipe(handle, &mut overlapped) == 0 } {
            let mut err = unsafe { libc::GetLastError() };

            if err == libc::ERROR_IO_PENDING as libc::DWORD {
                // If we've got a timeout, use WaitForSingleObject in tandem
                // with CancelIo to figure out if we should indeed get the
                // result.
                if self.deadline != 0 {
                    let now = ::io::timer::now();
                    let timeout = self.deadline < now || unsafe {
                        let ms = (self.deadline - now) as libc::DWORD;
                        let r = libc::WaitForSingleObject(overlapped.hEvent,
                                                          ms);
                        r != libc::WAIT_OBJECT_0
                    };
                    if timeout {
                        unsafe { let _ = c::CancelIo(handle); }
                        return Err(util::timeout("accept timed out"))
                    }
                }

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
                    err = unsafe { libc::GetLastError() };
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
        let new_handle = as_utf16_p(self.listener.name.as_str().unwrap(), |p| {
            unsafe { pipe(p, false) }
        });
        if new_handle == libc::INVALID_HANDLE_VALUE as libc::HANDLE {
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
            inner: UnsafeArc::new(Inner { handle: handle }),
            read: None,
            write: None,
        })
    }
}

impl rtio::RtioUnixAcceptor for UnixAcceptor {
    fn accept(&mut self) -> IoResult<~rtio::RtioPipe:Send> {
        self.native_accept().map(|s| ~s as ~rtio::RtioPipe:Send)
    }
    fn set_timeout(&mut self, timeout: Option<u64>) {
        self.deadline = timeout.map(|i| i + ::io::timer::now()).unwrap_or(0);
    }
}

