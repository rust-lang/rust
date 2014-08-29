// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use libc::c_int;
use std::mem;
use std::rt::task::BlockedTask;
use std::rt::rtio::IoResult;

use access;
use homing::{HomeHandle, HomingMissile};
use timer::TimerWatcher;
use uvll;
use uvio::UvIoFactory;
use {Loop, UvError, uv_error_to_io_error, Request, wakeup};
use {UvHandle, wait_until_woken_after};

/// Management of a timeout when gaining access to a portion of a duplex stream.
pub struct AccessTimeout<T> {
    state: TimeoutState,
    timer: Option<Box<TimerWatcher>>,
    pub access: access::Access<T>,
}

pub struct Guard<'a, T:'static> {
    state: &'a mut TimeoutState,
    pub access: access::Guard<'a, T>,
    pub can_timeout: bool,
}

#[deriving(PartialEq)]
enum TimeoutState {
    NoTimeout,
    TimeoutPending(ClientState),
    TimedOut,
}

#[deriving(PartialEq)]
enum ClientState {
    NoWaiter,
    AccessPending,
    RequestPending,
}

struct TimerContext {
    timeout: *mut AccessTimeout<()>,
    callback: fn(*mut AccessTimeout<()>, &TimerContext),
    user_unblock: fn(uint) -> Option<BlockedTask>,
    user_payload: uint,
}

impl<T: Send> AccessTimeout<T> {
    pub fn new(data: T) -> AccessTimeout<T> {
        AccessTimeout {
            state: NoTimeout,
            timer: None,
            access: access::Access::new(data),
        }
    }

    /// Grants access to half of a duplex stream, timing out if necessary.
    ///
    /// On success, Ok(Guard) is returned and access has been granted to the
    /// stream. If a timeout occurs, then Err is returned with an appropriate
    /// error.
    pub fn grant<'a>(&'a mut self, m: HomingMissile) -> IoResult<Guard<'a, T>> {
        // First, flag that we're attempting to acquire access. This will allow
        // us to cancel the pending grant if we timeout out while waiting for a
        // grant.
        match self.state {
            NoTimeout => {},
            TimeoutPending(ref mut client) => *client = AccessPending,
            TimedOut => return Err(uv_error_to_io_error(UvError(uvll::ECANCELED)))
        }
        let access = self.access.grant(self as *mut _ as uint, m);

        // After acquiring the grant, we need to flag ourselves as having a
        // pending request so the timeout knows to cancel the request.
        let can_timeout = match self.state {
            NoTimeout => false,
            TimeoutPending(ref mut client) => { *client = RequestPending; true }
            TimedOut => return Err(uv_error_to_io_error(UvError(uvll::ECANCELED)))
        };

        Ok(Guard {
            access: access,
            state: &mut self.state,
            can_timeout: can_timeout
        })
    }

    pub fn timed_out(&self) -> bool {
        match self.state {
            TimedOut => true,
            _ => false,
        }
    }

    /// Sets the pending timeout to the value specified.
    ///
    /// The home/loop variables are used to construct a timer if one has not
    /// been previously constructed.
    ///
    /// The callback will be invoked if the timeout elapses, and the data of
    /// the time will be set to `data`.
    pub fn set_timeout(&mut self, ms: Option<u64>,
                       home: &HomeHandle,
                       loop_: &Loop,
                       cb: fn(uint) -> Option<BlockedTask>,
                       data: uint) {
        self.state = NoTimeout;
        let ms = match ms {
            Some(ms) => ms,
            None => return match self.timer {
                Some(ref mut t) => t.stop(),
                None => {}
            }
        };

        // If we have a timeout, lazily initialize the timer which will be used
        // to fire when the timeout runs out.
        if self.timer.is_none() {
            let mut timer = box TimerWatcher::new_home(loop_, home.clone());
            let mut cx = box TimerContext {
                timeout: self as *mut _ as *mut AccessTimeout<()>,
                callback: real_cb::<T>,
                user_unblock: cb,
                user_payload: data,
            };
            unsafe {
                timer.set_data(&mut *cx);
                mem::forget(cx);
            }
            self.timer = Some(timer);
        }

        let timer = self.timer.as_mut().unwrap();
        unsafe {
            let cx = uvll::get_data_for_uv_handle(timer.handle);
            let cx = cx as *mut TimerContext;
            (*cx).user_unblock = cb;
            (*cx).user_payload = data;
        }
        timer.stop();
        timer.start(timer_cb, ms, 0);
        self.state = TimeoutPending(NoWaiter);

        extern fn timer_cb(timer: *mut uvll::uv_timer_t) {
            let cx: &TimerContext = unsafe {
                &*(uvll::get_data_for_uv_handle(timer) as *const TimerContext)
            };
            (cx.callback)(cx.timeout, cx);
        }

        fn real_cb<T: Send>(timeout: *mut AccessTimeout<()>, cx: &TimerContext) {
            let timeout = timeout as *mut AccessTimeout<T>;
            let me = unsafe { &mut *timeout };

            match mem::replace(&mut me.state, TimedOut) {
                TimedOut | NoTimeout => unreachable!(),
                TimeoutPending(NoWaiter) => {}
                TimeoutPending(AccessPending) => {
                    match unsafe { me.access.dequeue(me as *mut _ as uint) } {
                        Some(task) => task.reawaken(),
                        None => unreachable!(),
                    }
                }
                TimeoutPending(RequestPending) => {
                    match (cx.user_unblock)(cx.user_payload) {
                        Some(task) => task.reawaken(),
                        None => unreachable!(),
                    }
                }
            }
        }
    }
}

impl<T: Send> Clone for AccessTimeout<T> {
    fn clone(&self) -> AccessTimeout<T> {
        AccessTimeout {
            access: self.access.clone(),
            state: NoTimeout,
            timer: None,
        }
    }
}

#[unsafe_destructor]
impl<'a, T> Drop for Guard<'a, T> {
    fn drop(&mut self) {
        match *self.state {
            TimeoutPending(NoWaiter) | TimeoutPending(AccessPending) =>
                unreachable!(),

            NoTimeout | TimedOut => {}
            TimeoutPending(RequestPending) => {
                *self.state = TimeoutPending(NoWaiter);
            }
        }
    }
}

#[unsafe_destructor]
impl<T> Drop for AccessTimeout<T> {
    fn drop(&mut self) {
        match self.timer {
            Some(ref timer) => unsafe {
                let data = uvll::get_data_for_uv_handle(timer.handle);
                let _data: Box<TimerContext> = mem::transmute(data);
            },
            None => {}
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// Connect timeouts
////////////////////////////////////////////////////////////////////////////////

pub struct ConnectCtx {
    pub status: c_int,
    pub task: Option<BlockedTask>,
    pub timer: Option<Box<TimerWatcher>>,
}

impl ConnectCtx {
    pub fn connect<T>(
        mut self, obj: T, timeout: Option<u64>, io: &mut UvIoFactory,
        f: |&Request, &T, uvll::uv_connect_cb| -> c_int
    ) -> Result<T, UvError> {
        let mut req = Request::new(uvll::UV_CONNECT);
        let r = f(&req, &obj, connect_cb);
        return match r {
            0 => {
                req.defuse(); // uv callback now owns this request
                match timeout {
                    Some(t) => {
                        let mut timer = TimerWatcher::new(io);
                        timer.start(timer_cb, t, 0);
                        self.timer = Some(timer);
                    }
                    None => {}
                }
                wait_until_woken_after(&mut self.task, &io.loop_, || {
                    let data = &self as *const _ as *mut ConnectCtx;
                    match self.timer {
                        Some(ref mut timer) => unsafe { timer.set_data(data) },
                        None => {}
                    }
                    req.set_data(data);
                });
                // Make sure an erroneously fired callback doesn't have access
                // to the context any more.
                req.set_data(0 as *mut int);

                // If we failed because of a timeout, drop the TcpWatcher as
                // soon as possible because it's data is now set to null and we
                // want to cancel the callback ASAP.
                match self.status {
                    0 => Ok(obj),
                    n => { drop(obj); Err(UvError(n)) }
                }
            }
            n => Err(UvError(n))
        };

        extern fn timer_cb(handle: *mut uvll::uv_timer_t) {
            // Don't close the corresponding tcp request, just wake up the task
            // and let RAII take care of the pending watcher.
            let cx: &mut ConnectCtx = unsafe {
                &mut *(uvll::get_data_for_uv_handle(handle) as *mut ConnectCtx)
            };
            cx.status = uvll::ECANCELED;
            wakeup(&mut cx.task);
        }

        extern fn connect_cb(req: *mut uvll::uv_connect_t, status: c_int) {
            // This callback can be invoked with ECANCELED if the watcher is
            // closed by the timeout callback. In that case we just want to free
            // the request and be along our merry way.
            let req = Request::wrap(req);
            if status == uvll::ECANCELED { return }

            // Apparently on windows when the handle is closed this callback may
            // not be invoked with ECANCELED but rather another error code.
            // Either ways, if the data is null, then our timeout has expired
            // and there's nothing we can do.
            let data = unsafe { uvll::get_data_for_req(req.handle) };
            if data.is_null() { return }

            let cx: &mut ConnectCtx = unsafe { &mut *(data as *mut ConnectCtx) };
            cx.status = status;
            match cx.timer {
                Some(ref mut t) => t.stop(),
                None => {}
            }
            // Note that the timer callback doesn't cancel the connect request
            // (that's the job of uv_close()), so it's possible for this
            // callback to get triggered after the timeout callback fires, but
            // before the task wakes up. In that case, we did indeed
            // successfully connect, but we don't need to wake someone up. We
            // updated the status above (correctly so), and the task will pick
            // up on this when it wakes up.
            if cx.task.is_some() {
                wakeup(&mut cx.task);
            }
        }
    }
}

pub struct AcceptTimeout<T> {
    access: AccessTimeout<AcceptorState<T>>,
}

struct AcceptorState<T> {
    blocked_acceptor: Option<BlockedTask>,
    pending: Vec<IoResult<T>>,
}

impl<T: Send> AcceptTimeout<T> {
    pub fn new() -> AcceptTimeout<T> {
        AcceptTimeout {
            access: AccessTimeout::new(AcceptorState {
                blocked_acceptor: None,
                pending: Vec::new(),
            })
        }
    }

    pub fn accept(&mut self,
                  missile: HomingMissile,
                  loop_: &Loop) -> IoResult<T> {
        // If we've timed out but we're not closed yet, poll the state of the
        // queue to see if we can peel off a connection.
        if self.access.timed_out() && !self.access.access.is_closed(&missile) {
            let tmp = self.access.access.get_mut(&missile);
            return match tmp.pending.remove(0) {
                Some(msg) => msg,
                None => Err(uv_error_to_io_error(UvError(uvll::ECANCELED)))
            }
        }

        // Now that we're not polling, attempt to gain access and then peel off
        // a connection. If we have no pending connections, then we need to go
        // to sleep and wait for one.
        //
        // Note that if we're woken up for a pending connection then we're
        // guaranteed that the check above will not steal our connection due to
        // the single-threaded nature of the event loop.
        let mut guard = try!(self.access.grant(missile));
        if guard.access.is_closed() {
            return Err(uv_error_to_io_error(UvError(uvll::EOF)))
        }

        match guard.access.pending.remove(0) {
            Some(msg) => return msg,
            None => {}
        }

        wait_until_woken_after(&mut guard.access.blocked_acceptor, loop_, || {});

        match guard.access.pending.remove(0) {
            _ if guard.access.is_closed() => {
                Err(uv_error_to_io_error(UvError(uvll::EOF)))
            }
            Some(msg) => msg,
            None => Err(uv_error_to_io_error(UvError(uvll::ECANCELED)))
        }
    }

    pub unsafe fn push(&mut self, t: IoResult<T>) {
        let state = self.access.access.unsafe_get();
        (*state).pending.push(t);
        let _ = (*state).blocked_acceptor.take().map(|t| t.reawaken());
    }

    pub fn set_timeout(&mut self,
                       ms: Option<u64>,
                       loop_: &Loop,
                       home: &HomeHandle) {
        self.access.set_timeout(ms, home, loop_, cancel_accept::<T>,
                                self as *mut _ as uint);

        fn cancel_accept<T: Send>(me: uint) -> Option<BlockedTask> {
            unsafe {
                let me: &mut AcceptTimeout<T> = mem::transmute(me);
                (*me.access.access.unsafe_get()).blocked_acceptor.take()
            }
        }
    }

    pub fn close(&mut self, m: HomingMissile) {
        self.access.access.close(&m);
        let task = self.access.access.get_mut(&m).blocked_acceptor.take();
        drop(m);
        let _ = task.map(|t| t.reawaken());
    }
}

impl<T: Send> Clone for AcceptTimeout<T> {
    fn clone(&self) -> AcceptTimeout<T> {
        AcceptTimeout { access: self.access.clone() }
    }
}
