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
use std::io::IoResult;
use std::mem;
use std::rt::task::BlockedTask;

use access;
use homing::{HomeHandle, HomingMissile, HomingIO};
use timer::TimerWatcher;
use uvll;
use uvio::UvIoFactory;
use {Loop, UvError, uv_error_to_io_error, Request, wakeup};
use {UvHandle, wait_until_woken_after};

/// Management of a timeout when gaining access to a portion of a duplex stream.
pub struct AccessTimeout {
    state: TimeoutState,
    timer: Option<Box<TimerWatcher>>,
    pub access: access::Access,
}

pub struct Guard<'a> {
    state: &'a mut TimeoutState,
    pub access: access::Guard<'a>,
    pub can_timeout: bool,
}

#[deriving(Eq)]
enum TimeoutState {
    NoTimeout,
    TimeoutPending(ClientState),
    TimedOut,
}

#[deriving(Eq)]
enum ClientState {
    NoWaiter,
    AccessPending,
    RequestPending,
}

struct TimerContext {
    timeout: *mut AccessTimeout,
    callback: fn(uint) -> Option<BlockedTask>,
    payload: uint,
}

impl AccessTimeout {
    pub fn new() -> AccessTimeout {
        AccessTimeout {
            state: NoTimeout,
            timer: None,
            access: access::Access::new(),
        }
    }

    /// Grants access to half of a duplex stream, timing out if necessary.
    ///
    /// On success, Ok(Guard) is returned and access has been granted to the
    /// stream. If a timeout occurs, then Err is returned with an appropriate
    /// error.
    pub fn grant<'a>(&'a mut self, m: HomingMissile) -> IoResult<Guard<'a>> {
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
            let cx = box TimerContext {
                timeout: self as *mut _,
                callback: cb,
                payload: data,
            };
            unsafe {
                timer.set_data(&*cx);
                mem::forget(cx);
            }
            self.timer = Some(timer);
        }

        let timer = self.timer.get_mut_ref();
        unsafe {
            let cx = uvll::get_data_for_uv_handle(timer.handle);
            let cx = cx as *mut TimerContext;
            (*cx).callback = cb;
            (*cx).payload = data;
        }
        timer.stop();
        timer.start(timer_cb, ms, 0);
        self.state = TimeoutPending(NoWaiter);

        extern fn timer_cb(timer: *uvll::uv_timer_t) {
            let cx: &TimerContext = unsafe {
                &*(uvll::get_data_for_uv_handle(timer) as *TimerContext)
            };
            let me = unsafe { &mut *cx.timeout };

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
                    match (cx.callback)(cx.payload) {
                        Some(task) => task.reawaken(),
                        None => unreachable!(),
                    }
                }
            }
        }
    }
}

impl Clone for AccessTimeout {
    fn clone(&self) -> AccessTimeout {
        AccessTimeout {
            access: self.access.clone(),
            state: NoTimeout,
            timer: None,
        }
    }
}

#[unsafe_destructor]
impl<'a> Drop for Guard<'a> {
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

impl Drop for AccessTimeout {
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

pub struct AcceptTimeout {
    timer: Option<TimerWatcher>,
    timeout_tx: Option<Sender<()>>,
    timeout_rx: Option<Receiver<()>>,
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
                    let data = &self as *_;
                    match self.timer {
                        Some(ref mut timer) => unsafe { timer.set_data(data) },
                        None => {}
                    }
                    req.set_data(data);
                });
                // Make sure an erroneously fired callback doesn't have access
                // to the context any more.
                req.set_data(0 as *int);

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

        extern fn timer_cb(handle: *uvll::uv_timer_t) {
            // Don't close the corresponding tcp request, just wake up the task
            // and let RAII take care of the pending watcher.
            let cx: &mut ConnectCtx = unsafe {
                &mut *(uvll::get_data_for_uv_handle(handle) as *mut ConnectCtx)
            };
            cx.status = uvll::ECANCELED;
            wakeup(&mut cx.task);
        }

        extern fn connect_cb(req: *uvll::uv_connect_t, status: c_int) {
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

impl AcceptTimeout {
    pub fn new() -> AcceptTimeout {
        AcceptTimeout { timer: None, timeout_tx: None, timeout_rx: None }
    }

    pub fn accept<T: Send>(&mut self, c: &Receiver<IoResult<T>>) -> IoResult<T> {
        match self.timeout_rx {
            None => c.recv(),
            Some(ref rx) => {
                use std::comm::Select;

                // Poll the incoming channel first (don't rely on the order of
                // select just yet). If someone's pending then we should return
                // them immediately.
                match c.try_recv() {
                    Ok(data) => return data,
                    Err(..) => {}
                }

                // Use select to figure out which channel gets ready first. We
                // do some custom handling of select to ensure that we never
                // actually drain the timeout channel (we'll keep seeing the
                // timeout message in the future).
                let s = Select::new();
                let mut timeout = s.handle(rx);
                let mut data = s.handle(c);
                unsafe {
                    timeout.add();
                    data.add();
                }
                if s.wait() == timeout.id() {
                    Err(uv_error_to_io_error(UvError(uvll::ECANCELED)))
                } else {
                    c.recv()
                }
            }
        }
    }

    pub fn clear(&mut self) {
        match self.timeout_rx {
            Some(ref t) => { let _ = t.try_recv(); }
            None => {}
        }
        match self.timer {
            Some(ref mut t) => t.stop(),
            None => {}
        }
    }

    pub fn set_timeout<U, T: UvHandle<U> + HomingIO>(
        &mut self, ms: u64, t: &mut T
    ) {
        // If we have a timeout, lazily initialize the timer which will be used
        // to fire when the timeout runs out.
        if self.timer.is_none() {
            let loop_ = Loop::wrap(unsafe {
                uvll::get_loop_for_uv_handle(t.uv_handle())
            });
            let mut timer = TimerWatcher::new_home(&loop_, t.home().clone());
            unsafe {
                timer.set_data(self as *mut _ as *AcceptTimeout);
            }
            self.timer = Some(timer);
        }

        // Once we've got a timer, stop any previous timeout, reset it for the
        // current one, and install some new channels to send/receive data on
        let timer = self.timer.get_mut_ref();
        timer.stop();
        timer.start(timer_cb, ms, 0);
        let (tx, rx) = channel();
        self.timeout_tx = Some(tx);
        self.timeout_rx = Some(rx);

        extern fn timer_cb(timer: *uvll::uv_timer_t) {
            let acceptor: &mut AcceptTimeout = unsafe {
                &mut *(uvll::get_data_for_uv_handle(timer) as *mut AcceptTimeout)
            };
            // This send can never fail because if this timer is active then the
            // receiving channel is guaranteed to be alive
            acceptor.timeout_tx.get_ref().send(());
        }
    }
}
