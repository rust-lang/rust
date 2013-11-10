// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::libc::c_int;
use std::rt::io::signal::Signum;
use std::rt::sched::{SchedHandle, Scheduler};
use std::comm::{SharedChan, SendDeferred};
use std::rt::local::Local;
use std::rt::rtio::RtioSignal;

use super::{Loop, UvError, UvHandle};
use uvll;
use uvio::HomingIO;

pub struct SignalWatcher {
    handle: *uvll::uv_signal_t,
    home: SchedHandle,

    channel: SharedChan<Signum>,
    signal: Signum,
}

impl SignalWatcher {
    pub fn new(loop_: &mut Loop, signum: Signum,
               channel: SharedChan<Signum>) -> Result<~SignalWatcher, UvError> {
        let s = ~SignalWatcher {
            handle: UvHandle::alloc(None::<SignalWatcher>, uvll::UV_SIGNAL),
            home: get_handle_to_current_scheduler!(),
            channel: channel,
            signal: signum,
        };
        assert_eq!(unsafe {
            uvll::uv_signal_init(loop_.handle, s.handle)
        }, 0);

        match unsafe {
            uvll::uv_signal_start(s.handle, signal_cb, signum as c_int)
        } {
            0 => Ok(s.install()),
            n => Err(UvError(n)),
        }

    }
}

extern fn signal_cb(handle: *uvll::uv_signal_t, signum: c_int) {
    let s: &mut SignalWatcher = unsafe { UvHandle::from_uv_handle(&handle) };
    assert_eq!(signum as int, s.signal as int);
    s.channel.send_deferred(s.signal);
}

impl HomingIO for SignalWatcher {
    fn home<'r>(&'r mut self) -> &'r mut SchedHandle { &mut self.home }
}

impl UvHandle<uvll::uv_signal_t> for SignalWatcher {
    fn uv_handle(&self) -> *uvll::uv_signal_t { self.handle }
}

impl RtioSignal for SignalWatcher {}

impl Drop for SignalWatcher {
    fn drop(&mut self) {
        let _m = self.fire_homing_missile();
        self.close_async_();
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use std::cell::Cell;
    use super::super::local_loop;
    use std::rt::io::signal;
    use std::comm::{SharedChan, stream};

    #[test]
    fn closing_channel_during_drop_doesnt_kill_everything() {
        // see issue #10375, relates to timers as well.
        let (port, chan) = stream();
        let chan = SharedChan::new(chan);
        let _signal = SignalWatcher::new(local_loop(), signal::Interrupt,
                                         chan);

        let port = Cell::new(port);
        do spawn {
            port.take().try_recv();
        }

        // when we drop the SignalWatcher we're going to destroy the channel,
        // which must wake up the task on the other end
    }
}
