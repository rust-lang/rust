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
use std::io::signal::Signum;
use std::comm::SharedChan;
use std::rt::rtio::RtioSignal;

use homing::{HomingIO, HomeHandle};
use super::{UvError, UvHandle};
use uvll;
use uvio::UvIoFactory;

pub struct SignalWatcher {
    handle: *uvll::uv_signal_t,
    home: HomeHandle,

    channel: SharedChan<Signum>,
    signal: Signum,
}

impl SignalWatcher {
    pub fn new(io: &mut UvIoFactory, signum: Signum,
               channel: SharedChan<Signum>) -> Result<~SignalWatcher, UvError> {
        let s = ~SignalWatcher {
            handle: UvHandle::alloc(None::<SignalWatcher>, uvll::UV_SIGNAL),
            home: io.make_handle(),
            channel: channel,
            signal: signum,
        };
        assert_eq!(unsafe {
            uvll::uv_signal_init(io.uv_loop(), s.handle)
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
    s.channel.try_send(s.signal);
}

impl HomingIO for SignalWatcher {
    fn home<'r>(&'r mut self) -> &'r mut HomeHandle { &mut self.home }
}

impl UvHandle<uvll::uv_signal_t> for SignalWatcher {
    fn uv_handle(&self) -> *uvll::uv_signal_t { self.handle }
}

impl RtioSignal for SignalWatcher {}

impl Drop for SignalWatcher {
    fn drop(&mut self) {
        let _m = self.fire_homing_missile();
        self.close();
    }
}

#[cfg(test)]
mod test {
    use super::super::local_loop;
    use std::io::signal;
    use super::SignalWatcher;

    #[test]
    fn closing_channel_during_drop_doesnt_kill_everything() {
        // see issue #10375, relates to timers as well.
        let (port, chan) = SharedChan::new();
        let _signal = SignalWatcher::new(local_loop(), signal::Interrupt,
                                         chan);

        do spawn {
            port.try_recv();
        }

        // when we drop the SignalWatcher we're going to destroy the channel,
        // which must wake up the task on the other end
    }
}
