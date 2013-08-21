// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use option::*;
use result::*;

use rt::io::IoError;
use super::io::net::ip::{IpAddr, SocketAddr};
use rt::uv::uvio;

// XXX: ~object doesn't work currently so these are some placeholder
// types to use instead
pub type EventLoopObject = uvio::UvEventLoop;
pub type RemoteCallbackObject = uvio::UvRemoteCallback;
pub type IoFactoryObject = uvio::UvIoFactory;
pub type RtioTcpStreamObject = uvio::UvTcpStream;
pub type RtioTcpListenerObject = uvio::UvTcpListener;
pub type RtioUdpSocketObject = uvio::UvUdpSocket;
pub type RtioTimerObject = uvio::UvTimer;
pub type PausibleIdleCallback = uvio::UvPausibleIdleCallback;

pub trait EventLoop {
    fn run(&mut self);
    fn callback(&mut self, ~fn());
    fn pausible_idle_callback(&mut self) -> ~PausibleIdleCallback;
    fn callback_ms(&mut self, ms: u64, ~fn());
    fn remote_callback(&mut self, ~fn()) -> ~RemoteCallbackObject;
    /// The asynchronous I/O services. Not all event loops may provide one
    fn io<'a>(&'a mut self) -> Option<&'a mut IoFactoryObject>;
}

pub trait RemoteCallback {
    /// Trigger the remote callback. Note that the number of times the
    /// callback is run is not guaranteed. All that is guaranteed is
    /// that, after calling 'fire', the callback will be called at
    /// least once, but multiple callbacks may be coalesced and
    /// callbacks may be called more often requested. Destruction also
    /// triggers the callback.
    fn fire(&mut self);
}

pub trait IoFactory {
    fn tcp_connect(&mut self, addr: SocketAddr) -> Result<~RtioTcpStreamObject, IoError>;
    fn tcp_bind(&mut self, addr: SocketAddr) -> Result<~RtioTcpListenerObject, IoError>;
    fn udp_bind(&mut self, addr: SocketAddr) -> Result<~RtioUdpSocketObject, IoError>;
    fn timer_init(&mut self) -> Result<~RtioTimerObject, IoError>;
}

pub trait RtioTcpListener : RtioSocket {
    fn accept(&mut self) -> Result<~RtioTcpStreamObject, IoError>;
    fn accept_simultaneously(&mut self) -> Result<(), IoError>;
    fn dont_accept_simultaneously(&mut self) -> Result<(), IoError>;
}

pub trait RtioTcpStream : RtioSocket {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, IoError>;
    fn write(&mut self, buf: &[u8]) -> Result<(), IoError>;
    fn peer_name(&mut self) -> Result<SocketAddr, IoError>;
    fn control_congestion(&mut self) -> Result<(), IoError>;
    fn nodelay(&mut self) -> Result<(), IoError>;
    fn keepalive(&mut self, delay_in_seconds: uint) -> Result<(), IoError>;
    fn letdie(&mut self) -> Result<(), IoError>;
}

pub trait RtioSocket {
    fn socket_name(&mut self) -> Result<SocketAddr, IoError>;
}

pub trait RtioUdpSocket : RtioSocket {
    fn recvfrom(&mut self, buf: &mut [u8]) -> Result<(uint, SocketAddr), IoError>;
    fn sendto(&mut self, buf: &[u8], dst: SocketAddr) -> Result<(), IoError>;

    fn join_multicast(&mut self, multi: IpAddr) -> Result<(), IoError>;
    fn leave_multicast(&mut self, multi: IpAddr) -> Result<(), IoError>;

    fn loop_multicast_locally(&mut self) -> Result<(), IoError>;
    fn dont_loop_multicast_locally(&mut self) -> Result<(), IoError>;

    fn multicast_time_to_live(&mut self, ttl: int) -> Result<(), IoError>;
    fn time_to_live(&mut self, ttl: int) -> Result<(), IoError>;

    fn hear_broadcasts(&mut self) -> Result<(), IoError>;
    fn ignore_broadcasts(&mut self) -> Result<(), IoError>;
}

pub trait RtioTimer {
    fn sleep(&mut self, msecs: u64);
}
