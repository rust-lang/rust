// xfail-test not a test

use std::net::ip::IpAddr;

// FIXME: ~object doesn't work currently so these are some placeholder
// types to use instead
pub type EventLoopObject = super::uvio::UvEventLoop;
pub type IoFactoryObject = super::uvio::UvIoFactory;
pub type StreamObject = super::uvio::UvStream;
pub type TcpListenerObject = super::uvio::UvTcpListener;

pub trait EventLoop {
    fn run(&mut self);
    fn callback(&mut self, ~fn());
    /// The asynchronous I/O services. Not all event loops may provide one
    /// NOTE: Should be IoFactory
    fn io(&mut self) -> Option<&self/mut IoFactoryObject>;
}

pub trait IoFactory {
    fn connect(&mut self, addr: IpAddr) -> Option<~StreamObject>;
    fn bind(&mut self, addr: IpAddr) -> Option<~TcpListenerObject>;
}

pub trait TcpListener {
    fn listen(&mut self) -> Option<~StreamObject>;
}

pub trait Stream {
    fn read(&mut self, buf: &mut [u8]) -> Result<uint, ()>;
    fn write(&mut self, buf: &[u8]) -> Result<(), ()>;
}
