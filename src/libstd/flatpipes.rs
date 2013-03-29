// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Generic communication channels for things that can be represented as,
or transformed to and from, byte vectors.

The `FlatPort` and `FlatChan` types implement the generic channel and
port interface for arbitrary types and transport strategies. It can
particularly be used to send and recieve serializable types over I/O
streams.

`FlatPort` and `FlatChan` implement the same comm traits as pipe-based
ports and channels.

# Example

This example sends boxed integers across tasks using serialization.

~~~
let (port, chan) = serial::pipe_stream();

do task::spawn || {
    for int::range(0, 10) |i| {
        chan.send(@i)
    }
}

for int::range(0, 10) |i| {
    assert @i == port.recv()
}
~~~

# Safety Note

Flat pipes created from `io::Reader`s and `io::Writer`s share the same
blocking properties as the underlying stream. Since some implementations
block the scheduler thread, so will their pipes.

*/

// The basic send/recv interface FlatChan and PortChan will implement
use core::io;
use core::comm::GenericChan;
use core::comm::GenericPort;
use core::prelude::*;
use core::sys::size_of;
use core::vec;

/**
A FlatPort, consisting of a `BytePort` that recieves byte vectors,
and an `Unflattener` that converts the bytes to a value.

Create using the constructors in the `serial` and `pod` modules.
*/
pub struct FlatPort<T, U, P> {
    unflattener: U,
    byte_port: P
}

/**
A FlatChan, consisting of a `Flattener` that converts values to
byte vectors, and a `ByteChan` that transmits the bytes.

Create using the constructors in the `serial` and `pod` modules.
*/
pub struct FlatChan<T, F, C> {
    flattener: F,
    byte_chan: C
}

/**
Constructors for flat pipes that using serialization-based flattening.
*/
pub mod serial {
    pub use DefaultEncoder = ebml::writer::Encoder;
    pub use DefaultDecoder = ebml::reader::Decoder;

    use serialize::{Decodable, Encodable};
    use flatpipes::flatteners::{DeserializingUnflattener,
                                SerializingFlattener};
    use flatpipes::flatteners::{deserialize_buffer, serialize_value};
    use flatpipes::bytepipes::{ReaderBytePort, WriterByteChan};
    use flatpipes::bytepipes::{PipeBytePort, PipeByteChan};
    use flatpipes::{FlatPort, FlatChan};

    use core::io::{Reader, Writer};
    use core::comm::{Port, Chan};
    use core::comm;

    pub type ReaderPort<T, R> = FlatPort<
        T, DeserializingUnflattener<DefaultDecoder, T>,
        ReaderBytePort<R>>;
    pub type WriterChan<T, W> = FlatChan<
        T, SerializingFlattener<DefaultEncoder, T>, WriterByteChan<W>>;
    pub type PipePort<T> = FlatPort<
        T, DeserializingUnflattener<DefaultDecoder, T>, PipeBytePort>;
    pub type PipeChan<T> = FlatChan<
        T, SerializingFlattener<DefaultEncoder, T>, PipeByteChan>;

    /// Create a `FlatPort` from a `Reader`
    pub fn reader_port<T: Decodable<DefaultDecoder>,
                       R: Reader>(reader: R) -> ReaderPort<T, R> {
        let unflat: DeserializingUnflattener<DefaultDecoder, T> =
            DeserializingUnflattener::new(
                deserialize_buffer::<DefaultDecoder, T>);
        let byte_port = ReaderBytePort::new(reader);
        FlatPort::new(unflat, byte_port)
    }

    /// Create a `FlatChan` from a `Writer`
    pub fn writer_chan<T: Encodable<DefaultEncoder>,
                       W: Writer>(writer: W) -> WriterChan<T, W> {
        let flat: SerializingFlattener<DefaultEncoder, T> =
            SerializingFlattener::new(
                serialize_value::<DefaultEncoder, T>);
        let byte_chan = WriterByteChan::new(writer);
        FlatChan::new(flat, byte_chan)
    }

    /// Create a `FlatPort` from a `Port<~[u8]>`
    pub fn pipe_port<T:Decodable<DefaultDecoder>>(
        port: Port<~[u8]>
    ) -> PipePort<T> {
        let unflat: DeserializingUnflattener<DefaultDecoder, T> =
            DeserializingUnflattener::new(
                deserialize_buffer::<DefaultDecoder, T>);
        let byte_port = PipeBytePort::new(port);
        FlatPort::new(unflat, byte_port)
    }

    /// Create a `FlatChan` from a `Chan<~[u8]>`
    pub fn pipe_chan<T:Encodable<DefaultEncoder>>(
        chan: Chan<~[u8]>
    ) -> PipeChan<T> {
        let flat: SerializingFlattener<DefaultEncoder, T> =
            SerializingFlattener::new(
                serialize_value::<DefaultEncoder, T>);
        let byte_chan = PipeByteChan::new(chan);
        FlatChan::new(flat, byte_chan)
    }

    /// Create a pair of `FlatChan` and `FlatPort`, backed by pipes
    pub fn pipe_stream<T: Encodable<DefaultEncoder> +
                          Decodable<DefaultDecoder>>(
                          ) -> (PipePort<T>, PipeChan<T>) {
        let (port, chan) = comm::stream();
        return (pipe_port(port), pipe_chan(chan));
    }
}

// FIXME #4074 this doesn't correctly enforce POD bounds
/**
Constructors for flat pipes that send POD types using memcpy.

# Safety Note

This module is currently unsafe because it uses `Copy Owned` as a type
parameter bounds meaning POD (plain old data), but `Copy Owned` and
POD are not equivelant.

*/
pub mod pod {
    use flatpipes::flatteners::{PodUnflattener, PodFlattener};
    use flatpipes::bytepipes::{ReaderBytePort, WriterByteChan};
    use flatpipes::bytepipes::{PipeBytePort, PipeByteChan};
    use flatpipes::{FlatPort, FlatChan};

    use core::io::{Reader, Writer};
    use core::comm::{Port, Chan};
    use core::comm;
    use core::prelude::*;

    pub type ReaderPort<T, R> =
        FlatPort<T, PodUnflattener<T>, ReaderBytePort<R>>;
    pub type WriterChan<T, W> =
        FlatChan<T, PodFlattener<T>, WriterByteChan<W>>;
    pub type PipePort<T> = FlatPort<T, PodUnflattener<T>, PipeBytePort>;
    pub type PipeChan<T> = FlatChan<T, PodFlattener<T>, PipeByteChan>;

    /// Create a `FlatPort` from a `Reader`
    pub fn reader_port<T:Copy + Owned,R:Reader>(
        reader: R
    ) -> ReaderPort<T, R> {
        let unflat: PodUnflattener<T> = PodUnflattener::new();
        let byte_port = ReaderBytePort::new(reader);
        FlatPort::new(unflat, byte_port)
    }

    /// Create a `FlatChan` from a `Writer`
    pub fn writer_chan<T:Copy + Owned,W:Writer>(
        writer: W
    ) -> WriterChan<T, W> {
        let flat: PodFlattener<T> = PodFlattener::new();
        let byte_chan = WriterByteChan::new(writer);
        FlatChan::new(flat, byte_chan)
    }

    /// Create a `FlatPort` from a `Port<~[u8]>`
    pub fn pipe_port<T:Copy + Owned>(port: Port<~[u8]>) -> PipePort<T> {
        let unflat: PodUnflattener<T> = PodUnflattener::new();
        let byte_port = PipeBytePort::new(port);
        FlatPort::new(unflat, byte_port)
    }

    /// Create a `FlatChan` from a `Chan<~[u8]>`
    pub fn pipe_chan<T:Copy + Owned>(chan: Chan<~[u8]>) -> PipeChan<T> {
        let flat: PodFlattener<T> = PodFlattener::new();
        let byte_chan = PipeByteChan::new(chan);
        FlatChan::new(flat, byte_chan)
    }

    /// Create a pair of `FlatChan` and `FlatPort`, backed by pipes
    pub fn pipe_stream<T:Copy + Owned>() -> (PipePort<T>, PipeChan<T>) {
        let (port, chan) = comm::stream();
        return (pipe_port(port), pipe_chan(chan));
    }

}

/**
Flatteners present a value as a byte vector
*/
pub trait Flattener<T> {
    fn flatten(&self, val: T) -> ~[u8];
}

/**
Unflatteners convert a byte vector to a value
*/
pub trait Unflattener<T> {
    fn unflatten(&self, buf: ~[u8]) -> T;
}

/**
BytePorts are a simple interface for receiving a specified number
*/
pub trait BytePort {
    fn try_recv(&self, count: uint) -> Option<~[u8]>;
}

/**
ByteChans are a simple interface for sending bytes
*/
pub trait ByteChan {
    fn send(&self, val: ~[u8]);
}

static CONTINUE: [u8, ..4] = [0xAA, 0xBB, 0xCC, 0xDD];

impl<T,U:Unflattener<T>,P:BytePort> GenericPort<T> for FlatPort<T, U, P> {
    fn recv(&self) -> T {
        match self.try_recv() {
            Some(val) => val,
            None => fail!(~"port is closed")
        }
    }
    fn try_recv(&self) -> Option<T> {
        let command = match self.byte_port.try_recv(CONTINUE.len()) {
            Some(c) => c,
            None => {
                warn!("flatpipe: broken pipe");
                return None;
            }
        };

        if vec::eq(command, CONTINUE) {
            let msg_len = match self.byte_port.try_recv(size_of::<u64>()) {
                Some(bytes) => {
                    io::u64_from_be_bytes(bytes, 0, size_of::<u64>())
                },
                None => {
                    warn!("flatpipe: broken pipe");
                    return None;
                }
            };

            let msg_len = msg_len as uint;

            match self.byte_port.try_recv(msg_len) {
                Some(bytes) => {
                    Some(self.unflattener.unflatten(bytes))
                }
                None => {
                    warn!("flatpipe: broken pipe");
                    return None;
                }
            }
        }
        else {
            fail!(~"flatpipe: unrecognized command");
        }
    }
}

impl<T,F:Flattener<T>,C:ByteChan> GenericChan<T> for FlatChan<T, F, C> {
    fn send(&self, val: T) {
        self.byte_chan.send(CONTINUE.to_vec());
        let bytes = self.flattener.flatten(val);
        let len = bytes.len() as u64;
        do io::u64_to_be_bytes(len, size_of::<u64>()) |len_bytes| {
            self.byte_chan.send(len_bytes.to_vec());
        }
        self.byte_chan.send(bytes);
    }
}

pub impl<T,U:Unflattener<T>,P:BytePort> FlatPort<T, U, P> {
    fn new(u: U, p: P) -> FlatPort<T, U, P> {
        FlatPort {
            unflattener: u,
            byte_port: p
        }
    }
}

pub impl<T,F:Flattener<T>,C:ByteChan> FlatChan<T, F, C> {
    fn new(f: F, c: C) -> FlatChan<T, F, C> {
        FlatChan {
            flattener: f,
            byte_chan: c
        }
    }
}


pub mod flatteners {
    use ebml;
    use flatpipes::{Flattener, Unflattener};
    use io_util::BufReader;
    use json;
    use serialize::{Encoder, Decoder, Encodable, Decodable};

    use core::cast;
    use core::io::{Writer, Reader, ReaderUtil};
    use core::prelude::*;
    use core::ptr;
    use core::sys::size_of;
    use core::vec;


    // FIXME #4074: Copy + Owned != POD
    pub struct PodUnflattener<T> {
        bogus: ()
    }

    pub struct PodFlattener<T> {
        bogus: ()
    }

    impl<T:Copy + Owned> Unflattener<T> for PodUnflattener<T> {
        fn unflatten(&self, buf: ~[u8]) -> T {
            fail_unless!(size_of::<T>() != 0);
            fail_unless!(size_of::<T>() == buf.len());
            let addr_of_init: &u8 = unsafe { &*vec::raw::to_ptr(buf) };
            let addr_of_value: &T = unsafe { cast::transmute(addr_of_init) };
            copy *addr_of_value
        }
    }

    impl<T:Copy + Owned> Flattener<T> for PodFlattener<T> {
        fn flatten(&self, val: T) -> ~[u8] {
            fail_unless!(size_of::<T>() != 0);
            let val: *T = ptr::to_unsafe_ptr(&val);
            let byte_value = val as *u8;
            unsafe { vec::from_buf(byte_value, size_of::<T>()) }
        }
    }

    pub impl<T:Copy + Owned> PodUnflattener<T> {
        fn new() -> PodUnflattener<T> {
            PodUnflattener {
                bogus: ()
            }
        }
    }

    pub impl<T:Copy + Owned> PodFlattener<T> {
        fn new() -> PodFlattener<T> {
            PodFlattener {
                bogus: ()
            }
        }
    }


    pub type DeserializeBuffer<T> = ~fn(buf: &[u8]) -> T;

    pub struct DeserializingUnflattener<D, T> {
        deserialize_buffer: DeserializeBuffer<T>
    }

    pub type SerializeValue<T> = ~fn(val: &T) -> ~[u8];

    pub struct SerializingFlattener<S, T> {
        serialize_value: SerializeValue<T>
    }

    impl<D:Decoder,T:Decodable<D>> Unflattener<T>
            for DeserializingUnflattener<D, T> {
        fn unflatten(&self, buf: ~[u8]) -> T {
            (self.deserialize_buffer)(buf)
        }
    }

    impl<S:Encoder,T:Encodable<S>> Flattener<T>
            for SerializingFlattener<S, T> {
        fn flatten(&self, val: T) -> ~[u8] {
            (self.serialize_value)(&val)
        }
    }

    pub impl<D:Decoder,T:Decodable<D>> DeserializingUnflattener<D, T> {
        fn new(deserialize_buffer: DeserializeBuffer<T>)
                   -> DeserializingUnflattener<D, T> {
            DeserializingUnflattener {
                deserialize_buffer: deserialize_buffer
            }
        }
    }

    pub impl<S:Encoder,T:Encodable<S>> SerializingFlattener<S, T> {
        fn new(serialize_value: SerializeValue<T>)
                   -> SerializingFlattener<S, T> {
            SerializingFlattener {
                serialize_value: serialize_value
            }
        }
    }

    /*
    Implementations of the serialization functions required by
    SerializingFlattener
    */

    pub fn deserialize_buffer<D: Decoder + FromReader,
                              T: Decodable<D>>(buf: &[u8]) -> T {
        let buf = vec::from_slice(buf);
        let buf_reader = @BufReader::new(buf);
        let reader = buf_reader as @Reader;
        let deser: D = FromReader::from_reader(reader);
        Decodable::decode(&deser)
    }

    pub fn serialize_value<D: Encoder + FromWriter,
                           T: Encodable<D>>(val: &T) -> ~[u8] {
        do io::with_bytes_writer |writer| {
            let ser = FromWriter::from_writer(writer);
            val.encode(&ser);
        }
    }

    pub trait FromReader {
        fn from_reader(r: @Reader) -> Self;
    }

    pub trait FromWriter {
        fn from_writer(w: @Writer) -> Self;
    }

    impl<'self> FromReader for json::Decoder<'self> {
        fn from_reader(r: @Reader) -> json::Decoder<'self> {
            match json::from_reader(r) {
                Ok(json) => {
                    json::Decoder(json)
                }
                Err(e) => fail!(fmt!("flatpipe: can't parse json: %?", e))
            }
        }
    }

    impl FromWriter for json::Encoder {
        fn from_writer(w: @Writer) -> json::Encoder {
            json::Encoder(w)
        }
    }

    impl FromReader for ebml::reader::Decoder {
        fn from_reader(r: @Reader) -> ebml::reader::Decoder {
            let buf = @r.read_whole_stream();
            let doc = ebml::reader::Doc(buf);
            ebml::reader::Decoder(doc)
        }
    }

    impl FromWriter for ebml::writer::Encoder {
        fn from_writer(w: @Writer) -> ebml::writer::Encoder {
            ebml::writer::Encoder(w)
        }
    }

}

pub mod bytepipes {
    use flatpipes::{ByteChan, BytePort};

    use core::io::{Writer, Reader, ReaderUtil};
    use core::comm::{Port, Chan};
    use core::prelude::*;

    pub struct ReaderBytePort<R> {
        reader: R
    }

    pub struct WriterByteChan<W> {
        writer: W
    }

    impl<R:Reader> BytePort for ReaderBytePort<R> {
        fn try_recv(&self, count: uint) -> Option<~[u8]> {
            let mut left = count;
            let mut bytes = ~[];
            while !self.reader.eof() && left > 0 {
                fail_unless!(left <= count);
                fail_unless!(left > 0);
                let new_bytes = self.reader.read_bytes(left);
                bytes.push_all(new_bytes);
                fail_unless!(new_bytes.len() <= left);
                left -= new_bytes.len();
            }

            if left == 0 {
                return Some(bytes);
            } else {
                warn!("flatpipe: dropped %? broken bytes", left);
                return None;
            }
        }
    }

    impl<W:Writer> ByteChan for WriterByteChan<W> {
        fn send(&self, val: ~[u8]) {
            self.writer.write(val);
        }
    }

    pub impl<R:Reader> ReaderBytePort<R> {
        fn new(r: R) -> ReaderBytePort<R> {
            ReaderBytePort {
                reader: r
            }
        }
    }

    pub impl<W:Writer> WriterByteChan<W> {
        fn new(w: W) -> WriterByteChan<W> {
            WriterByteChan {
                writer: w
            }
        }
    }

    pub struct PipeBytePort {
        port: comm::Port<~[u8]>,
        mut buf: ~[u8]
    }

    pub struct PipeByteChan {
        chan: comm::Chan<~[u8]>
    }

    impl BytePort for PipeBytePort {
        fn try_recv(&self, count: uint) -> Option<~[u8]> {
            if vec::uniq_len(&const self.buf) >= count {
                let mut bytes = ::core::util::replace(&mut self.buf, ~[]);
                self.buf = bytes.slice(count, bytes.len()).to_owned();
                bytes.truncate(count);
                return Some(bytes);
            } else if vec::uniq_len(&const self.buf) > 0 {
                let mut bytes = ::core::util::replace(&mut self.buf, ~[]);
                fail_unless!(count > bytes.len());
                match self.try_recv(count - bytes.len()) {
                    Some(rest) => {
                        bytes.push_all(rest);
                        return Some(bytes);
                    }
                    None => return None
                }
            } else if vec::uniq_len(&const self.buf) == 0 {
                match self.port.try_recv() {
                    Some(buf) => {
                        fail_unless!(!buf.is_empty());
                        self.buf = buf;
                        return self.try_recv(count);
                    }
                    None => return None
                }
            } else {
                ::core::util::unreachable()
            }
        }
    }

    impl ByteChan for PipeByteChan {
        fn send(&self, val: ~[u8]) {
            self.chan.send(val)
        }
    }

    pub impl PipeBytePort {
        fn new(p: Port<~[u8]>) -> PipeBytePort {
            PipeBytePort {
                port: p,
                buf: ~[]
            }
        }
    }

    pub impl PipeByteChan {
        fn new(c: Chan<~[u8]>) -> PipeByteChan {
            PipeByteChan {
                chan: c
            }
        }
    }

}

#[cfg(test)]
mod test {
    use core::prelude::*;

    use flatpipes::{Flattener, Unflattener};
    use flatpipes::bytepipes::*;
    use flatpipes::pod;
    use flatpipes::serial;
    use io_util::BufReader;
    use flatpipes::{BytePort, FlatChan, FlatPort};
    use net::tcp::TcpSocketBuf;

    use core::int;
    use core::io::BytesWriter;
    use core::prelude::*;
    use core::result;
    use core::task;

    #[test]
    fn test_serializing_memory_stream() {
        let writer = BytesWriter();
        let chan = serial::writer_chan(writer);

        chan.send(10);

        let bytes = copy chan.byte_chan.writer.bytes;

        let reader = BufReader::new(bytes);
        let port = serial::reader_port(reader);

        let res: int = port.recv();
        fail_unless!(res == 10i);
    }

    #[test]
    fn test_serializing_pipes() {
        let (port, chan) = serial::pipe_stream();

        do task::spawn || {
            for int::range(0, 10) |i| {
                chan.send(i)
            }
        }

        for int::range(0, 10) |i| {
            fail_unless!(i == port.recv())
        }
    }

    #[test]
    fn test_serializing_boxes() {
        let (port, chan) = serial::pipe_stream();

        do task::spawn || {
            for int::range(0, 10) |i| {
                chan.send(@i)
            }
        }

        for int::range(0, 10) |i| {
            fail_unless!(@i == port.recv())
        }
    }

    #[test]
    fn test_pod_memory_stream() {
        let writer = BytesWriter();
        let chan = pod::writer_chan(writer);

        chan.send(10);

        let bytes = copy chan.byte_chan.writer.bytes;

        let reader = BufReader::new(bytes);
        let port = pod::reader_port(reader);

        let res: int = port.recv();
        fail_unless!(res == 10);
    }

    #[test]
    fn test_pod_pipes() {
        let (port, chan) = pod::pipe_stream();

        do task::spawn || {
            for int::range(0, 10) |i| {
                chan.send(i)
            }
        }

        for int::range(0, 10) |i| {
            fail_unless!(i == port.recv())
        }
    }

    // FIXME #2064: Networking doesn't work on x86
    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_pod_tcp_stream() {
        fn reader_port(buf: TcpSocketBuf
                      ) -> pod::ReaderPort<int, TcpSocketBuf> {
            pod::reader_port(buf)
        }
        fn writer_chan(buf: TcpSocketBuf
                      ) -> pod::WriterChan<int, TcpSocketBuf> {
            pod::writer_chan(buf)
        }
        test_some_tcp_stream(reader_port, writer_chan, 9666);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_serializing_tcp_stream() {
        fn reader_port(buf: TcpSocketBuf
                      ) -> serial::ReaderPort<int, TcpSocketBuf> {
            serial::reader_port(buf)
        }
        fn writer_chan(buf: TcpSocketBuf
                      ) -> serial::WriterChan<int, TcpSocketBuf> {
            serial::writer_chan(buf)
        }
        test_some_tcp_stream(reader_port, writer_chan, 9667);
    }

    type ReaderPortFactory<U> =
        ~fn(TcpSocketBuf) -> FlatPort<int, U, ReaderBytePort<TcpSocketBuf>>;
    type WriterChanFactory<F> =
        ~fn(TcpSocketBuf) -> FlatChan<int, F, WriterByteChan<TcpSocketBuf>>;

    fn test_some_tcp_stream<U:Unflattener<int>,F:Flattener<int>>(
        reader_port: ReaderPortFactory<U>,
        writer_chan: WriterChanFactory<F>,
        port: uint) {

        use core::cell::Cell;
        use net::ip;
        use net::tcp;
        use uv;

        // Indicate to the client task that the server is listening
        let (begin_connect_port, begin_connect_chan) = comm::stream();
        // The connection is sent from the server task to the receiver task
        // to handle the connection
        let (accept_port, accept_chan) = comm::stream();
        // The main task will wait until the test is over to proceed
        let (finish_port, finish_chan) = comm::stream();

        let addr0 = ip::v4::parse_addr("127.0.0.1");

        let begin_connect_chan = Cell(begin_connect_chan);
        let accept_chan = Cell(accept_chan);

        // The server task
        let addr = copy addr0;
        do task::spawn || {
            let iotask = &uv::global_loop::get();
            let begin_connect_chan = begin_connect_chan.take();
            let accept_chan = accept_chan.take();
            let listen_res = do tcp::listen(
                copy addr, port, 128, iotask, |_kill_ch| {
                    // Tell the sender to initiate the connection
                    debug!("listening");
                    begin_connect_chan.send(())
                }) |new_conn, kill_ch| {

                // Incoming connection. Send it to the receiver task to accept
                let (res_port, res_chan) = comm::stream();
                accept_chan.send((new_conn, res_chan));
                // Wait until the connection is accepted
                res_port.recv();

                // Stop listening
                kill_ch.send(None)
            };

            fail_unless!(listen_res.is_ok());
        }

        // Client task
        let addr = copy addr0;
        do task::spawn || {

            // Wait for the server to start listening
            begin_connect_port.recv();

            debug!("connecting");
            let iotask = &uv::global_loop::get();
            let connect_result = tcp::connect(copy addr, port, iotask);
            fail_unless!(connect_result.is_ok());
            let sock = result::unwrap(connect_result);
            let socket_buf: tcp::TcpSocketBuf = tcp::socket_buf(sock);

            // TcpSocketBuf is a Writer!
            let chan = writer_chan(socket_buf);

            for int::range(0, 10) |i| {
                debug!("sending %?", i);
                chan.send(i)
            }
        }

        // Reciever task
        do task::spawn || {
            // Wait for a connection
            let (conn, res_chan) = accept_port.recv();

            debug!("accepting connection");
            let accept_result = tcp::accept(conn);
            debug!("accepted");
            fail_unless!(accept_result.is_ok());
            let sock = result::unwrap(accept_result);
            res_chan.send(());

            let socket_buf: tcp::TcpSocketBuf = tcp::socket_buf(sock);

            // TcpSocketBuf is a Reader!
            let port = reader_port(socket_buf);

            for int::range(0, 10) |i| {
                let j = port.recv();
                debug!("receieved %?", j);
                fail_unless!(i == j);
            }

            // The test is over!
            finish_chan.send(());
        }

        finish_port.recv();
    }

    // Tests that the different backends behave the same when the
    // binary streaming protocol is broken
    mod broken_protocol {
        use core::prelude::*;

        use flatpipes::{BytePort, FlatPort};
        use flatpipes::flatteners::PodUnflattener;
        use flatpipes::pod;
        use io_util::BufReader;

        use core::io;
        use core::sys;
        use core::task;

        type PortLoader<P> =
            ~fn(~[u8]) -> FlatPort<int, PodUnflattener<int>, P>;

        fn reader_port_loader(bytes: ~[u8]
                             ) -> pod::ReaderPort<int, BufReader> {
            let reader = BufReader::new(bytes);
            pod::reader_port(reader)
        }

        fn pipe_port_loader(bytes: ~[u8]
                           ) -> pod::PipePort<int> {
            let (port, chan) = comm::stream();
            if !bytes.is_empty() {
                chan.send(bytes);
            }
            pod::pipe_port(port)
        }

        fn test_try_recv_none1<P:BytePort>(loader: PortLoader<P>) {
            let bytes = ~[];
            let port = loader(bytes);
            let res: Option<int> = port.try_recv();
            fail_unless!(res.is_none());
        }

        #[test]
        fn test_try_recv_none1_reader() {
            test_try_recv_none1(reader_port_loader);
        }
        #[test]
        fn test_try_recv_none1_pipe() {
            test_try_recv_none1(pipe_port_loader);
        }

        fn test_try_recv_none2<P:BytePort>(loader: PortLoader<P>) {
            // The control word in the protocol is interrupted
            let bytes = ~[0];
            let port = loader(bytes);
            let res: Option<int> = port.try_recv();
            fail_unless!(res.is_none());
        }

        #[test]
        fn test_try_recv_none2_reader() {
            test_try_recv_none2(reader_port_loader);
        }
        #[test]
        fn test_try_recv_none2_pipe() {
            test_try_recv_none2(pipe_port_loader);
        }

        fn test_try_recv_none3<P:BytePort>(loader: PortLoader<P>) {
            static CONTINUE: [u8, ..4] = [0xAA, 0xBB, 0xCC, 0xDD];
            // The control word is followed by garbage
            let bytes = CONTINUE.to_vec() + ~[0];
            let port = loader(bytes);
            let res: Option<int> = port.try_recv();
            fail_unless!(res.is_none());
        }

        #[test]
        fn test_try_recv_none3_reader() {
            test_try_recv_none3(reader_port_loader);
        }
        #[test]
        fn test_try_recv_none3_pipe() {
            test_try_recv_none3(pipe_port_loader);
        }

        fn test_try_recv_none4<P:BytePort>(+loader: PortLoader<P>) {
            fail_unless!(do task::try || {
                static CONTINUE: [u8, ..4] = [0xAA, 0xBB, 0xCC, 0xDD];
                // The control word is followed by a valid length,
                // then undeserializable garbage
                let len_bytes = do io::u64_to_be_bytes(
                    1, sys::size_of::<u64>()) |len_bytes| {
                    len_bytes.to_vec()
                };
                let bytes = CONTINUE.to_vec() + len_bytes + ~[0, 0, 0, 0];

                let port = loader(bytes);

                let _res: Option<int> = port.try_recv();
            }.is_err());
        }

        #[test]
        #[ignore(cfg(windows))]
        fn test_try_recv_none4_reader() {
            test_try_recv_none4(reader_port_loader);
        }
        #[test]
        #[ignore(cfg(windows))]
        fn test_try_recv_none4_pipe() {
            test_try_recv_none4(pipe_port_loader);
        }
    }

}
