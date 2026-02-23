use std::{
    collections::VecDeque,
    io::{self, BufRead, Read, Write},
    sync::{Arc, Condvar, Mutex},
    thread,
};

use paths::Utf8PathBuf;
use proc_macro_api::{
    ServerError,
    bidirectional_protocol::msg::{
        BidirectionalMessage, Request as BiRequest, Response as BiResponse, SubRequest, SubResponse,
    },
    legacy_protocol::msg::{FlatTree, Message, Request, Response, SpanDataIndexMap},
};
use span::{Edition, EditionedFileId, FileId, Span, SpanAnchor, SyntaxContext, TextRange};
use tt::{Delimiter, DelimiterKind, TopSubtreeBuilder};

/// Shared state for an in-memory byte channel.
#[derive(Default)]
struct ChannelState {
    buffer: VecDeque<u8>,
    closed: bool,
}

type InMemoryChannel = Arc<(Mutex<ChannelState>, Condvar)>;

/// Writer end of an in-memory channel.
pub(crate) struct ChannelWriter {
    state: InMemoryChannel,
}

impl Write for ChannelWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock().unwrap();
        if state.closed {
            return Err(io::Error::new(io::ErrorKind::BrokenPipe, "channel closed"));
        }
        state.buffer.extend(buf);
        cvar.notify_all();
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Drop for ChannelWriter {
    fn drop(&mut self) {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock().unwrap();
        state.closed = true;
        cvar.notify_all();
    }
}

/// Reader end of an in-memory channel.
pub(crate) struct ChannelReader {
    state: InMemoryChannel,
    internal_buf: Vec<u8>,
}

impl Read for ChannelReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock().unwrap();

        while state.buffer.is_empty() && !state.closed {
            state = cvar.wait(state).unwrap();
        }

        if state.buffer.is_empty() && state.closed {
            return Ok(0);
        }

        let to_read = buf.len().min(state.buffer.len());
        for (dst, src) in buf.iter_mut().zip(state.buffer.drain(..to_read)) {
            *dst = src;
        }
        Ok(to_read)
    }
}

impl BufRead for ChannelReader {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        let (lock, cvar) = &*self.state;
        let mut state = lock.lock().unwrap();

        while state.buffer.is_empty() && !state.closed {
            state = cvar.wait(state).unwrap();
        }

        self.internal_buf.clear();
        self.internal_buf.extend(&state.buffer);
        Ok(&self.internal_buf)
    }

    fn consume(&mut self, amt: usize) {
        let (lock, _) = &*self.state;
        let mut state = lock.lock().unwrap();
        let to_drain = amt.min(state.buffer.len());
        drop(state.buffer.drain(..to_drain));
    }
}

/// Creates a connected pair of channels for bidirectional communication.
fn create_channel_pair() -> (ChannelWriter, ChannelReader, ChannelWriter, ChannelReader) {
    // Channel for client -> server communication
    let client_to_server = Arc::new((
        Mutex::new(ChannelState { buffer: VecDeque::new(), closed: false }),
        Condvar::new(),
    ));
    let client_writer = ChannelWriter { state: client_to_server.clone() };
    let server_reader = ChannelReader { state: client_to_server, internal_buf: Vec::new() };

    // Channel for server -> client communication
    let server_to_client = Arc::new((
        Mutex::new(ChannelState { buffer: VecDeque::new(), closed: false }),
        Condvar::new(),
    ));

    let server_writer = ChannelWriter { state: server_to_client.clone() };
    let client_reader = ChannelReader { state: server_to_client, internal_buf: Vec::new() };

    (client_writer, client_reader, server_writer, server_reader)
}

pub(crate) fn proc_macro_test_dylib_path() -> Utf8PathBuf {
    let path = proc_macro_test::PROC_MACRO_TEST_LOCATION;
    if path.is_empty() {
        panic!("proc-macro-test dylib not available (requires nightly toolchain)");
    }
    path.into()
}

/// Creates a simple empty token tree suitable for testing.
pub(crate) fn create_empty_token_tree(
    version: u32,
    span_data_table: &mut SpanDataIndexMap,
) -> FlatTree {
    let anchor = SpanAnchor {
        file_id: EditionedFileId::new(FileId::from_raw(0), Edition::CURRENT),
        ast_id: span::ROOT_ERASED_FILE_AST_ID,
    };
    let span = Span {
        range: TextRange::empty(0.into()),
        anchor,
        ctx: SyntaxContext::root(Edition::CURRENT),
    };

    let builder = TopSubtreeBuilder::new(Delimiter {
        open: span,
        close: span,
        kind: DelimiterKind::Invisible,
    });
    let tt = builder.build();

    FlatTree::from_subtree(tt.view(), version, span_data_table)
}

pub(crate) fn with_server<F, R>(format: proc_macro_api::ProtocolFormat, test_fn: F) -> R
where
    F: FnOnce(&mut dyn Write, &mut dyn BufRead) -> R,
{
    let (mut client_writer, mut client_reader, mut server_writer, mut server_reader) =
        create_channel_pair();

    let server_handle = thread::spawn(move || {
        proc_macro_srv_cli::main_loop::run(&mut server_reader, &mut server_writer, format)
    });

    let result = test_fn(&mut client_writer, &mut client_reader);

    drop(client_writer);

    match server_handle.join() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            if !matches!(
                e.kind(),
                io::ErrorKind::BrokenPipe
                    | io::ErrorKind::UnexpectedEof
                    | io::ErrorKind::InvalidData
            ) {
                panic!("Server error: {e}");
            }
        }
        Err(e) => std::panic::resume_unwind(e),
    }

    result
}

trait TestProtocol {
    type Request;
    type Response;

    fn request(&self, writer: &mut dyn Write, req: Self::Request);
    fn receive(&self, reader: &mut dyn BufRead, writer: &mut dyn Write) -> Self::Response;
}

#[allow(dead_code)]
struct JsonLegacy;

impl TestProtocol for JsonLegacy {
    type Request = Request;
    type Response = Response;

    fn request(&self, writer: &mut dyn Write, req: Request) {
        req.write(writer).expect("failed to write request");
    }

    fn receive(&self, reader: &mut dyn BufRead, _writer: &mut dyn Write) -> Response {
        let mut buf = String::new();
        Response::read(reader, &mut buf)
            .expect("failed to read response")
            .expect("no response received")
    }
}

#[allow(dead_code)]
struct PostcardBidirectional<F>
where
    F: Fn(SubRequest) -> Result<SubResponse, ServerError>,
{
    callback: F,
}

impl<F> TestProtocol for PostcardBidirectional<F>
where
    F: Fn(SubRequest) -> Result<SubResponse, ServerError>,
{
    type Request = BiRequest;
    type Response = BiResponse;

    fn request(&self, writer: &mut dyn Write, req: BiRequest) {
        let msg = BidirectionalMessage::Request(req);
        msg.write(writer).expect("failed to write request");
    }

    fn receive(&self, reader: &mut dyn BufRead, writer: &mut dyn Write) -> BiResponse {
        let mut buf = Vec::new();

        loop {
            let msg = BidirectionalMessage::read(reader, &mut buf)
                .expect("failed to read message")
                .expect("no message received");

            match msg {
                BidirectionalMessage::Response(resp) => return resp,
                BidirectionalMessage::SubRequest(sr) => {
                    let reply = (self.callback)(sr).expect("subrequest callback failed");
                    let msg = BidirectionalMessage::SubResponse(reply);
                    msg.write(writer).expect("failed to write subresponse");
                }
                other => panic!("unexpected message: {other:?}"),
            }
        }
    }
}

#[allow(dead_code)]
pub(crate) fn request_legacy(
    writer: &mut dyn Write,
    reader: &mut dyn BufRead,
    request: Request,
) -> Response {
    let protocol = JsonLegacy;
    protocol.request(writer, request);
    protocol.receive(reader, writer)
}

#[allow(dead_code)]
pub(crate) fn request_bidirectional<F>(
    writer: &mut dyn Write,
    reader: &mut dyn BufRead,
    request: BiRequest,
    callback: F,
) -> BiResponse
where
    F: Fn(SubRequest) -> Result<SubResponse, ServerError>,
{
    let protocol = PostcardBidirectional { callback };
    protocol.request(writer, request);
    protocol.receive(reader, writer)
}
