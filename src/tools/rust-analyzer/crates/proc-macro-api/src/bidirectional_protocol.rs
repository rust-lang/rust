//! Bidirectional protocol methods

use std::{
    io::{self, BufRead, Write},
    sync::Arc,
};

use base_db::SourceDatabase;
use paths::AbsPath;
use span::{FileId, Span};

use crate::{
    Codec, ProcMacro, ProcMacroKind, ServerError,
    bidirectional_protocol::msg::{
        Envelope, ExpandMacro, ExpandMacroData, ExpnGlobals, Kind, Payload, Request, RequestId,
        Response, SubRequest, SubResponse,
    },
    legacy_protocol::{
        SpanMode,
        msg::{
            FlatTree, ServerConfig, SpanDataIndexMap, deserialize_span_data_index_map,
            serialize_span_data_index_map,
        },
    },
    process::ProcMacroServerProcess,
    transport::codec::{json::JsonProtocol, postcard::PostcardProtocol},
    version,
};

pub mod msg;

pub trait ClientCallbacks {
    fn handle_sub_request(&mut self, id: u64, req: SubRequest) -> Result<SubResponse, ServerError>;
}

pub fn run_conversation<C: Codec>(
    writer: &mut dyn Write,
    reader: &mut dyn BufRead,
    buf: &mut C::Buf,
    id: RequestId,
    initial: Payload,
    callbacks: &mut dyn ClientCallbacks,
) -> Result<Payload, ServerError> {
    let msg = Envelope { id, kind: Kind::Request, payload: initial };
    let encoded = C::encode(&msg).map_err(wrap_encode)?;
    C::write(writer, &encoded).map_err(wrap_io("failed to write initial request"))?;

    loop {
        let maybe_buf = C::read(reader, buf).map_err(wrap_io("failed to read message"))?;
        let Some(b) = maybe_buf else {
            return Err(ServerError {
                message: "proc-macro server closed the stream".into(),
                io: Some(Arc::new(io::Error::new(io::ErrorKind::UnexpectedEof, "closed"))),
            });
        };

        let msg: Envelope = C::decode(b).map_err(wrap_decode)?;

        if msg.id != id {
            return Err(ServerError {
                message: format!("unexpected message id {}, expected {}", msg.id, id),
                io: None,
            });
        }

        match (msg.kind, msg.payload) {
            (Kind::SubRequest, Payload::SubRequest(sr)) => {
                let resp = callbacks.handle_sub_request(id, sr)?;
                let reply =
                    Envelope { id, kind: Kind::SubResponse, payload: Payload::SubResponse(resp) };
                let encoded = C::encode(&reply).map_err(wrap_encode)?;
                C::write(writer, &encoded).map_err(wrap_io("failed to write sub-response"))?;
            }
            (Kind::Response, payload) => {
                return Ok(payload);
            }
            (kind, payload) => {
                return Err(ServerError {
                    message: format!(
                        "unexpected message kind {:?} with payload {:?}",
                        kind, payload
                    ),
                    io: None,
                });
            }
        }
    }
}

fn wrap_io(msg: &'static str) -> impl Fn(io::Error) -> ServerError {
    move |err| ServerError { message: msg.into(), io: Some(Arc::new(err)) }
}

fn wrap_encode(err: io::Error) -> ServerError {
    ServerError { message: "failed to encode message".into(), io: Some(Arc::new(err)) }
}

fn wrap_decode(err: io::Error) -> ServerError {
    ServerError { message: "failed to decode message".into(), io: Some(Arc::new(err)) }
}

pub(crate) fn version_check(srv: &ProcMacroServerProcess) -> Result<u32, ServerError> {
    let request = Payload::Request(Request::ApiVersionCheck {});

    struct NoCallbacks;
    impl ClientCallbacks for NoCallbacks {
        fn handle_sub_request(
            &mut self,
            _id: u64,
            _req: SubRequest,
        ) -> Result<SubResponse, ServerError> {
            Err(ServerError { message: "sub-request not supported here".into(), io: None })
        }
    }

    let mut callbacks = NoCallbacks;

    let response_payload =
        run_bidirectional(srv, (0, Kind::Request, request).into(), &mut callbacks)?;

    match response_payload {
        Payload::Response(Response::ApiVersionCheck(version)) => Ok(version),
        other => {
            Err(ServerError { message: format!("unexpected response: {:?}", other), io: None })
        }
    }
}

/// Enable support for rust-analyzer span mode if the server supports it.
pub(crate) fn enable_rust_analyzer_spans(
    srv: &ProcMacroServerProcess,
) -> Result<SpanMode, ServerError> {
    let request =
        Payload::Request(Request::SetConfig(ServerConfig { span_mode: SpanMode::RustAnalyzer }));

    struct NoCallbacks;
    impl ClientCallbacks for NoCallbacks {
        fn handle_sub_request(
            &mut self,
            _id: u64,
            _req: SubRequest,
        ) -> Result<SubResponse, ServerError> {
            Err(ServerError { message: "sub-request not supported here".into(), io: None })
        }
    }

    let mut callbacks = NoCallbacks;

    let response_payload =
        run_bidirectional(srv, (0, Kind::Request, request).into(), &mut callbacks)?;

    match response_payload {
        Payload::Response(Response::SetConfig(ServerConfig { span_mode })) => Ok(span_mode),
        _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
    }
}

/// Finds proc-macros in a given dynamic library.
pub(crate) fn find_proc_macros(
    srv: &ProcMacroServerProcess,
    dylib_path: &AbsPath,
) -> Result<Result<Vec<(String, ProcMacroKind)>, String>, ServerError> {
    let request =
        Payload::Request(Request::ListMacros { dylib_path: dylib_path.to_path_buf().into() });

    struct NoCallbacks;
    impl ClientCallbacks for NoCallbacks {
        fn handle_sub_request(
            &mut self,
            _id: u64,
            _req: SubRequest,
        ) -> Result<SubResponse, ServerError> {
            Err(ServerError { message: "sub-request not supported here".into(), io: None })
        }
    }

    let mut callbacks = NoCallbacks;

    let response_payload =
        run_bidirectional(srv, (0, Kind::Request, request).into(), &mut callbacks)?;

    match response_payload {
        Payload::Response(Response::ListMacros(it)) => Ok(it),
        _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
    }
}

pub(crate) fn expand(
    proc_macro: &ProcMacro,
    db: &dyn SourceDatabase,
    subtree: tt::SubtreeView<'_, Span>,
    attr: Option<tt::SubtreeView<'_, Span>>,
    env: Vec<(String, String)>,
    def_site: Span,
    call_site: Span,
    mixed_site: Span,
    current_dir: String,
) -> Result<Result<tt::TopSubtree<span::SpanData<span::SyntaxContext>>, String>, crate::ServerError>
{
    let version = proc_macro.process.version();
    let mut span_data_table = SpanDataIndexMap::default();
    let def_site = span_data_table.insert_full(def_site).0;
    let call_site = span_data_table.insert_full(call_site).0;
    let mixed_site = span_data_table.insert_full(mixed_site).0;
    let task = Payload::Request(Request::ExpandMacro(Box::new(ExpandMacro {
        data: ExpandMacroData {
            macro_body: FlatTree::from_subtree(subtree, version, &mut span_data_table),
            macro_name: proc_macro.name.to_string(),
            attributes: attr
                .map(|subtree| FlatTree::from_subtree(subtree, version, &mut span_data_table)),
            has_global_spans: ExpnGlobals {
                serialize: version >= version::HAS_GLOBAL_SPANS,
                def_site,
                call_site,
                mixed_site,
            },
            span_data_table: if proc_macro.process.rust_analyzer_spans() {
                serialize_span_data_index_map(&span_data_table)
            } else {
                Vec::new()
            },
        },
        lib: proc_macro.dylib_path.to_path_buf().into(),
        env,
        current_dir: Some(current_dir),
    })));

    struct Callbacks<'de> {
        db: &'de dyn SourceDatabase,
    }
    impl<'db> ClientCallbacks for Callbacks<'db> {
        fn handle_sub_request(
            &mut self,
            _id: u64,
            req: SubRequest,
        ) -> Result<SubResponse, ServerError> {
            match req {
                SubRequest::SourceText { file_id, start, end } => {
                    let file = FileId::from_raw(file_id);
                    let text = self.db.file_text(file).text(self.db);

                    let slice = text.get(start as usize..end as usize).map(|s| s.to_owned());

                    Ok(SubResponse::SourceTextResult { text: slice })
                }
            }
        }
    }

    let mut callbacks = Callbacks { db };

    let response_payload =
        run_bidirectional(&proc_macro.process, (0, Kind::Request, task).into(), &mut callbacks)?;

    match response_payload {
        Payload::Response(Response::ExpandMacro(it)) => Ok(it
            .map(|tree| {
                let mut expanded = FlatTree::to_subtree_resolved(tree, version, &span_data_table);
                if proc_macro.needs_fixup_change() {
                    proc_macro.change_fixup_to_match_old_server(&mut expanded);
                }
                expanded
            })
            .map_err(|msg| msg.0)),
        Payload::Response(Response::ExpandMacroExtended(it)) => Ok(it
            .map(|resp| {
                let mut expanded = FlatTree::to_subtree_resolved(
                    resp.tree,
                    version,
                    &deserialize_span_data_index_map(&resp.span_data_table),
                );
                if proc_macro.needs_fixup_change() {
                    proc_macro.change_fixup_to_match_old_server(&mut expanded);
                }
                expanded
            })
            .map_err(|msg| msg.0)),
        _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
    }
}

fn run_bidirectional(
    srv: &ProcMacroServerProcess,
    msg: Envelope,
    callbacks: &mut dyn ClientCallbacks,
) -> Result<Payload, ServerError> {
    if let Some(server_error) = srv.exited() {
        return Err(server_error.clone());
    }

    if srv.use_postcard() {
        srv.run_bidirectional::<PostcardProtocol>(msg.id, msg.payload, callbacks)
    } else {
        srv.run_bidirectional::<JsonProtocol>(msg.id, msg.payload, callbacks)
    }
}
