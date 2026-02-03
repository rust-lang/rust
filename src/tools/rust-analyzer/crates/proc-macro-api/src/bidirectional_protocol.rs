//! Bidirectional protocol methods

use std::{
    io::{self, BufRead, Write},
    panic::{AssertUnwindSafe, catch_unwind},
    sync::Arc,
};

use paths::AbsPath;
use span::Span;

use crate::{
    ProcMacro, ProcMacroKind, ServerError,
    bidirectional_protocol::msg::{
        BidirectionalMessage, ExpandMacro, ExpandMacroData, ExpnGlobals, Request, Response,
        SubRequest, SubResponse,
    },
    legacy_protocol::{
        SpanMode,
        msg::{
            FlatTree, ServerConfig, SpanDataIndexMap, deserialize_span_data_index_map,
            serialize_span_data_index_map,
        },
    },
    process::ProcMacroServerProcess,
    transport::postcard,
};

pub mod msg;

pub type SubCallback<'a> = &'a dyn Fn(SubRequest) -> Result<SubResponse, ServerError>;

pub fn run_conversation(
    writer: &mut dyn Write,
    reader: &mut dyn BufRead,
    buf: &mut Vec<u8>,
    msg: BidirectionalMessage,
    callback: SubCallback<'_>,
) -> Result<BidirectionalMessage, ServerError> {
    let encoded = postcard::encode(&msg).map_err(wrap_encode)?;
    postcard::write(writer, &encoded).map_err(wrap_io("failed to write initial request"))?;

    loop {
        let maybe_buf = postcard::read(reader, buf).map_err(wrap_io("failed to read message"))?;
        let Some(b) = maybe_buf else {
            return Err(ServerError {
                message: "proc-macro server closed the stream".into(),
                io: Some(Arc::new(io::Error::new(io::ErrorKind::UnexpectedEof, "closed"))),
            });
        };

        let msg: BidirectionalMessage = postcard::decode(b).map_err(wrap_decode)?;

        match msg {
            BidirectionalMessage::Response(response) => {
                return Ok(BidirectionalMessage::Response(response));
            }
            BidirectionalMessage::SubRequest(sr) => {
                // TODO: Avoid `AssertUnwindSafe` by making the callback `UnwindSafe` once `ExpandDatabase`
                // becomes unwind-safe (currently blocked by `parking_lot::RwLock` in the VFS).
                let resp = match catch_unwind(AssertUnwindSafe(|| callback(sr))) {
                    Ok(Ok(resp)) => BidirectionalMessage::SubResponse(resp),
                    Ok(Err(err)) => BidirectionalMessage::SubResponse(SubResponse::Cancel {
                        reason: err.to_string(),
                    }),
                    Err(_) => BidirectionalMessage::SubResponse(SubResponse::Cancel {
                        reason: "callback panicked or was cancelled".into(),
                    }),
                };

                let encoded = postcard::encode(&resp).map_err(wrap_encode)?;
                postcard::write(writer, &encoded)
                    .map_err(wrap_io("failed to write sub-response"))?;
            }
            _ => {
                return Err(ServerError {
                    message: format!("unexpected message {:?}", msg),
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

pub(crate) fn version_check(
    srv: &ProcMacroServerProcess,
    callback: SubCallback<'_>,
) -> Result<u32, ServerError> {
    let request = BidirectionalMessage::Request(Request::ApiVersionCheck {});

    let response_payload = run_request(srv, request, callback)?;

    match response_payload {
        BidirectionalMessage::Response(Response::ApiVersionCheck(version)) => Ok(version),
        other => {
            Err(ServerError { message: format!("unexpected response: {:?}", other), io: None })
        }
    }
}

/// Enable support for rust-analyzer span mode if the server supports it.
pub(crate) fn enable_rust_analyzer_spans(
    srv: &ProcMacroServerProcess,
    callback: SubCallback<'_>,
) -> Result<SpanMode, ServerError> {
    let request = BidirectionalMessage::Request(Request::SetConfig(ServerConfig {
        span_mode: SpanMode::RustAnalyzer,
    }));

    let response_payload = run_request(srv, request, callback)?;

    match response_payload {
        BidirectionalMessage::Response(Response::SetConfig(ServerConfig { span_mode })) => {
            Ok(span_mode)
        }
        _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
    }
}

/// Finds proc-macros in a given dynamic library.
pub(crate) fn find_proc_macros(
    srv: &ProcMacroServerProcess,
    dylib_path: &AbsPath,
    callback: SubCallback<'_>,
) -> Result<Result<Vec<(String, ProcMacroKind)>, String>, ServerError> {
    let request = BidirectionalMessage::Request(Request::ListMacros {
        dylib_path: dylib_path.to_path_buf().into(),
    });

    let response_payload = run_request(srv, request, callback)?;

    match response_payload {
        BidirectionalMessage::Response(Response::ListMacros(it)) => Ok(it),
        _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
    }
}

pub(crate) fn expand(
    proc_macro: &ProcMacro,
    process: &ProcMacroServerProcess,
    subtree: tt::SubtreeView<'_>,
    attr: Option<tt::SubtreeView<'_>>,
    env: Vec<(String, String)>,
    def_site: Span,
    call_site: Span,
    mixed_site: Span,
    current_dir: String,
    callback: SubCallback<'_>,
) -> Result<Result<tt::TopSubtree, String>, crate::ServerError> {
    let version = process.version();
    let mut span_data_table = SpanDataIndexMap::default();
    let def_site = span_data_table.insert_full(def_site).0;
    let call_site = span_data_table.insert_full(call_site).0;
    let mixed_site = span_data_table.insert_full(mixed_site).0;
    let task = BidirectionalMessage::Request(Request::ExpandMacro(Box::new(ExpandMacro {
        data: ExpandMacroData {
            macro_body: FlatTree::from_subtree(subtree, version, &mut span_data_table),
            macro_name: proc_macro.name.to_string(),
            attributes: attr
                .map(|subtree| FlatTree::from_subtree(subtree, version, &mut span_data_table)),
            has_global_spans: ExpnGlobals { def_site, call_site, mixed_site },
            span_data_table: if process.rust_analyzer_spans() {
                serialize_span_data_index_map(&span_data_table)
            } else {
                Vec::new()
            },
        },
        lib: proc_macro.dylib_path.to_path_buf().into(),
        env,
        current_dir: Some(current_dir),
    })));

    let response_payload = run_request(process, task, callback)?;

    match response_payload {
        BidirectionalMessage::Response(Response::ExpandMacro(it)) => Ok(it
            .map(|tree| {
                let mut expanded = FlatTree::to_subtree_resolved(tree, version, &span_data_table);
                if proc_macro.needs_fixup_change() {
                    proc_macro.change_fixup_to_match_old_server(&mut expanded);
                }
                expanded
            })
            .map_err(|msg| msg.0)),
        BidirectionalMessage::Response(Response::ExpandMacroExtended(it)) => Ok(it
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

fn run_request(
    srv: &ProcMacroServerProcess,
    msg: BidirectionalMessage,
    callback: SubCallback<'_>,
) -> Result<BidirectionalMessage, ServerError> {
    if let Some(err) = srv.exited() {
        return Err(err.clone());
    }
    srv.run_bidirectional(msg, callback)
}

pub fn reject_subrequests(req: SubRequest) -> Result<SubResponse, ServerError> {
    Err(ServerError { message: format!("{req:?} sub-request not supported here"), io: None })
}
