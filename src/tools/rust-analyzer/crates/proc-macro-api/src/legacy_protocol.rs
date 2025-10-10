//! The initial proc-macro-srv protocol, soon to be deprecated.

pub mod json;
pub mod msg;

use std::{
    io::{BufRead, Write},
    sync::Arc,
};

use paths::AbsPath;
use span::Span;

use crate::{
    ProcMacro, ProcMacroKind, ServerError,
    legacy_protocol::{
        json::{read_json, write_json},
        msg::{
            ExpandMacro, ExpandMacroData, ExpnGlobals, FlatTree, Message, Request, Response,
            ServerConfig, SpanDataIndexMap, deserialize_span_data_index_map,
            flat::serialize_span_data_index_map,
        },
    },
    process::ProcMacroServerProcess,
    version,
};

pub(crate) use crate::legacy_protocol::msg::SpanMode;

/// Legacy span type, only defined here as it is still used by the proc-macro server.
/// While rust-analyzer doesn't use this anymore at all, RustRover relies on the legacy type for
/// proc-macro expansion.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpanId(pub u32);

impl std::fmt::Debug for SpanId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

pub(crate) fn version_check(srv: &ProcMacroServerProcess) -> Result<u32, ServerError> {
    let request = Request::ApiVersionCheck {};
    let response = send_task(srv, request)?;

    match response {
        Response::ApiVersionCheck(version) => Ok(version),
        _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
    }
}

/// Enable support for rust-analyzer span mode if the server supports it.
pub(crate) fn enable_rust_analyzer_spans(
    srv: &ProcMacroServerProcess,
) -> Result<SpanMode, ServerError> {
    let request = Request::SetConfig(ServerConfig { span_mode: SpanMode::RustAnalyzer });
    let response = send_task(srv, request)?;

    match response {
        Response::SetConfig(ServerConfig { span_mode }) => Ok(span_mode),
        _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
    }
}

/// Finds proc-macros in a given dynamic library.
pub(crate) fn find_proc_macros(
    srv: &ProcMacroServerProcess,
    dylib_path: &AbsPath,
) -> Result<Result<Vec<(String, ProcMacroKind)>, String>, ServerError> {
    let request = Request::ListMacros { dylib_path: dylib_path.to_path_buf().into() };

    let response = send_task(srv, request)?;

    match response {
        Response::ListMacros(it) => Ok(it),
        _ => Err(ServerError { message: "unexpected response".to_owned(), io: None }),
    }
}

pub(crate) fn expand(
    proc_macro: &ProcMacro,
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
    let task = ExpandMacro {
        data: ExpandMacroData {
            macro_body: FlatTree::new(subtree, version, &mut span_data_table),
            macro_name: proc_macro.name.to_string(),
            attributes: attr.map(|subtree| FlatTree::new(subtree, version, &mut span_data_table)),
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
    };

    let response = send_task(&proc_macro.process, Request::ExpandMacro(Box::new(task)))?;

    match response {
        Response::ExpandMacro(it) => Ok(it
            .map(|tree| {
                let mut expanded = FlatTree::to_subtree_resolved(tree, version, &span_data_table);
                if proc_macro.needs_fixup_change() {
                    proc_macro.change_fixup_to_match_old_server(&mut expanded);
                }
                expanded
            })
            .map_err(|msg| msg.0)),
        Response::ExpandMacroExtended(it) => Ok(it
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

/// Sends a request to the proc-macro server and waits for a response.
fn send_task(srv: &ProcMacroServerProcess, req: Request) -> Result<Response, ServerError> {
    if let Some(server_error) = srv.exited() {
        return Err(server_error.clone());
    }

    srv.send_task(send_request, req)
}

/// Sends a request to the server and reads the response.
fn send_request(
    mut writer: &mut dyn Write,
    mut reader: &mut dyn BufRead,
    req: Request,
    buf: &mut String,
) -> Result<Option<Response>, ServerError> {
    req.write(write_json, &mut writer).map_err(|err| ServerError {
        message: "failed to write request".into(),
        io: Some(Arc::new(err)),
    })?;
    let res = Response::read(read_json, &mut reader, buf).map_err(|err| ServerError {
        message: "failed to read response".into(),
        io: Some(Arc::new(err)),
    })?;
    Ok(res)
}
