//! The main loop of the proc-macro server.
use proc_macro_api::{
    Codec, ProtocolFormat,
    bidirectional_protocol::msg as bidirectional,
    legacy_protocol::msg as legacy,
    transport::codec::{json::JsonProtocol, postcard::PostcardProtocol},
    version::CURRENT_API_VERSION,
};
use std::{
    io::{self, BufRead, Write},
    ops::Range,
};

use legacy::Message;

use proc_macro_srv::{EnvSnapshot, SpanId};

struct SpanTrans;

impl legacy::SpanTransformer for SpanTrans {
    type Table = ();
    type Span = SpanId;
    fn token_id_of(
        _: &mut Self::Table,
        span: Self::Span,
    ) -> proc_macro_api::legacy_protocol::SpanId {
        proc_macro_api::legacy_protocol::SpanId(span.0)
    }
    fn span_for_token_id(
        _: &Self::Table,
        id: proc_macro_api::legacy_protocol::SpanId,
    ) -> Self::Span {
        SpanId(id.0)
    }
}

pub fn run(
    stdin: &mut (dyn BufRead + Send + Sync),
    stdout: &mut (dyn Write + Send + Sync),
    format: ProtocolFormat,
) -> io::Result<()> {
    match format {
        ProtocolFormat::JsonLegacy => run_old::<JsonProtocol>(stdin, stdout),
        ProtocolFormat::BidirectionalPostcardPrototype => {
            run_new::<PostcardProtocol>(stdin, stdout)
        }
    }
}

fn run_new<C: Codec>(
    stdin: &mut (dyn BufRead + Send + Sync),
    stdout: &mut (dyn Write + Send + Sync),
) -> io::Result<()> {
    fn macro_kind_to_api(kind: proc_macro_srv::ProcMacroKind) -> proc_macro_api::ProcMacroKind {
        match kind {
            proc_macro_srv::ProcMacroKind::CustomDerive => {
                proc_macro_api::ProcMacroKind::CustomDerive
            }
            proc_macro_srv::ProcMacroKind::Bang => proc_macro_api::ProcMacroKind::Bang,
            proc_macro_srv::ProcMacroKind::Attr => proc_macro_api::ProcMacroKind::Attr,
        }
    }

    let mut buf = C::Buf::default();

    let env_snapshot = EnvSnapshot::default();
    let srv = proc_macro_srv::ProcMacroSrv::new(&env_snapshot);

    let mut span_mode = legacy::SpanMode::Id;

    'outer: loop {
        let req_opt = bidirectional::BidirectionalMessage::read::<C>(stdin, &mut buf)?;
        let Some(req) = req_opt else {
            break 'outer;
        };

        match req {
            bidirectional::BidirectionalMessage::Request(request) => match request {
                bidirectional::Request::ListMacros { dylib_path } => {
                    let res = srv.list_macros(&dylib_path).map(|macros| {
                        macros
                            .into_iter()
                            .map(|(name, kind)| (name, macro_kind_to_api(kind)))
                            .collect()
                    });

                    send_response::<C>(stdout, bidirectional::Response::ListMacros(res))?;
                }

                bidirectional::Request::ApiVersionCheck {} => {
                    send_response::<C>(
                        stdout,
                        bidirectional::Response::ApiVersionCheck(CURRENT_API_VERSION),
                    )?;
                }

                bidirectional::Request::SetConfig(config) => {
                    span_mode = config.span_mode;
                    send_response::<C>(stdout, bidirectional::Response::SetConfig(config))?;
                }
                bidirectional::Request::ExpandMacro(task) => {
                    handle_expand::<C>(&srv, stdin, stdout, &mut buf, span_mode, *task)?;
                }
            },
            _ => continue,
        }
    }

    Ok(())
}

fn handle_expand<C: Codec>(
    srv: &proc_macro_srv::ProcMacroSrv<'_>,
    stdin: &mut (dyn BufRead + Send + Sync),
    stdout: &mut (dyn Write + Send + Sync),
    buf: &mut C::Buf,
    span_mode: legacy::SpanMode,
    task: bidirectional::ExpandMacro,
) -> io::Result<()> {
    match span_mode {
        legacy::SpanMode::Id => handle_expand_id::<C>(srv, stdout, task),
        legacy::SpanMode::RustAnalyzer => handle_expand_ra::<C>(srv, stdin, stdout, buf, task),
    }
}

fn handle_expand_id<C: Codec>(
    srv: &proc_macro_srv::ProcMacroSrv<'_>,
    stdout: &mut dyn Write,
    task: bidirectional::ExpandMacro,
) -> io::Result<()> {
    let bidirectional::ExpandMacro { lib, env, current_dir, data } = task;
    let bidirectional::ExpandMacroData {
        macro_body,
        macro_name,
        attributes,
        has_global_spans: bidirectional::ExpnGlobals { def_site, call_site, mixed_site, .. },
        ..
    } = data;

    let def_site = SpanId(def_site as u32);
    let call_site = SpanId(call_site as u32);
    let mixed_site = SpanId(mixed_site as u32);

    let macro_body =
        macro_body.to_tokenstream_unresolved::<SpanTrans>(CURRENT_API_VERSION, |_, b| b);
    let attributes = attributes
        .map(|it| it.to_tokenstream_unresolved::<SpanTrans>(CURRENT_API_VERSION, |_, b| b));

    let res = srv
        .expand(
            lib,
            &env,
            current_dir,
            &macro_name,
            macro_body,
            attributes,
            def_site,
            call_site,
            mixed_site,
            None,
        )
        .map(|it| {
            legacy::FlatTree::from_tokenstream_raw::<SpanTrans>(it, call_site, CURRENT_API_VERSION)
        })
        .map_err(|e| legacy::PanicMessage(e.into_string().unwrap_or_default()));

    send_response::<C>(stdout, bidirectional::Response::ExpandMacro(res))
}

struct ProcMacroClientHandle<'a, C: Codec> {
    stdin: &'a mut (dyn BufRead + Send + Sync),
    stdout: &'a mut (dyn Write + Send + Sync),
    buf: &'a mut C::Buf,
}

impl<'a, C: Codec> ProcMacroClientHandle<'a, C> {
    fn roundtrip(
        &mut self,
        req: bidirectional::SubRequest,
    ) -> Option<bidirectional::BidirectionalMessage> {
        let msg = bidirectional::BidirectionalMessage::SubRequest(req);

        if msg.write::<C>(&mut *self.stdout).is_err() {
            return None;
        }

        match bidirectional::BidirectionalMessage::read::<C>(&mut *self.stdin, self.buf) {
            Ok(Some(msg)) => Some(msg),
            _ => None,
        }
    }
}

impl<C: Codec> proc_macro_srv::ProcMacroClientInterface for ProcMacroClientHandle<'_, C> {
    fn file(&mut self, file_id: proc_macro_srv::span::FileId) -> String {
        match self.roundtrip(bidirectional::SubRequest::FilePath { file_id: file_id.index() }) {
            Some(bidirectional::BidirectionalMessage::SubResponse(
                bidirectional::SubResponse::FilePathResult { name },
            )) => name,
            _ => String::new(),
        }
    }

    fn source_text(
        &mut self,
        proc_macro_srv::span::Span { range, anchor, ctx: _ }: proc_macro_srv::span::Span,
    ) -> Option<String> {
        match self.roundtrip(bidirectional::SubRequest::SourceText {
            file_id: anchor.file_id.as_u32(),
            ast_id: anchor.ast_id.into_raw(),
            start: range.start().into(),
            end: range.end().into(),
        }) {
            Some(bidirectional::BidirectionalMessage::SubResponse(
                bidirectional::SubResponse::SourceTextResult { text },
            )) => text,
            _ => None,
        }
    }

    fn local_file(&mut self, file_id: proc_macro_srv::span::FileId) -> Option<String> {
        match self.roundtrip(bidirectional::SubRequest::LocalFilePath { file_id: file_id.index() })
        {
            Some(bidirectional::BidirectionalMessage::SubResponse(
                bidirectional::SubResponse::LocalFilePathResult { name },
            )) => name,
            _ => None,
        }
    }

    fn line_column(&mut self, span: proc_macro_srv::span::Span) -> Option<(u32, u32)> {
        let proc_macro_srv::span::Span { range, anchor, ctx: _ } = span;
        match self.roundtrip(bidirectional::SubRequest::LineColumn {
            file_id: anchor.file_id.as_u32(),
            ast_id: anchor.ast_id.into_raw(),
            offset: range.start().into(),
        }) {
            Some(bidirectional::BidirectionalMessage::SubResponse(
                bidirectional::SubResponse::LineColumnResult { line, column },
            )) => Some((line, column)),
            _ => None,
        }
    }

    fn byte_range(
        &mut self,
        proc_macro_srv::span::Span { range, anchor, ctx: _ }: proc_macro_srv::span::Span,
    ) -> Range<usize> {
        match self.roundtrip(bidirectional::SubRequest::ByteRange {
            file_id: anchor.file_id.as_u32(),
            ast_id: anchor.ast_id.into_raw(),
            start: range.start().into(),
            end: range.end().into(),
        }) {
            Some(bidirectional::BidirectionalMessage::SubResponse(
                bidirectional::SubResponse::ByteRangeResult { range },
            )) => range,
            _ => Range { start: range.start().into(), end: range.end().into() },
        }
    }
}

fn handle_expand_ra<C: Codec>(
    srv: &proc_macro_srv::ProcMacroSrv<'_>,
    stdin: &mut (dyn BufRead + Send + Sync),
    stdout: &mut (dyn Write + Send + Sync),
    buf: &mut C::Buf,
    task: bidirectional::ExpandMacro,
) -> io::Result<()> {
    let bidirectional::ExpandMacro {
        lib,
        env,
        current_dir,
        data:
            bidirectional::ExpandMacroData {
                macro_body,
                macro_name,
                attributes,
                has_global_spans: bidirectional::ExpnGlobals { def_site, call_site, mixed_site, .. },
                span_data_table,
            },
    } = task;

    let mut span_data_table = legacy::deserialize_span_data_index_map(&span_data_table);

    let def_site = span_data_table[def_site];
    let call_site = span_data_table[call_site];
    let mixed_site = span_data_table[mixed_site];

    let macro_body =
        macro_body.to_tokenstream_resolved(CURRENT_API_VERSION, &span_data_table, |a, b| {
            srv.join_spans(a, b).unwrap_or(b)
        });

    let attributes = attributes.map(|it| {
        it.to_tokenstream_resolved(CURRENT_API_VERSION, &span_data_table, |a, b| {
            srv.join_spans(a, b).unwrap_or(b)
        })
    });

    let res = srv
        .expand(
            lib,
            &env,
            current_dir,
            &macro_name,
            macro_body,
            attributes,
            def_site,
            call_site,
            mixed_site,
            Some(&mut ProcMacroClientHandle::<C> { stdin, stdout, buf }),
        )
        .map(|it| {
            (
                legacy::FlatTree::from_tokenstream(
                    it,
                    CURRENT_API_VERSION,
                    call_site,
                    &mut span_data_table,
                ),
                legacy::serialize_span_data_index_map(&span_data_table),
            )
        })
        .map(|(tree, span_data_table)| bidirectional::ExpandMacroExtended { tree, span_data_table })
        .map_err(|e| legacy::PanicMessage(e.into_string().unwrap_or_default()));

    send_response::<C>(stdout, bidirectional::Response::ExpandMacroExtended(res))
}

fn run_old<C: Codec>(
    stdin: &mut (dyn BufRead + Send + Sync),
    stdout: &mut (dyn Write + Send + Sync),
) -> io::Result<()> {
    fn macro_kind_to_api(kind: proc_macro_srv::ProcMacroKind) -> proc_macro_api::ProcMacroKind {
        match kind {
            proc_macro_srv::ProcMacroKind::CustomDerive => {
                proc_macro_api::ProcMacroKind::CustomDerive
            }
            proc_macro_srv::ProcMacroKind::Bang => proc_macro_api::ProcMacroKind::Bang,
            proc_macro_srv::ProcMacroKind::Attr => proc_macro_api::ProcMacroKind::Attr,
        }
    }

    let mut buf = C::Buf::default();
    let mut read_request = || legacy::Request::read::<C>(stdin, &mut buf);
    let mut write_response = |msg: legacy::Response| msg.write::<C>(stdout);

    let env = EnvSnapshot::default();
    let srv = proc_macro_srv::ProcMacroSrv::new(&env);

    let mut span_mode = legacy::SpanMode::Id;

    while let Some(req) = read_request()? {
        let res = match req {
            legacy::Request::ListMacros { dylib_path } => {
                legacy::Response::ListMacros(srv.list_macros(&dylib_path).map(|macros| {
                    macros.into_iter().map(|(name, kind)| (name, macro_kind_to_api(kind))).collect()
                }))
            }
            legacy::Request::ExpandMacro(task) => {
                let legacy::ExpandMacro {
                    lib,
                    env,
                    current_dir,
                    data:
                        legacy::ExpandMacroData {
                            macro_body,
                            macro_name,
                            attributes,
                            has_global_spans:
                                legacy::ExpnGlobals { serialize: _, def_site, call_site, mixed_site },
                            span_data_table,
                        },
                } = *task;
                match span_mode {
                    legacy::SpanMode::Id => legacy::Response::ExpandMacro({
                        let def_site = SpanId(def_site as u32);
                        let call_site = SpanId(call_site as u32);
                        let mixed_site = SpanId(mixed_site as u32);

                        let macro_body = macro_body
                            .to_tokenstream_unresolved::<SpanTrans>(CURRENT_API_VERSION, |_, b| b);
                        let attributes = attributes.map(|it| {
                            it.to_tokenstream_unresolved::<SpanTrans>(CURRENT_API_VERSION, |_, b| b)
                        });

                        srv.expand(
                            lib,
                            &env,
                            current_dir,
                            &macro_name,
                            macro_body,
                            attributes,
                            def_site,
                            call_site,
                            mixed_site,
                            None,
                        )
                        .map(|it| {
                            legacy::FlatTree::from_tokenstream_raw::<SpanTrans>(
                                it,
                                call_site,
                                CURRENT_API_VERSION,
                            )
                        })
                        .map_err(|e| e.into_string().unwrap_or_default())
                        .map_err(legacy::PanicMessage)
                    }),
                    legacy::SpanMode::RustAnalyzer => legacy::Response::ExpandMacroExtended({
                        let mut span_data_table =
                            legacy::deserialize_span_data_index_map(&span_data_table);

                        let def_site = span_data_table[def_site];
                        let call_site = span_data_table[call_site];
                        let mixed_site = span_data_table[mixed_site];

                        let macro_body = macro_body.to_tokenstream_resolved(
                            CURRENT_API_VERSION,
                            &span_data_table,
                            |a, b| srv.join_spans(a, b).unwrap_or(b),
                        );
                        let attributes = attributes.map(|it| {
                            it.to_tokenstream_resolved(
                                CURRENT_API_VERSION,
                                &span_data_table,
                                |a, b| srv.join_spans(a, b).unwrap_or(b),
                            )
                        });
                        srv.expand(
                            lib,
                            &env,
                            current_dir,
                            &macro_name,
                            macro_body,
                            attributes,
                            def_site,
                            call_site,
                            mixed_site,
                            None,
                        )
                        .map(|it| {
                            (
                                legacy::FlatTree::from_tokenstream(
                                    it,
                                    CURRENT_API_VERSION,
                                    call_site,
                                    &mut span_data_table,
                                ),
                                legacy::serialize_span_data_index_map(&span_data_table),
                            )
                        })
                        .map(|(tree, span_data_table)| legacy::ExpandMacroExtended {
                            tree,
                            span_data_table,
                        })
                        .map_err(|e| e.into_string().unwrap_or_default())
                        .map_err(legacy::PanicMessage)
                    }),
                }
            }
            legacy::Request::ApiVersionCheck {} => {
                legacy::Response::ApiVersionCheck(CURRENT_API_VERSION)
            }
            legacy::Request::SetConfig(config) => {
                span_mode = config.span_mode;
                legacy::Response::SetConfig(config)
            }
        };
        write_response(res)?
    }

    Ok(())
}

fn send_response<C: Codec>(
    stdout: &mut dyn Write,
    resp: bidirectional::Response,
) -> io::Result<()> {
    let resp = bidirectional::BidirectionalMessage::Response(resp);
    resp.write::<C>(stdout)
}
