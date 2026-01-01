//! The main loop of the proc-macro server.
use proc_macro_api::{
    Codec,
    bidirectional_protocol::msg as bidirectional,
    legacy_protocol::msg as legacy,
    transport::codec::{json::JsonProtocol, postcard::PostcardProtocol},
    version::CURRENT_API_VERSION,
};
use std::io;

use legacy::Message;

use proc_macro_srv::{EnvSnapshot, SpanId};

use crate::ProtocolFormat;
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

pub(crate) fn run(format: ProtocolFormat) -> io::Result<()> {
    match format {
        ProtocolFormat::JsonLegacy => run_::<JsonProtocol>(),
        ProtocolFormat::PostcardLegacy => run_::<PostcardProtocol>(),
        ProtocolFormat::BidirectionalPostcardPrototype => run_new::<PostcardProtocol>(),
    }
}

fn run_new<C: Codec>() -> io::Result<()> {
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
    let mut stdin = io::stdin();
    let mut stdout = io::stdout();

    let env_snapshot = EnvSnapshot::default();
    let srv = proc_macro_srv::ProcMacroSrv::new(&env_snapshot);

    let mut span_mode = legacy::SpanMode::Id;

    'outer: loop {
        let req_opt =
            bidirectional::BidirectionalMessage::read::<_, C>(&mut stdin.lock(), &mut buf)?;
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

                    send_response::<C>(&stdout, bidirectional::Response::ListMacros(res))?;
                }

                bidirectional::Request::ApiVersionCheck {} => {
                    // bidirectional::Response::ApiVersionCheck(CURRENT_API_VERSION).write::<_, C>(stdout)
                    send_response::<C>(
                        &stdout,
                        bidirectional::Response::ApiVersionCheck(CURRENT_API_VERSION),
                    )?;
                }

                bidirectional::Request::SetConfig(config) => {
                    span_mode = config.span_mode;
                    send_response::<C>(&stdout, bidirectional::Response::SetConfig(config))?;
                }
                bidirectional::Request::ExpandMacro(task) => {
                    handle_expand::<C>(&srv, &mut stdin, &mut stdout, &mut buf, span_mode, *task)?;
                }
            },
            _ => continue,
        }
    }

    Ok(())
}

fn handle_expand<C: Codec>(
    srv: &proc_macro_srv::ProcMacroSrv<'_>,
    stdin: &io::Stdin,
    stdout: &io::Stdout,
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
    stdout: &io::Stdout,
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

    send_response::<C>(&stdout, bidirectional::Response::ExpandMacro(res))
}

struct ProcMacroClientHandle<'a, C: Codec> {
    stdin: &'a io::Stdin,
    stdout: &'a io::Stdout,
    buf: &'a mut C::Buf,
}

impl<C: Codec> proc_macro_srv::ProcMacroClientInterface for ProcMacroClientHandle<'_, C> {
    fn source_text(&mut self, file_id: u32, start: u32, end: u32) -> Option<String> {
        let req = bidirectional::BidirectionalMessage::SubRequest(
            bidirectional::SubRequest::SourceText { file_id, start, end },
        );

        if req.write::<_, C>(&mut self.stdout.lock()).is_err() {
            return None;
        }

        let msg = match bidirectional::BidirectionalMessage::read::<_, C>(
            &mut self.stdin.lock(),
            self.buf,
        ) {
            Ok(Some(msg)) => msg,
            _ => return None,
        };

        match msg {
            bidirectional::BidirectionalMessage::SubResponse(
                bidirectional::SubResponse::SourceTextResult { text },
            ) => text,
            _ => None,
        }
    }
}

fn handle_expand_ra<C: Codec>(
    srv: &proc_macro_srv::ProcMacroSrv<'_>,
    stdin: &io::Stdin,
    stdout: &io::Stdout,
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

    send_response::<C>(&stdout, bidirectional::Response::ExpandMacroExtended(res))
}

fn run_<C: Codec>() -> io::Result<()> {
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
    let mut read_request = || legacy::Request::read::<_, C>(&mut io::stdin().lock(), &mut buf);
    let write_response = |msg: legacy::Response| msg.write::<_, C>(&mut io::stdout().lock());

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

fn send_response<C: Codec>(stdout: &io::Stdout, resp: bidirectional::Response) -> io::Result<()> {
    let resp = bidirectional::BidirectionalMessage::Response(resp);
    resp.write::<_, C>(&mut stdout.lock())
}
