//! The main loop of the proc-macro server.
use std::io;

use crossbeam_channel::unbounded;
use proc_macro_api::bidirectional_protocol::msg::Request;
use proc_macro_api::{
    Codec,
    bidirectional_protocol::msg::{Envelope, Kind, Payload},
    legacy_protocol::msg::{
        self, ExpandMacroData, ExpnGlobals, Message, SpanMode, SpanTransformer,
        deserialize_span_data_index_map, serialize_span_data_index_map,
    },
    transport::codec::{json::JsonProtocol, postcard::PostcardProtocol},
    version::CURRENT_API_VERSION,
};
use proc_macro_srv::{EnvSnapshot, SpanId};

use crate::ProtocolFormat;
struct SpanTrans;

impl SpanTransformer for SpanTrans {
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
        ProtocolFormat::JsonNew => run_new::<JsonProtocol>(),
        ProtocolFormat::PostcardNew => run_new::<PostcardProtocol>(),
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
    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();

    let env_snapshot = EnvSnapshot::default();
    let srv = proc_macro_srv::ProcMacroSrv::new(&env_snapshot);

    let mut span_mode = SpanMode::Id;

    'outer: loop {
        let req_opt = Envelope::read::<_, C>(&mut stdin, &mut buf)?;
        let Some(req) = req_opt else {
            break 'outer;
        };

        match (req.kind, req.payload) {
            (Kind::Request, Payload::Request(request)) => match request {
                Request::ListMacros { dylib_path } => {
                    let res = srv.list_macros(&dylib_path).map(|macros| {
                        macros
                            .into_iter()
                            .map(|(name, kind)| (name, macro_kind_to_api(kind)))
                            .collect()
                    });

                    let resp_env = Envelope {
                        id: req.id,
                        kind: Kind::Response,
                        payload: Payload::Response(
                            proc_macro_api::bidirectional_protocol::msg::Response::ListMacros(res),
                        ),
                    };

                    resp_env.write::<_, C>(&mut stdout)?;
                }

                Request::ApiVersionCheck {} => {
                    let resp_env = Envelope {
                        id: req.id,
                        kind: Kind::Response,
                        payload: Payload::Response(
                            proc_macro_api::bidirectional_protocol::msg::Response::ApiVersionCheck(
                                CURRENT_API_VERSION,
                            ),
                        ),
                    };
                    resp_env.write::<_, C>(&mut stdout)?;
                }

                Request::SetConfig(config) => {
                    span_mode = config.span_mode;
                    let resp_env = Envelope {
                        id: req.id,
                        kind: Kind::Response,
                        payload: Payload::Response(
                            proc_macro_api::bidirectional_protocol::msg::Response::SetConfig(
                                config,
                            ),
                        ),
                    };
                    resp_env.write::<_, C>(&mut stdout)?;
                }

                Request::ExpandMacro(task) => {
                    let proc_macro_api::bidirectional_protocol::msg::ExpandMacro {
                        lib,
                        env,
                        current_dir,
                        data:
                            proc_macro_api::bidirectional_protocol::msg::ExpandMacroData {
                                macro_body,
                                macro_name,
                                attributes,
                                has_global_spans:
                                    proc_macro_api::bidirectional_protocol::msg::ExpnGlobals {
                                        serialize: _,
                                        def_site,
                                        call_site,
                                        mixed_site,
                                    },
                                span_data_table,
                            },
                    } = *task;

                    match span_mode {
                        SpanMode::Id => {
                            let def_site = SpanId(def_site as u32);
                            let call_site = SpanId(call_site as u32);
                            let mixed_site = SpanId(mixed_site as u32);

                            let macro_body = macro_body.to_tokenstream_unresolved::<SpanTrans>(
                                CURRENT_API_VERSION,
                                |_, b| b,
                            );
                            let attributes = attributes.map(|it| {
                                it.to_tokenstream_unresolved::<SpanTrans>(
                                    CURRENT_API_VERSION,
                                    |_, b| b,
                                )
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
                                )
                                .map(|it| {
                                    msg::FlatTree::from_tokenstream_raw::<SpanTrans>(
                                        it,
                                        call_site,
                                        CURRENT_API_VERSION,
                                    )
                                })
                                .map_err(|e| e.into_string().unwrap_or_default())
                                .map_err(msg::PanicMessage);

                            let resp_env = Envelope {
                                id: req.id,
                                kind: Kind::Response,
                                payload: Payload::Response(proc_macro_api::bidirectional_protocol::msg::Response::ExpandMacro(res)),
                            };

                            resp_env.write::<_, C>(&mut stdout)?;
                        }

                        SpanMode::RustAnalyzer => {
                            let mut span_data_table =
                                deserialize_span_data_index_map(&span_data_table);

                            let def_site_span = span_data_table[def_site];
                            let call_site_span = span_data_table[call_site];
                            let mixed_site_span = span_data_table[mixed_site];

                            let macro_body_ts = macro_body.to_tokenstream_resolved(
                                CURRENT_API_VERSION,
                                &span_data_table,
                                |a, b| srv.join_spans(a, b).unwrap_or(b),
                            );
                            let attributes_ts = attributes.map(|it| {
                                it.to_tokenstream_resolved(
                                    CURRENT_API_VERSION,
                                    &span_data_table,
                                    |a, b| srv.join_spans(a, b).unwrap_or(b),
                                )
                            });

                            let (subreq_tx, subreq_rx) = unbounded::<proc_macro_srv::SubRequest>();
                            let (subresp_tx, subresp_rx) =
                                unbounded::<proc_macro_srv::SubResponse>();
                            let (result_tx, result_rx) = crossbeam_channel::bounded(1);

                            std::thread::scope(|scope| {
                                let srv_ref = &srv;

                                scope.spawn({
                                    let lib = lib.clone();
                                    let env = env.clone();
                                    let current_dir = current_dir.clone();
                                    let macro_name = macro_name.clone();
                                    move || {
                                        let res = srv_ref
                                            .expand_with_channels(
                                                lib,
                                                &env,
                                                current_dir,
                                                &macro_name,
                                                macro_body_ts,
                                                attributes_ts,
                                                def_site_span,
                                                call_site_span,
                                                mixed_site_span,
                                                subresp_rx,
                                                subreq_tx,
                                            )
                                            .map(|it| {
                                                (
                                                    msg::FlatTree::from_tokenstream(
                                                        it,
                                                        CURRENT_API_VERSION,
                                                        call_site_span,
                                                        &mut span_data_table,
                                                    ),
                                                    serialize_span_data_index_map(&span_data_table),
                                                )
                                            })
                                            .map(|(tree, span_data_table)| {
                                                proc_macro_api::bidirectional_protocol::msg::ExpandMacroExtended { tree, span_data_table }
                                            })
                                            .map_err(|e| e.into_string().unwrap_or_default())
                                            .map_err(msg::PanicMessage);
                                        let _ = result_tx.send(res);
                                    }
                                });

                                loop {
                                    if let Ok(res) = result_rx.try_recv() {
                                        let resp_env = Envelope {
                                            id: req.id,
                                            kind: Kind::Response,
                                            payload: Payload::Response(
                                                proc_macro_api::bidirectional_protocol::msg::Response::ExpandMacroExtended(res),
                                            ),
                                        };
                                        resp_env.write::<_, C>(&mut stdout).unwrap();
                                        break;
                                    }

                                    let subreq = match subreq_rx.recv() {
                                        Ok(r) => r,
                                        Err(_) => {
                                            break;
                                        }
                                    };

                                    let sub_env = Envelope {
                                        id: req.id,
                                        kind: Kind::SubRequest,
                                        payload: Payload::SubRequest(from_srv_req(subreq)),
                                    };
                                    sub_env.write::<_, C>(&mut stdout).unwrap();

                                    let resp_opt =
                                        Envelope::read::<_, C>(&mut stdin, &mut buf).unwrap();
                                    let resp = match resp_opt {
                                        Some(env) => env,
                                        None => {
                                            break;
                                        }
                                    };

                                    match (resp.kind, resp.payload) {
                                        (Kind::SubResponse, Payload::SubResponse(subresp)) => {
                                            let _ = subresp_tx.send(from_client_res(subresp));
                                        }
                                        _ => {
                                            break;
                                        }
                                    }
                                }
                            });
                        }
                    }
                }
            },
            _ => {}
        }
    }

    Ok(())
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
    let mut read_request = || msg::Request::read::<_, C>(&mut io::stdin().lock(), &mut buf);
    let write_response = |msg: msg::Response| msg.write::<_, C>(&mut io::stdout().lock());

    let env = EnvSnapshot::default();
    let srv = proc_macro_srv::ProcMacroSrv::new(&env);

    let mut span_mode = SpanMode::Id;

    while let Some(req) = read_request()? {
        let res = match req {
            msg::Request::ListMacros { dylib_path } => {
                msg::Response::ListMacros(srv.list_macros(&dylib_path).map(|macros| {
                    macros.into_iter().map(|(name, kind)| (name, macro_kind_to_api(kind))).collect()
                }))
            }
            msg::Request::ExpandMacro(task) => {
                let msg::ExpandMacro {
                    lib,
                    env,
                    current_dir,
                    data:
                        ExpandMacroData {
                            macro_body,
                            macro_name,
                            attributes,
                            has_global_spans:
                                ExpnGlobals { serialize: _, def_site, call_site, mixed_site },
                            span_data_table,
                        },
                } = *task;
                match span_mode {
                    SpanMode::Id => msg::Response::ExpandMacro({
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
                        )
                        .map(|it| {
                            msg::FlatTree::from_tokenstream_raw::<SpanTrans>(
                                it,
                                call_site,
                                CURRENT_API_VERSION,
                            )
                        })
                        .map_err(|e| e.into_string().unwrap_or_default())
                        .map_err(msg::PanicMessage)
                    }),
                    SpanMode::RustAnalyzer => msg::Response::ExpandMacroExtended({
                        let mut span_data_table = deserialize_span_data_index_map(&span_data_table);

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
                        )
                        .map(|it| {
                            (
                                msg::FlatTree::from_tokenstream(
                                    it,
                                    CURRENT_API_VERSION,
                                    call_site,
                                    &mut span_data_table,
                                ),
                                serialize_span_data_index_map(&span_data_table),
                            )
                        })
                        .map(|(tree, span_data_table)| msg::ExpandMacroExtended {
                            tree,
                            span_data_table,
                        })
                        .map_err(|e| e.into_string().unwrap_or_default())
                        .map_err(msg::PanicMessage)
                    }),
                }
            }
            msg::Request::ApiVersionCheck {} => msg::Response::ApiVersionCheck(CURRENT_API_VERSION),
            msg::Request::SetConfig(config) => {
                span_mode = config.span_mode;
                msg::Response::SetConfig(config)
            }
        };
        write_response(res)?
    }

    Ok(())
}

fn from_srv_req(
    value: proc_macro_srv::SubRequest,
) -> proc_macro_api::bidirectional_protocol::msg::SubRequest {
    match value {
        proc_macro_srv::SubRequest::SourceText { file_id, start, end } => {
            proc_macro_api::bidirectional_protocol::msg::SubRequest::SourceText {
                file_id: file_id.file_id().index(),
                start,
                end,
            }
        }
    }
}

fn from_client_res(
    value: proc_macro_api::bidirectional_protocol::msg::SubResponse,
) -> proc_macro_srv::SubResponse {
    match value {
        proc_macro_api::bidirectional_protocol::msg::SubResponse::SourceTextResult { text } => {
            proc_macro_srv::SubResponse::SourceTextResult { text }
        }
    }
}
