//! The main loop of the proc-macro server.
use std::io;

use proc_macro_api::{
    legacy_protocol::{
        json::{read_json, write_json},
        msg::{
            self, ExpandMacroData, ExpnGlobals, Message, SpanMode, SpanTransformer,
            deserialize_span_data_index_map, serialize_span_data_index_map,
        },
    },
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
        ProtocolFormat::Json => run_json(),
        #[cfg(feature = "postcard")]
        ProtocolFormat::Postcard => unimplemented!(),
    }
}

fn run_json() -> io::Result<()> {
    fn macro_kind_to_api(kind: proc_macro_srv::ProcMacroKind) -> proc_macro_api::ProcMacroKind {
        match kind {
            proc_macro_srv::ProcMacroKind::CustomDerive => {
                proc_macro_api::ProcMacroKind::CustomDerive
            }
            proc_macro_srv::ProcMacroKind::Bang => proc_macro_api::ProcMacroKind::Bang,
            proc_macro_srv::ProcMacroKind::Attr => proc_macro_api::ProcMacroKind::Attr,
        }
    }

    let mut buf = String::new();
    let mut read_request = || msg::Request::read(read_json, &mut io::stdin().lock(), &mut buf);
    let write_response = |msg: msg::Response| msg.write(write_json, &mut io::stdout().lock());

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

                        let macro_body =
                            macro_body.to_subtree_unresolved::<SpanTrans>(CURRENT_API_VERSION);
                        let attributes = attributes
                            .map(|it| it.to_subtree_unresolved::<SpanTrans>(CURRENT_API_VERSION));

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
                            msg::FlatTree::new_raw::<SpanTrans>(
                                tt::SubtreeView::new(&it),
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

                        let macro_body =
                            macro_body.to_subtree_resolved(CURRENT_API_VERSION, &span_data_table);
                        let attributes = attributes.map(|it| {
                            it.to_subtree_resolved(CURRENT_API_VERSION, &span_data_table)
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
                                msg::FlatTree::new(
                                    tt::SubtreeView::new(&it),
                                    CURRENT_API_VERSION,
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
