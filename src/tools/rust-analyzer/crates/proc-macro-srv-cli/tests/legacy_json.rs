//! Integration tests for the proc-macro-srv-cli main loop.
//!
//! These tests exercise the full client-server RPC procedure using in-memory
//! channels without needing to spawn the actual server and client processes.

#![cfg(feature = "sysroot-abi")]

mod common {
    pub(crate) mod utils;
}

use common::utils::{create_empty_token_tree, proc_macro_test_dylib_path, request, with_server};
use expect_test::expect;
use proc_macro_api::{
    legacy_protocol::msg::{
        ExpandMacro, ExpandMacroData, ExpnGlobals, PanicMessage, Request, Response, ServerConfig,
        SpanDataIndexMap, SpanMode,
    },
    version::CURRENT_API_VERSION,
};

#[test]
fn test_version_check() {
    with_server(|writer, reader| {
        let response = request(writer, reader, Request::ApiVersionCheck {});

        match response {
            Response::ApiVersionCheck(version) => {
                assert_eq!(version, CURRENT_API_VERSION);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    });
}

#[test]
fn test_list_macros() {
    with_server(|writer, reader| {
        let dylib_path = proc_macro_test_dylib_path();
        let response = request(writer, reader, Request::ListMacros { dylib_path });

        let Response::ListMacros(Ok(macros)) = response else {
            panic!("expected successful ListMacros response");
        };

        let mut macro_list: Vec<_> =
            macros.iter().map(|(name, kind)| format!("{name} [{kind:?}]")).collect();
        macro_list.sort();
        let macro_list_str = macro_list.join("\n");

        expect![[r#"
            DeriveEmpty [CustomDerive]
            DeriveError [CustomDerive]
            DerivePanic [CustomDerive]
            DeriveReemit [CustomDerive]
            attr_error [Attr]
            attr_noop [Attr]
            attr_panic [Attr]
            fn_like_clone_tokens [Bang]
            fn_like_error [Bang]
            fn_like_mk_idents [Bang]
            fn_like_mk_literals [Bang]
            fn_like_noop [Bang]
            fn_like_panic [Bang]
            fn_like_span_join [Bang]
            fn_like_span_line_column [Bang]
            fn_like_span_ops [Bang]"#]]
        .assert_eq(&macro_list_str);
    });
}

#[test]
fn test_list_macros_invalid_path() {
    with_server(|writer, reader| {
        let response = request(
            writer,
            reader,
            Request::ListMacros { dylib_path: "/nonexistent/path/to/dylib.so".into() },
        );

        match response {
            Response::ListMacros(Err(e)) => assert!(
                e.starts_with("Cannot create expander for /nonexistent/path/to/dylib.so"),
                "{e}"
            ),
            other => panic!("expected error response, got: {other:?}"),
        }
    });
}

#[test]
fn test_set_config() {
    with_server(|writer, reader| {
        let config = ServerConfig { span_mode: SpanMode::Id };
        let response = request(writer, reader, Request::SetConfig(config));

        match response {
            Response::SetConfig(returned_config) => {
                assert_eq!(returned_config.span_mode, SpanMode::Id);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    });
}

#[test]
fn test_set_config_rust_analyzer_mode() {
    with_server(|writer, reader| {
        let config = ServerConfig { span_mode: SpanMode::RustAnalyzer };
        let response = request(writer, reader, Request::SetConfig(config));

        match response {
            Response::SetConfig(returned_config) => {
                assert_eq!(returned_config.span_mode, SpanMode::RustAnalyzer);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    });
}

#[test]
fn test_expand_macro_panic() {
    with_server(|writer, reader| {
        let dylib_path = proc_macro_test_dylib_path();

        let version_response = request(writer, reader, Request::ApiVersionCheck {});
        let Response::ApiVersionCheck(version) = version_response else {
            panic!("expected version check response");
        };

        let mut span_data_table = SpanDataIndexMap::default();
        let macro_body = create_empty_token_tree(version, &mut span_data_table);

        let expand_request = Request::ExpandMacro(Box::new(ExpandMacro {
            lib: dylib_path,
            env: vec![],
            current_dir: None,
            data: ExpandMacroData {
                macro_body,
                macro_name: "fn_like_panic".to_owned(),
                attributes: None,
                has_global_spans: ExpnGlobals {
                    serialize: version >= 3,
                    def_site: 0,
                    call_site: 0,
                    mixed_site: 0,
                },
                span_data_table: vec![],
            },
        }));

        let response = request(writer, reader, expand_request);

        match response {
            Response::ExpandMacro(Err(PanicMessage(msg))) => {
                assert!(msg.contains("fn_like_panic"), "panic message should mention the macro");
            }
            Response::ExpandMacro(Ok(_)) => {
                panic!("expected panic, but macro succeeded");
            }
            other => panic!("unexpected response: {other:?}"),
        }
    });
}

#[test]
fn test_basic_call_flow() {
    with_server(|writer, reader| {
        let dylib_path = proc_macro_test_dylib_path();

        let response1 = request(writer, reader, Request::ApiVersionCheck {});
        assert!(matches!(response1, Response::ApiVersionCheck(_)));

        let response2 =
            request(writer, reader, Request::SetConfig(ServerConfig { span_mode: SpanMode::Id }));
        assert!(matches!(response2, Response::SetConfig(_)));

        let response3 =
            request(writer, reader, Request::ListMacros { dylib_path: dylib_path.clone() });
        assert!(matches!(response3, Response::ListMacros(Ok(_))));
    });
}

#[test]
fn test_expand_nonexistent_macro() {
    with_server(|writer, reader| {
        let dylib_path = proc_macro_test_dylib_path();

        let version_response = request(writer, reader, Request::ApiVersionCheck {});
        let Response::ApiVersionCheck(version) = version_response else {
            panic!("expected version check response");
        };

        let mut span_data_table = SpanDataIndexMap::default();
        let macro_body = create_empty_token_tree(version, &mut span_data_table);

        let expand_request = Request::ExpandMacro(Box::new(ExpandMacro {
            lib: dylib_path,
            env: vec![],
            current_dir: None,
            data: ExpandMacroData {
                macro_body,
                macro_name: "NonexistentMacro".to_owned(),
                attributes: None,
                has_global_spans: ExpnGlobals {
                    serialize: version >= 3,
                    def_site: 0,
                    call_site: 0,
                    mixed_site: 0,
                },
                span_data_table: vec![],
            },
        }));

        let response = request(writer, reader, expand_request);

        match response {
            Response::ExpandMacro(Err(PanicMessage(msg))) => {
                expect!["proc-macro `NonexistentMacro` is missing"].assert_eq(&msg)
            }
            other => panic!("expected error for nonexistent macro, got: {other:?}"),
        }
    });
}
