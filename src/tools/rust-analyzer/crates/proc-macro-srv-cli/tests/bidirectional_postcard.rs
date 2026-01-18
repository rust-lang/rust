#![cfg(feature = "sysroot-abi")]

mod common {
    pub(crate) mod utils;
}

use common::utils::{
    create_empty_token_tree, proc_macro_test_dylib_path, request_bidirectional, with_server,
};
use expect_test::expect;
use proc_macro_api::{
    ProtocolFormat::BidirectionalPostcardPrototype,
    bidirectional_protocol::{
        msg::{ExpandMacro, ExpandMacroData, ExpnGlobals, Request, Response},
        reject_subrequests,
    },
    legacy_protocol::msg::{PanicMessage, ServerConfig, SpanDataIndexMap, SpanMode},
    version::CURRENT_API_VERSION,
};

#[test]
fn test_bidi_version_check_bidirectional() {
    with_server(BidirectionalPostcardPrototype, |writer, reader| {
        let response =
            request_bidirectional(writer, reader, Request::ApiVersionCheck {}, reject_subrequests);

        match response {
            Response::ApiVersionCheck(version) => {
                assert_eq!(version, CURRENT_API_VERSION);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    });
}

#[test]
fn test_bidi_list_macros() {
    with_server(BidirectionalPostcardPrototype, |writer, reader| {
        let dylib_path = proc_macro_test_dylib_path();
        let response = request_bidirectional(
            writer,
            reader,
            Request::ListMacros { dylib_path },
            &reject_subrequests,
        );

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
fn test_bidi_list_macros_invalid_path() {
    with_server(BidirectionalPostcardPrototype, |writer, reader| {
        let response = request_bidirectional(
            writer,
            reader,
            Request::ListMacros { dylib_path: "/nonexistent/path/to/dylib.so".into() },
            reject_subrequests,
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
fn test_bidi_set_config() {
    with_server(BidirectionalPostcardPrototype, |writer, reader| {
        let config = ServerConfig { span_mode: SpanMode::Id };
        let response =
            request_bidirectional(writer, reader, Request::SetConfig(config), reject_subrequests);

        match response {
            Response::SetConfig(returned_config) => {
                assert_eq!(returned_config.span_mode, SpanMode::Id);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    });
}

#[test]
fn test_bidi_set_config_rust_analyzer_mode() {
    with_server(BidirectionalPostcardPrototype, |writer, reader| {
        let config = ServerConfig { span_mode: SpanMode::RustAnalyzer };
        let response =
            request_bidirectional(writer, reader, Request::SetConfig(config), reject_subrequests);

        match response {
            Response::SetConfig(returned_config) => {
                assert_eq!(returned_config.span_mode, SpanMode::RustAnalyzer);
            }
            other => panic!("unexpected response: {other:?}"),
        }
    });
}

#[test]
fn test_bidi_expand_macro_panic() {
    with_server(BidirectionalPostcardPrototype, |writer, reader| {
        let dylib_path = proc_macro_test_dylib_path();

        let mut span_data_table = SpanDataIndexMap::default();
        let macro_body =
            common::utils::create_empty_token_tree(CURRENT_API_VERSION, &mut span_data_table);

        let request1 = Request::ExpandMacro(Box::new(ExpandMacro {
            lib: dylib_path,
            env: vec![],
            current_dir: None,
            data: ExpandMacroData {
                macro_body,
                macro_name: "fn_like_panic".to_owned(),
                attributes: None,
                has_global_spans: ExpnGlobals { def_site: 0, call_site: 0, mixed_site: 0 },
                span_data_table: vec![],
            },
        }));

        let response = request_bidirectional(writer, reader, request1, reject_subrequests);

        match response {
            Response::ExpandMacro(Err(PanicMessage(msg))) => {
                assert!(msg.contains("fn_like_panic"), "panic message should mention macro name");
            }
            other => panic!("expected panic response, got: {other:?}"),
        }
    });
}

#[test]
fn test_bidi_basic_call_flow() {
    with_server(BidirectionalPostcardPrototype, |writer, reader| {
        let dylib_path = proc_macro_test_dylib_path();

        let response1 =
            request_bidirectional(writer, reader, Request::ApiVersionCheck {}, reject_subrequests);
        assert!(matches!(response1, Response::ApiVersionCheck(_)));

        let response2 = request_bidirectional(
            writer,
            reader,
            Request::SetConfig(ServerConfig { span_mode: SpanMode::Id }),
            reject_subrequests,
        );
        assert!(matches!(response2, Response::SetConfig(_)));

        let response3 = request_bidirectional(
            writer,
            reader,
            Request::ListMacros { dylib_path: dylib_path.clone() },
            reject_subrequests,
        );
        assert!(matches!(response3, Response::ListMacros(Ok(_))));
    });
}

#[test]
fn test_bidi_expand_nonexistent_macro() {
    with_server(BidirectionalPostcardPrototype, |writer, reader| {
        let dylib_path = proc_macro_test_dylib_path();

        let version_response =
            request_bidirectional(writer, reader, Request::ApiVersionCheck {}, reject_subrequests);
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
                has_global_spans: ExpnGlobals { def_site: 0, call_site: 0, mixed_site: 0 },
                span_data_table: vec![],
            },
        }));

        let response = request_bidirectional(writer, reader, expand_request, reject_subrequests);

        match response {
            Response::ExpandMacro(Err(PanicMessage(msg))) => {
                expect!["proc-macro `NonexistentMacro` is missing"].assert_eq(&msg)
            }
            other => panic!("expected error for nonexistent macro, got: {other:?}"),
        }
    });
}
