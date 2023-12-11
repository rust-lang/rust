//! utils used in proc-macro tests

use base_db::{
    span::{ErasedFileAstId, SpanAnchor, SpanData, SyntaxContextId},
    FileId,
};
use expect_test::Expect;
use proc_macro_api::msg::{SpanDataIndexMap, TokenId};
use tt::TextRange;

use crate::{dylib, proc_macro_test_dylib_path, ProcMacroSrv};

fn parse_string<S: tt::Span>(code: &str, call_site: S) -> Option<crate::server::TokenStream<S>> {
    // This is a bit strange. We need to parse a string into a token stream into
    // order to create a tt::SubTree from it in fixtures. `into_subtree` is
    // implemented by all the ABIs we have so we arbitrarily choose one ABI to
    // write a `parse_string` function for and use that. The tests don't really
    // care which ABI we're using as the `into_subtree` function isn't part of
    // the ABI and shouldn't change between ABI versions.
    crate::server::TokenStream::from_str(code, call_site).ok()
}

pub fn assert_expand(macro_name: &str, ra_fixture: &str, expect: Expect, expect_s: Expect) {
    assert_expand_impl(macro_name, ra_fixture, None, expect, expect_s);
}

pub fn assert_expand_attr(
    macro_name: &str,
    ra_fixture: &str,
    attr_args: &str,
    expect: Expect,
    expect_s: Expect,
) {
    assert_expand_impl(macro_name, ra_fixture, Some(attr_args), expect, expect_s);
}

fn assert_expand_impl(
    macro_name: &str,
    input: &str,
    attr: Option<&str>,
    expect: Expect,
    expect_s: Expect,
) {
    let path = proc_macro_test_dylib_path();
    let expander = dylib::Expander::new(&path).unwrap();

    let def_site = TokenId(0);
    let call_site = TokenId(1);
    let mixed_site = TokenId(2);
    let input_ts = parse_string(input, call_site).unwrap();
    let attr_ts = attr.map(|attr| parse_string(attr, call_site).unwrap().into_subtree(call_site));

    let res = expander
        .expand(
            macro_name,
            input_ts.into_subtree(call_site),
            attr_ts,
            def_site,
            call_site,
            mixed_site,
        )
        .unwrap();
    expect.assert_eq(&format!("{res:?}"));

    let def_site = SpanData {
        range: TextRange::new(0.into(), 150.into()),
        anchor: SpanAnchor {
            file_id: FileId::from_raw(41),
            ast_id: ErasedFileAstId::from_raw(From::from(1)),
        },
        ctx: SyntaxContextId::ROOT,
    };
    let call_site = SpanData {
        range: TextRange::new(52.into(), 77.into()),
        anchor: SpanAnchor {
            file_id: FileId::from_raw(42),
            ast_id: ErasedFileAstId::from_raw(From::from(2)),
        },
        ctx: SyntaxContextId::ROOT,
    };
    let mixed_site = call_site;

    let fixture = parse_string(input, call_site).unwrap();
    let attr = attr.map(|attr| parse_string(attr, call_site).unwrap().into_subtree(call_site));

    let res = expander
        .expand(macro_name, fixture.into_subtree(call_site), attr, def_site, call_site, mixed_site)
        .unwrap();
    expect_s.assert_eq(&format!("{res:?}"));
}

pub(crate) fn list() -> Vec<String> {
    let dylib_path = proc_macro_test_dylib_path();
    let mut srv = ProcMacroSrv::default();
    let res = srv.list_macros(&dylib_path).unwrap();
    res.into_iter().map(|(name, kind)| format!("{name} [{kind:?}]")).collect()
}
