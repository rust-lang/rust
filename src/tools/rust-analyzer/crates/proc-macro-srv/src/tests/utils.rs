//! utils used in proc-macro tests

use expect_test::Expect;
use proc_macro_api::msg::TokenId;
use span::{EditionedFileId, ErasedFileAstId, FileId, Span, SpanAnchor, SyntaxContextId};
use tt::TextRange;

use crate::{dylib, proc_macro_test_dylib_path, EnvSnapshot, ProcMacroSrv};

fn parse_string(call_site: TokenId, src: &str) -> crate::server_impl::TokenStream<TokenId> {
    crate::server_impl::TokenStream::with_subtree(
        syntax_bridge::parse_to_token_tree_static_span(span::Edition::CURRENT, call_site, src)
            .unwrap(),
    )
}

fn parse_string_spanned(
    anchor: SpanAnchor,
    call_site: SyntaxContextId,
    src: &str,
) -> crate::server_impl::TokenStream<Span> {
    crate::server_impl::TokenStream::with_subtree(
        syntax_bridge::parse_to_token_tree(span::Edition::CURRENT, anchor, call_site, src).unwrap(),
    )
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
    let input_ts = parse_string(call_site, input);
    let attr_ts = attr.map(|attr| parse_string(call_site, attr).into_subtree(call_site));

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

    let def_site = Span {
        range: TextRange::new(0.into(), 150.into()),
        anchor: SpanAnchor {
            file_id: EditionedFileId::current_edition(FileId::from_raw(41)),
            ast_id: ErasedFileAstId::from_raw(1),
        },
        ctx: SyntaxContextId::ROOT,
    };
    let call_site = Span {
        range: TextRange::new(0.into(), 100.into()),
        anchor: SpanAnchor {
            file_id: EditionedFileId::current_edition(FileId::from_raw(42)),
            ast_id: ErasedFileAstId::from_raw(2),
        },
        ctx: SyntaxContextId::ROOT,
    };
    let mixed_site = call_site;

    let fixture = parse_string_spanned(call_site.anchor, call_site.ctx, input);
    let attr = attr.map(|attr| {
        parse_string_spanned(call_site.anchor, call_site.ctx, attr).into_subtree(call_site)
    });

    let res = expander
        .expand(macro_name, fixture.into_subtree(call_site), attr, def_site, call_site, mixed_site)
        .unwrap();
    expect_s.assert_eq(&format!("{res:#?}"));
}

pub(crate) fn list() -> Vec<String> {
    let dylib_path = proc_macro_test_dylib_path();
    let env = EnvSnapshot::new();
    let mut srv = ProcMacroSrv::new(&env);
    let res = srv.list_macros(&dylib_path).unwrap();
    res.into_iter().map(|(name, kind)| format!("{name} [{kind:?}]")).collect()
}
