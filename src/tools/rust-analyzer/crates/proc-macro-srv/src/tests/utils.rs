//! utils used in proc-macro tests

use expect_test::Expect;
use span::{
    EditionedFileId, FileId, ROOT_ERASED_FILE_AST_ID, Span, SpanAnchor, SyntaxContext, TextRange,
};

use crate::{
    EnvSnapshot, ProcMacroSrv, SpanId, dylib, proc_macro_test_dylib_path, token_stream::TokenStream,
};

fn parse_string(call_site: SpanId, src: &str) -> TokenStream<SpanId> {
    TokenStream::from_str(src, call_site).unwrap()
}

fn parse_string_spanned(
    anchor: SpanAnchor,
    call_site: SyntaxContext,
    src: &str,
) -> TokenStream<Span> {
    TokenStream::from_str(src, Span { range: TextRange::default(), anchor, ctx: call_site })
        .unwrap()
}

pub fn assert_expand(
    macro_name: &str,
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    expect: Expect,
    expect_spanned: Expect,
) {
    assert_expand_impl(macro_name, ra_fixture, None, expect, expect_spanned);
}

pub fn assert_expand_attr(
    macro_name: &str,
    #[rust_analyzer::rust_fixture] ra_fixture: &str,
    attr_args: &str,
    expect: Expect,
    expect_spanned: Expect,
) {
    assert_expand_impl(macro_name, ra_fixture, Some(attr_args), expect, expect_spanned);
}

fn assert_expand_impl(
    macro_name: &str,
    input: &str,
    attr: Option<&str>,
    expect: Expect,
    expect_spanned: Expect,
) {
    let path = proc_macro_test_dylib_path();
    let expander = dylib::Expander::new(&temp_dir::TempDir::new().unwrap(), &path).unwrap();

    let def_site = SpanId(0);
    let call_site = SpanId(1);
    let mixed_site = SpanId(2);
    let input_ts = parse_string(call_site, input);
    let attr_ts = attr.map(|attr| parse_string(call_site, attr));
    let input_ts_string = format!("{input_ts:?}");
    let attr_ts_string = attr_ts.as_ref().map(|it| format!("{it:?}"));

    let res =
        expander.expand(macro_name, input_ts, attr_ts, def_site, call_site, mixed_site).unwrap();
    expect.assert_eq(&format!(
        "{input_ts_string}{}{}{}",
        if attr_ts_string.is_some() { "\n\n" } else { "" },
        attr_ts_string.unwrap_or_default(),
        if res.is_empty() { String::new() } else { format!("\n\n{res:?}") }
    ));

    let def_site = Span {
        range: TextRange::new(0.into(), 150.into()),
        anchor: SpanAnchor {
            file_id: EditionedFileId::current_edition(FileId::from_raw(41)),
            ast_id: ROOT_ERASED_FILE_AST_ID,
        },
        ctx: SyntaxContext::root(span::Edition::CURRENT),
    };
    let call_site = Span {
        range: TextRange::new(0.into(), 100.into()),
        anchor: SpanAnchor {
            file_id: EditionedFileId::current_edition(FileId::from_raw(42)),
            ast_id: ROOT_ERASED_FILE_AST_ID,
        },
        ctx: SyntaxContext::root(span::Edition::CURRENT),
    };
    let mixed_site = call_site;

    let fixture = parse_string_spanned(call_site.anchor, call_site.ctx, input);
    let attr = attr.map(|attr| parse_string_spanned(call_site.anchor, call_site.ctx, attr));
    let fixture_string = format!("{fixture:?}");
    let attr_string = attr.as_ref().map(|it| format!("{it:?}"));

    let res = expander.expand(macro_name, fixture, attr, def_site, call_site, mixed_site).unwrap();
    expect_spanned.assert_eq(&format!(
        "{fixture_string}{}{}{}",
        if attr_string.is_some() { "\n\n" } else { "" },
        attr_string.unwrap_or_default(),
        if res.is_empty() { String::new() } else { format!("\n\n{res:?}") }
    ));
}

pub(crate) fn list() -> Vec<String> {
    let dylib_path = proc_macro_test_dylib_path();
    let env = EnvSnapshot::default();
    let srv = ProcMacroSrv::new(&env);
    let res = srv.list_macros(&dylib_path).unwrap();
    res.into_iter().map(|(name, kind)| format!("{name} [{kind:?}]")).collect()
}
